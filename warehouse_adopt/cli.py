"""``warehouse-adopt`` CLI skeleton: ``init | finetune | infer``.

Shared by all versions (uncertainty / federated / active-learning);
version specs add subcommands rather than new entry points.

Examples
--------
::

    # 1. Decompose a checkpoint and materialise variants.
    warehouse-adopt init \
        --user-module mymodel.py --checkpoint base.pt \
        --output-dir runs/demo --decomposition balanced:6 --variants 4

    #    (or source-free auto-construction, no user module needed yet)
    warehouse-adopt init \
        --model torchvision:resnet18?num_classes=10 --checkpoint base.pt \
        --output-dir runs/demo --decomposition boundary:conv1+bn1,layer1,layer2,layer3,layer4,avgpool+fc

    # 2. Diversify (offline or online engine).
    warehouse-adopt finetune \
        --user-module mymodel.py --warehouse runs/demo/warehouse \
        --mode offline --epochs 2

    # 3. Ensembled inference + uncertainty.
    warehouse-adopt infer \
        --user-module mymodel.py --warehouse runs/demo/warehouse \
        --num-configurations 32 --task classification --output preds.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .contract import load_checkpoint_state_dict, load_user_module, resolve_model_factory
from .decompose import parse_decomposition
from .engine import ENGINE_MODES
from .inference import infer_with_uncertainty
from .training import ConsistencyBalance, finetune_warehouse
from .warehouse import Warehouse, initialize_variants


def cmd_init(args: argparse.Namespace) -> int:
    """Decompose a checkpoint and materialise initial variants on disk."""
    if bool(args.user_module) == bool(args.model):
        raise SystemExit("init: pass exactly one of --user-module or --model.")
    if args.user_module:
        user = load_user_module(args.user_module)
        model = user.model_factory()
        if args.checkpoint:
            user.load_base_checkpoint(model, args.checkpoint)
    else:
        factory = resolve_model_factory(args.model)
        model = factory()
        if args.checkpoint:
            model.load_state_dict(load_checkpoint_state_dict(args.checkpoint), strict=True)
    if not args.checkpoint:
        print("WARNING: no --checkpoint given; using the factory's random init as the base.")

    decompose_fn, desc = parse_decomposition(args.decomposition)
    block_specs = decompose_fn(model)
    print(f"Decomposed into {len(block_specs)} blocks ({desc}):")
    for spec in block_specs:
        print(
            f"  block {spec.id:>2d}  {spec.name:<24s} "
            f"{spec.num_params:>12d} elements / {len(spec.state_dict_keys)} tensors"
        )

    warehouse_root = Path(args.output_dir) / "warehouse"
    warehouse = Warehouse(warehouse_root, block_specs, num_variants=args.variants)
    initialize_variants(
        model, block_specs, args.variants, warehouse, noise_std=args.noise_std, seed=args.seed
    )
    print(f"Initialised {warehouse}")
    return 0


def cmd_finetune(args: argparse.Namespace) -> int:
    """Diversify the warehouse with the consistency loss."""
    user = load_user_module(args.user_module)
    warehouse = Warehouse.load(args.warehouse)
    print(f"Loaded {warehouse}")
    consistency = (
        float(args.fixed_weight)
        if args.consistency_mode == "fixed"
        else ConsistencyBalance(
            adaptive=True,
            target_ratio=args.target_ratio,
            warmup_steps=args.warmup_steps,
            ema_decay=args.ema_decay,
        )
    )
    stats = finetune_warehouse(
        warehouse,
        user,
        mode=args.mode,
        epochs=args.epochs,
        max_steps=args.max_steps,
        lr=args.lr,
        consistency=consistency,
        device=args.device,
        pool_device=args.pool_device,
        seed=args.seed,
        flush_every=args.flush_every,
        log_every=args.log_every,
    )
    print(f"Fine-tuning complete: {len(stats)} steps ({args.mode} engine).")
    return 0


def cmd_infer(args: argparse.Namespace) -> int:
    """Run ensembled inference; save mean/std/entropy to ``--output``."""
    user = load_user_module(args.user_module)
    warehouse = Warehouse.load(args.warehouse)
    pred = infer_with_uncertainty(
        warehouse,
        user,
        mode=args.mode,
        num_configurations=args.num_configurations,
        task=args.task,
        device=args.device,
        pool_device=args.pool_device,
        seed=args.seed,
        show_progress=not args.quiet,
    )
    payload = {
        "mean": pred.mean,
        "std": pred.std,
        "num_configurations": pred.num_configurations,
        "block_configurations": pred.block_configurations,
    }
    if pred.predictive_entropy is not None:
        payload["predictive_entropy"] = pred.predictive_entropy
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"Saved predictions to {out}  (mean {tuple(pred.mean.shape)}, "
          f"{pred.num_configurations} configurations)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="warehouse-adopt", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="Decompose a checkpoint and materialise variants.")
    p.add_argument("--user-module", help="Path to the 5-callable user .py file.")
    p.add_argument("--model", help="Auto-construction spec: torchvision:<n> | timm:<n> | hf:<repo>.")
    p.add_argument("--checkpoint", help="Path to the pretrained state-dict file.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--decomposition", default="balanced:6",
                   help="balanced:K | top-level | manual:<re>;<re>;… | boundary:p1+p2,p3,…")
    p.add_argument("--variants", type=int, default=4)
    p.add_argument("--noise-std", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("finetune", help="Diversify the warehouse (offline/online engine).")
    p.add_argument("--user-module", required=True)
    p.add_argument("--warehouse", required=True)
    p.add_argument("--mode", choices=ENGINE_MODES, default="offline")
    p.add_argument("--pool-device", default=None,
                   help="Where the variant pool lives: disk (offline only), cpu, cuda[:K]. "
                        "Default: offline→disk, online→cpu. See agent_specs/00 §6.1.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--flush-every", type=int, default=0,
                   help="Online engine: persist dirty slices every N steps (0 = at end only).")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--consistency-mode", choices=["adaptive", "fixed"], default="adaptive")
    p.add_argument("--target-ratio", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--ema-decay", type=float, default=0.9)
    p.add_argument("--fixed-weight", type=float, default=0.1)
    p.set_defaults(func=cmd_finetune)

    p = sub.add_parser("infer", help="Ensembled inference + uncertainty.")
    p.add_argument("--user-module", required=True)
    p.add_argument("--warehouse", required=True)
    p.add_argument("--mode", choices=ENGINE_MODES, default="offline")
    p.add_argument("--pool-device", default=None,
                   help="Where the variant pool lives: disk (offline only), cpu, cuda[:K]. "
                        "Default: offline→disk, online→cpu.")
    p.add_argument("--num-configurations", type=int, default=16)
    p.add_argument("--task", choices=["classification", "regression", "segmentation", "auto"],
                   default="auto")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="predictions.pt")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_infer)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
