"""MNIST smoke test for the core warehouse — offline AND online engines.

Proves the core spec (agent_specs/00_core_warehouse.md) is implementable:

* init → finetune → infer runs end-to-end with the offline engine;
* the same pipeline runs with the online engine;
* the two engines are **numerically equivalent** for a fixed
  seed/schedule (identical final variant tensors, identical content
  hashes, identical ensemble predictions);
* variant 0 reproduces the base checkpoint exactly;
* norm-layer buffers travel with their block;
* tied weights are never split across blocks.

Runs on CPU in a few seconds. Uses the real MNIST under ``~/data`` (or
``$WAREHOUSE_ADOPT_DATA``); falls back to a deterministic synthetic
stand-in if torchvision/MNIST is unavailable so the equivalence proof
still executes.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

from warehouse_adopt import (
    UserModule,
    Warehouse,
    decompose_balanced,
    decompose_boundary,
    finetune_warehouse,
    infer_with_uncertainty,
    initialize_variants,
    make_engine,
)

BASE_SEED = 7
INIT_SEED = 0
TRAIN_SEED = 123
NOISE_STD = 1e-2
NUM_VARIANTS = 2
STEPS = 6
LR = 1e-3


# --------------------------------------------------------------------- model


class MLPBN(nn.Module):
    """Small MLP with BatchNorm so buffers (running stats) are exercised."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def model_factory() -> nn.Module:
    return MLPBN()


def make_base_model() -> nn.Module:
    torch.manual_seed(BASE_SEED)
    return model_factory()


# --------------------------------------------------------------------- data


def _mnist_loaders():
    from torchvision import datasets, transforms  # noqa: PLC0415

    root = Path(os.environ.get("WAREHOUSE_ADOPT_DATA", "~/data")).expanduser()
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    try:
        train_ds = datasets.MNIST(root, train=True, download=False, transform=tfm)
        val_ds = datasets.MNIST(root, train=False, download=False, transform=tfm)
    except RuntimeError:
        train_ds = datasets.MNIST(root, train=True, download=True, transform=tfm)
        val_ds = datasets.MNIST(root, train=False, download=True, transform=tfm)
    train = Subset(train_ds, range(512))
    val = Subset(val_ds, range(256))
    return (
        DataLoader(train, batch_size=64, shuffle=False, num_workers=0),
        DataLoader(val, batch_size=128, shuffle=False, num_workers=0),
    )


def _synthetic_loaders():
    gen = torch.Generator().manual_seed(99)
    x_train = torch.randn(512, 1, 28, 28, generator=gen)
    y_train = torch.randint(0, 10, (512,), generator=gen)
    x_val = torch.randn(256, 1, 28, 28, generator=gen)
    y_val = torch.randint(0, 10, (256,), generator=gen)
    return (
        DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=False),
        DataLoader(TensorDataset(x_val, y_val), batch_size=128, shuffle=False),
    )


def get_loaders():
    try:
        return _mnist_loaders()
    except Exception:  # torchvision missing or data unavailable
        return _synthetic_loaders()


def make_user() -> UserModule:
    train_loader, val_loader = get_loaders()

    def load_base_checkpoint(model: nn.Module, path: str) -> None:
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))

    def get_dataloaders():
        return train_loader, val_loader

    def infer_fn(model: nn.Module, batch) -> torch.Tensor:
        x, _ = batch
        return model(x)

    def loss_fn(out: torch.Tensor, batch) -> torch.Tensor:
        _, y = batch
        return F.cross_entropy(out, y)

    def consistency_fn(out_a: torch.Tensor, out_b: torch.Tensor) -> torch.Tensor:
        return F.kl_div(
            F.log_softmax(out_a, dim=-1), F.softmax(out_b, dim=-1), reduction="batchmean"
        )

    return UserModule(
        model_factory=model_factory,
        load_base_checkpoint=load_base_checkpoint,
        get_dataloaders=get_dataloaders,
        infer_fn=infer_fn,
        loss_fn=loss_fn,
        consistency_fn=consistency_fn,
    )


# ------------------------------------------------------------------ fixtures


def build_warehouse(root: Path) -> Warehouse:
    """Identical warehouse every call: fixed base weights + fixed init seed."""
    base = make_base_model()
    blocks = decompose_boundary(base, [["fc1", "bn1"], "fc2", "fc3"])
    wh = Warehouse(root, blocks, num_variants=NUM_VARIANTS)
    initialize_variants(base, blocks, NUM_VARIANTS, wh, noise_std=NOISE_STD, seed=INIT_SEED)
    return wh


def run_pipeline(root: Path, mode: str):
    """init → finetune → infer with the given engine mode."""
    wh = build_warehouse(root)
    user = make_user()
    stats = finetune_warehouse(
        wh,
        user,
        mode=mode,
        epochs=2,
        max_steps=STEPS,
        lr=LR,
        consistency=0.1,
        device="cpu",
        seed=TRAIN_SEED,
        writer_id=f"smoke-{mode}",
    )
    pred = infer_with_uncertainty(
        wh, user, mode=mode, num_configurations=4, task="classification",
        device="cpu", seed=0,
    )
    return wh, stats, pred


# --------------------------------------------------------------------- tests


@pytest.mark.parametrize("mode", ["offline", "online"])
def test_end_to_end(tmp_path: Path, mode: str) -> None:
    wh, stats, pred = run_pipeline(tmp_path / mode, mode)

    assert len(stats) == STEPS
    assert all(torch.isfinite(torch.tensor(s.total_loss)) for s in stats)

    # Prediction shapes and uncertainty channel.
    assert pred.mean.shape == (256, 10)
    assert pred.std.shape == (256, 10)
    assert pred.predictive_entropy is not None
    assert pred.predictive_entropy.shape == (256,)
    assert pred.num_configurations == 4
    assert torch.isfinite(pred.mean).all()

    # Provenance: init wrote every slice; training wrote at least one.
    for spec in wh.block_specs:
        for v in range(NUM_VARIANTS):
            trace = wh.provenance(spec.id, v)
            assert trace and trace[0].writer_id == "init"
    assert wh.variants_written_by(f"smoke-{mode}")

    # Metadata hashes match bytes on disk.
    assert wh.verify() == []

    # Norm-layer buffers travelled with their block: the trained
    # warehouse holds advanced BN step counters.
    bn_block = next(s for s in wh.block_specs if "bn1.num_batches_tracked" in s.state_dict_keys)
    counters = [
        wh.load_variant(bn_block.id, v)["bn1.num_batches_tracked"].item()
        for v in range(NUM_VARIANTS)
    ]
    assert max(counters) > 0


def test_engines_numerically_equivalent(tmp_path: Path) -> None:
    """Same seed/schedule ⇒ offline and online produce identical warehouses."""
    wh_off = build_warehouse(tmp_path / "off")
    wh_on = build_warehouse(tmp_path / "on")

    # Identical starting state (same base weights, same init noise).
    for spec in wh_off.block_specs:
        for v in range(NUM_VARIANTS):
            assert wh_off.content_hash(spec.id, v) == wh_on.content_hash(spec.id, v)

    user_off, user_on = make_user(), make_user()
    for wh, user, mode in ((wh_off, user_off, "offline"), (wh_on, user_on, "online")):
        finetune_warehouse(
            wh, user, mode=mode, epochs=2, max_steps=STEPS, lr=LR,
            consistency=0.1, device="cpu", seed=TRAIN_SEED, writer_id="eq",
        )

    # Bit-identical final content, slice by slice — and equal hashes.
    for spec in wh_off.block_specs:
        for v in range(NUM_VARIANTS):
            a = wh_off.load_variant(spec.id, v)
            b = wh_on.load_variant(spec.id, v)
            for key in spec.state_dict_keys:
                assert torch.equal(a[key], b[key]), (
                    f"block {spec.id} variant {v} key {key} differs between engines"
                )
            assert wh_off.content_hash(spec.id, v) == wh_on.content_hash(spec.id, v)

    # Identical ensemble predictions from either engine on either warehouse.
    pred_off = infer_with_uncertainty(
        wh_off, user_off, mode="offline", num_configurations=4,
        task="classification", device="cpu", seed=0,
    )
    pred_on = infer_with_uncertainty(
        wh_on, user_on, mode="online", num_configurations=4,
        task="classification", device="cpu", seed=0,
    )
    assert torch.equal(pred_off.mean, pred_on.mean)
    assert torch.equal(pred_off.std, pred_on.std)


def test_variant0_reproduces_base(tmp_path: Path) -> None:
    """The all-zeros configuration is the unperturbed base checkpoint."""
    wh = build_warehouse(tmp_path / "wh")
    base_state = make_base_model().state_dict()
    engine = make_engine("offline", wh, model_factory, device="cpu")
    model = engine.assemble({s.id: 0 for s in wh.block_specs})
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, base_state[key]), f"variant 0 differs at {key}"


def test_tied_weights_never_split() -> None:
    """Balanced decomposition keeps weight-tied modules in one block."""

    class TiedLM(nn.Module):
        """Embedding ↔ output-head weight tying, the classic shared-storage case."""

        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(20, 8)
            self.mid = nn.Linear(8, 8)
            self.head = nn.Linear(8, 20, bias=False)
            self.head.weight = self.embed.weight  # true tie: shared Parameter

        def forward(self, idx):
            return self.head(self.mid(self.embed(idx)))

    model = TiedLM()
    blocks = decompose_balanced(model, 3)
    owner = {}
    for spec in blocks:
        for key in spec.state_dict_keys:
            owner[key] = spec.id
    assert owner["embed.weight"] == owner["head.weight"], "tied keys split across blocks"
