"""Streaming ensemble inference with uncertainty (Welford accumulation).

For each sampled block configuration: assemble a sub-model, run the
user's ``infer_fn`` over the validation loader, fold the outputs into a
running mean/second-moment. Per-configuration outputs are never stored;
memory is ``O(output_size)`` regardless of how many configurations are
sampled.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .contract import UserModule, to_device
from .engine import make_engine
from .warehouse import Warehouse


@dataclass
class EnsemblePrediction:
    """Aggregated ensemble prediction with uncertainty.

    ``mean``/``std`` have shape ``(N, …)`` over the validation set;
    ``predictive_entropy`` (classification only) has shape ``(N,)`` and
    is the entropy of ``softmax(mean)`` — use it, not ``std``, for
    classification uncertainty.
    """

    mean: torch.Tensor
    std: torch.Tensor
    num_configurations: int
    predictive_entropy: Optional[torch.Tensor] = None
    block_configurations: List[Dict[int, int]] = field(default_factory=list)


def sample_configurations(
    warehouse: Warehouse, num_configurations: int, seed: int = 0
) -> List[Dict[int, int]]:
    """Sample distinct configurations (or enumerate all if few enough).

    If ``num_configurations >= warehouse.num_configurations()``, the
    full Cartesian product is returned in lexicographic order;
    otherwise that many distinct configurations are sampled uniformly
    without replacement using ``random.Random(seed)``.
    """
    block_ids = warehouse.block_ids()
    total = warehouse.num_configurations()
    if num_configurations >= total:
        return [
            dict(zip(block_ids, combo))
            for combo in itertools.product(
                range(warehouse.num_variants), repeat=warehouse.num_blocks
            )
        ]
    rng = random.Random(seed)
    seen: set = set()
    configs: List[Dict[int, int]] = []
    while len(configs) < num_configurations:
        combo = tuple(rng.randrange(warehouse.num_variants) for _ in block_ids)
        if combo in seen:
            continue
        seen.add(combo)
        configs.append(dict(zip(block_ids, combo)))
    return configs


def infer_with_uncertainty(
    warehouse: Warehouse,
    user: UserModule,
    *,
    mode: str = "offline",
    num_configurations: int = 16,
    task: str = "auto",
    device: str | torch.device = "cpu",
    pool_device: Optional[str | torch.device] = None,
    seed: int = 0,
    show_progress: bool = False,
) -> EnsemblePrediction:
    """Ensembled inference over sampled configurations → mean + uncertainty.

    ``task``: ``"classification"`` adds predictive entropy;
    ``"regression"``/``"segmentation"`` return only mean/std; ``"auto"``
    treats rank-2 outputs as classification. Works with either engine
    ``mode``; results are engine-independent. Configurations are
    evaluated **sequentially**, so the compute footprint is one
    sub-model (= the unwrapped model) regardless of how many are
    sampled; ``pool_device`` only moves the variant pool (None = mode
    default: offline→disk, online→cpu).
    """
    engine = make_engine(
        mode, warehouse, user.model_factory, device=device, pool_device=pool_device
    )
    _, val_loader = user.get_dataloaders()
    configurations = sample_configurations(warehouse, num_configurations, seed=seed)

    running_mean: Optional[torch.Tensor] = None
    running_m2: Optional[torch.Tensor] = None
    n_seen = 0
    try:
        for i, cfg in enumerate(configurations):
            model = engine.assemble(cfg, train=False)
            pieces: List[torch.Tensor] = []
            with torch.no_grad():
                for batch in val_loader:
                    out = user.infer_fn(model, to_device(batch, engine.device))
                    pieces.append(out.detach().cpu())
            full = torch.cat(pieces, dim=0)
            if mode == "offline":
                del model
                if engine.device.type == "cuda":
                    torch.cuda.empty_cache()

            n_seen += 1
            if running_mean is None:
                running_mean = full.clone()
                running_m2 = torch.zeros_like(full)
            else:
                delta = full - running_mean
                running_mean = running_mean + delta / n_seen
                delta2 = full - running_mean
                running_m2 = running_m2 + delta * delta2
            if show_progress:
                print(f"[{i + 1}/{len(configurations)}] cfg={cfg} -> {tuple(full.shape)}")
    finally:
        engine.close()

    assert running_mean is not None and running_m2 is not None
    variance = running_m2 / max(n_seen - 1, 1)
    std = variance.clamp_min(0).sqrt()

    pred = EnsemblePrediction(
        mean=running_mean,
        std=std,
        num_configurations=n_seen,
        block_configurations=configurations,
    )
    is_classification = task == "classification" or (task == "auto" and running_mean.dim() == 2)
    if is_classification:
        probs = F.softmax(running_mean, dim=-1)
        pred.predictive_entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    return pred
