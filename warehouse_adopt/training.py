"""Diversification training: canonical step + adaptive consistency weighting.

The canonical diversification step (both engines MUST follow this exact
order so they stay numerically equivalent):

1. Assemble the frozen *reference* configuration ``cfg_ref`` in eval
   mode; compute ``out_ref = infer_fn(model, batch)`` under
   ``torch.no_grad()``.
2. Assemble the *update* configuration ``cfg_update`` in train mode;
   compute ``out = infer_fn(model, batch)``.
3. ``loss = loss_fn(out, batch) + λ · consistency_fn(out, out_ref)``.
4. Take one step with a **fresh Adam optimizer** (persistent optimizer
   state across configurations would defeat block isolation).
5. ``engine.writeback(model, cfg_update)`` — only the update config's
   blocks are persisted.

λ comes from :class:`ConsistencyBalance` (adaptive EMA balancing with
linear warm-up; lag-1 so a loss never scales its own gradient).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from .contract import UserModule, to_device
from .engine import AssemblyEngine, make_engine
from .warehouse import Warehouse


# ---------------------------------------------------------------------------
# Consistency balancing (ported from SASWISE src/adapters/finetune.py)
# ---------------------------------------------------------------------------


@dataclass
class ConsistencyBalance:
    """Schedule for the consistency weight λ in ``L = L_task + λ·L_cons``.

    Adaptive mode (default) recomputes λ each step so the consistency
    term contributes ``target_ratio`` of the total loss magnitude::

        λ = (target_ratio / (1 − target_ratio)) · EMA[L_task] / EMA[L_cons]

    clipped to ``[min_weight, max_weight]``, with a linear warm-up from
    0 over the first ``warmup_steps`` steps. Fixed mode uses
    ``fixed_weight`` (warm-up still applies).
    """

    adaptive: bool = True
    target_ratio: float = 0.1
    fixed_weight: float = 0.1
    warmup_steps: int = 100
    ema_decay: float = 0.9
    min_weight: float = 0.0
    max_weight: float = 100.0

    def __post_init__(self) -> None:
        if self.adaptive and not (0.0 < self.target_ratio < 1.0):
            raise ValueError(f"target_ratio must be in (0, 1), got {self.target_ratio}")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in [0, 1), got {self.ema_decay}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.min_weight < 0 or self.max_weight < self.min_weight:
            raise ValueError(
                f"Need 0 <= min_weight <= max_weight, got {self.min_weight}, {self.max_weight}"
            )


class ConsistencyBalancer:
    """Lag-1 EMA tracker producing the current λ.

    ``current_weight()`` reads the EMAs as they were *before* the most
    recent ``observe``, so step ``t``'s weight is derived from steps
    ``0..t-1`` — a loss never scales its own gradient.
    """

    def __init__(self, config: ConsistencyBalance) -> None:
        self.config = config
        self._task_ema: Optional[float] = None
        self._cons_ema: Optional[float] = None
        self._step = 0

    def current_weight(self) -> float:
        cfg = self.config
        if not cfg.adaptive:
            base = cfg.fixed_weight
        elif self._task_ema is None or self._cons_ema is None:
            base = 0.0
        else:
            ratio = cfg.target_ratio
            base = (ratio / (1.0 - ratio)) * (self._task_ema / max(self._cons_ema, 1e-12))
            base = max(cfg.min_weight, min(cfg.max_weight, base))
        warmup = 1.0 if cfg.warmup_steps == 0 else min(1.0, self._step / cfg.warmup_steps)
        return float(warmup * base)

    def observe(self, task_loss: float, cons_loss: float) -> None:
        decay = self.config.ema_decay
        if self._task_ema is None:
            self._task_ema = float(task_loss)
            self._cons_ema = float(cons_loss)
        else:
            self._task_ema = decay * self._task_ema + (1.0 - decay) * float(task_loss)
            self._cons_ema = decay * self._cons_ema + (1.0 - decay) * float(cons_loss)
        self._step += 1

    @property
    def step(self) -> int:
        return self._step


def default_consistency_fn(out_update: torch.Tensor, out_ref: torch.Tensor) -> torch.Tensor:
    """MSE between raw outputs — robust default across task types.

    For classification, KL of softmaxes is sharper; supply
    ``consistency_fn`` in the user module to override.
    """
    return F.mse_loss(out_update, out_ref)


# ---------------------------------------------------------------------------
# Canonical step
# ---------------------------------------------------------------------------


def sample_distinct_configs(
    block_ids: List[int], num_variants: int, rng: random.Random
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Two random full configurations differing in ≥1 block (update, reference).

    Consumes ``rng`` in a fixed order so a seeded ``random.Random``
    reproduces the schedule exactly.
    """
    cfg_update = {bid: rng.randrange(num_variants) for bid in block_ids}
    cfg_ref = {bid: rng.randrange(num_variants) for bid in block_ids}
    if num_variants > 1 and cfg_update == cfg_ref:
        flip = rng.choice(block_ids)
        cfg_ref[flip] = (cfg_ref[flip] + 1) % num_variants
    return cfg_update, cfg_ref


@dataclass
class StepStats:
    """Telemetry of one diversification step."""

    step: int
    total_loss: float
    task_loss: float
    consistency_loss: float
    weight: float
    cfg_update: Dict[int, int] = field(default_factory=dict)
    cfg_ref: Dict[int, int] = field(default_factory=dict)


def diversification_step(
    engine: AssemblyEngine,
    user: UserModule,
    batch,
    cfg_update: Dict[int, int],
    cfg_ref: Dict[int, int],
    *,
    lr: float,
    weight: float,
    step: int = 0,
    writer_id: str = "local",
) -> StepStats:
    """One canonical diversification step (see module docstring for the order).

    ``batch`` must already be on the engine's device.
    """
    consistency_fn = user.consistency_fn or default_consistency_fn

    # (1) Frozen reference forward.
    ref_model = engine.assemble(cfg_ref, train=False)
    with torch.no_grad():
        out_ref = user.infer_fn(ref_model, batch).detach()
    del ref_model

    # (2) Update forward.
    model = engine.assemble(cfg_update, train=True)
    out = user.infer_fn(model, batch)

    # (3) Combined loss.
    task_loss = user.loss_fn(out, batch)
    cons_loss = consistency_fn(out, out_ref)
    loss = task_loss + weight * cons_loss

    # (4) Fresh optimizer per step. foreach=False keeps step-time temporaries
    # O(largest tensor) instead of ~2× param bytes (identical arithmetic; the
    # resource model in agent_specs/00 §6.1 assumes this).
    optim = torch.optim.Adam(model.parameters(), lr=lr, foreach=False)
    optim.zero_grad()
    loss.backward()
    optim.step()

    # (5) Persist only the update configuration's blocks.
    engine.writeback(model, cfg_update, writer_id=writer_id, round=step)

    return StepStats(
        step=step,
        total_loss=float(loss.item()),
        task_loss=float(task_loss.item()),
        consistency_loss=float(cons_loss.item()),
        weight=float(weight),
        cfg_update=dict(cfg_update),
        cfg_ref=dict(cfg_ref),
    )


def finetune_warehouse(
    warehouse: Warehouse,
    user: UserModule,
    *,
    mode: str = "offline",
    epochs: int = 1,
    max_steps: Optional[int] = None,
    lr: float = 1e-4,
    consistency: Union[ConsistencyBalance, float, None] = None,
    device: str | torch.device = "cpu",
    pool_device: Optional[str | torch.device] = None,
    seed: int = 0,
    writer_id: str = "local",
    flush_every: int = 0,
    log_every: int = 0,
) -> List[StepStats]:
    """Diversify every variant with the consistency loss; returns step stats.

    Deterministic for a fixed ``seed`` and dataloader order: the
    configuration schedule comes from ``random.Random(seed)`` only.
    ``consistency`` accepts a :class:`ConsistencyBalance`, a plain
    float (fixed λ, no warm-up), or None (adaptive default).
    ``pool_device`` selects where the variant pool lives (forwarded to
    :func:`make_engine`; None = mode default: offline→disk, online→cpu).
    ``flush_every > 0`` flushes the online engine every that many
    steps; both engines are flushed/closed at the end.
    """
    if consistency is None:
        consistency = ConsistencyBalance()
    elif isinstance(consistency, (int, float)):
        consistency = ConsistencyBalance(
            adaptive=False, fixed_weight=float(consistency), warmup_steps=0
        )
    elif not isinstance(consistency, ConsistencyBalance):
        raise TypeError(
            f"consistency must be ConsistencyBalance, float, or None; got {type(consistency).__name__}"
        )
    balancer = ConsistencyBalancer(consistency)

    engine = make_engine(
        mode, warehouse, user.model_factory, device=device, pool_device=pool_device
    )
    train_loader, _ = user.get_dataloaders()
    block_ids = warehouse.block_ids()
    rng = random.Random(seed)

    stats: List[StepStats] = []
    step = 0
    try:
        for _epoch in range(epochs):
            for batch in train_loader:
                if max_steps is not None and step >= max_steps:
                    break
                weight = balancer.current_weight()
                cfg_update, cfg_ref = sample_distinct_configs(
                    block_ids, warehouse.num_variants, rng
                )
                batch_d = to_device(batch, engine.device)
                st = diversification_step(
                    engine,
                    user,
                    batch_d,
                    cfg_update,
                    cfg_ref,
                    lr=lr,
                    weight=weight,
                    step=step,
                    writer_id=writer_id,
                )
                balancer.observe(st.task_loss, st.consistency_loss)
                stats.append(st)
                step += 1
                if flush_every and step % flush_every == 0:
                    engine.flush()
                if log_every and step % log_every == 0:
                    print(
                        f"step {step:>6d}  loss={st.total_loss:.4f}  "
                        f"task={st.task_loss:.4f}  cons={st.consistency_loss:.4f}  "
                        f"λ={st.weight:.4f}"
                    )
            if max_steps is not None and step >= max_steps:
                break
    finally:
        engine.close()
    return stats


def recalibrate_bn(
    model: nn.Module,
    loader,
    user: UserModule,
    *,
    num_batches: int = 50,
    device: str | torch.device = "cpu",
    reset: bool = True,
) -> nn.Module:
    """Re-estimate norm-layer running statistics on ``loader`` (in place).

    Optional repair pass after assembling blocks whose BN statistics
    were trained under different configurations. Forwards
    ``num_batches`` batches in train mode under ``no_grad`` (only
    running stats change); ``reset=True`` zeroes the stats first.
    """
    device = torch.device(device)
    model = model.to(device)
    if reset:
        for m in model.modules():
            if isinstance(m, (nn.modules.batchnorm._BatchNorm,)):
                m.reset_running_stats()
    model.train(True)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            user.infer_fn(model, to_device(batch, device))
    model.train(False)
    return model
