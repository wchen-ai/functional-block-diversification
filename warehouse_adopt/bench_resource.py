"""Resource-model demonstration: measure, don't assert.

Validates agent_specs/00_core_warehouse.md §6.1 by driving a fixed,
deterministic assemble/train schedule through the real engines in three
pool placements — offline/disk, online/cpu, online/cuda — at inference
(A = 1 active sub-model) and diversification training, and checking:

* engine byte counters (pool→device, device→pool, disk read/write,
  block swaps) **equal** the closed-form predictions exactly;
* the resident pool measures exactly N × model bytes
  (``OnlineEngine.pool_resident_bytes`` after ``prefetch_pool``);
* on CUDA, ``torch.cuda.max_memory_allocated`` falls inside the
  formula bounds — in particular, training peaks at
  weights + grads + 2 Adam moments (≈ 4 × param bytes) **without** a
  second resident sub-model (the canonical step order time-multiplexes
  the frozen reference M_j before the update model M_k; a concurrent
  implementation would peak ≥ 5 × param bytes).

Run it:

    python -m warehouse_adopt.bench_resource          # add --device cuda on a GPU box

Pytest coverage of the same checks lives in tests/test_resource_model.py.
Host RSS is printed for context only (peak RSS is process-monotonic and
allocator-cached, so it is not asserted; the *pool's* host footprint is
asserted exactly via pool_resident_bytes instead).
"""

from __future__ import annotations

import argparse
import resource
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contract import UserModule
from .decompose import decompose_boundary
from .engine import AssemblyEngine, make_engine, state_dict_bytes
from .training import diversification_step
from .warehouse import Warehouse, initialize_variants

# --------------------------------------------------------------- bench setup

NUM_VARIANTS = 3          # N
BATCH = 128
LR = 1e-3
WEIGHT = 0.1

# Fixed deterministic schedules (B=3 blocks, N=3 variants).
INFER_CONFIGS: List[Dict[int, int]] = [
    {0: 0, 1: 0, 2: 0},
    {0: 1, 1: 1, 2: 1},
    {0: 2, 1: 2, 2: 2},
    {0: 0, 1: 1, 2: 2},
]
TRAIN_SCHEDULE: List[Tuple[Dict[int, int], Dict[int, int]]] = [
    # (cfg_update, cfg_ref) — varying overlap so swap accounting is exercised
    ({0: 0, 1: 1, 2: 2}, {0: 1, 1: 0, 2: 2}),
    ({0: 0, 1: 1, 2: 2}, {0: 2, 1: 1, 2: 0}),   # ref shares block 1 with update
    ({0: 1, 1: 2, 2: 0}, {0: 0, 1: 1, 2: 2}),
    ({0: 1, 1: 2, 2: 0}, {0: 1, 1: 0, 2: 1}),   # ref shares block 0 with update
]


class BenchNet(nn.Module):
    """MLP+BN sized so weights (~5.3 MB fp32) dominate allocator rounding."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def model_factory() -> nn.Module:
    return BenchNet()


def make_user() -> UserModule:
    def infer_fn(model: nn.Module, batch) -> torch.Tensor:
        x, _ = batch
        return model(x)

    def loss_fn(out: torch.Tensor, batch) -> torch.Tensor:
        _, y = batch
        return F.cross_entropy(out, y)

    return UserModule(
        model_factory=model_factory,
        load_base_checkpoint=lambda m, p: None,
        get_dataloaders=lambda: (None, None),     # the bench owns its batches
        infer_fn=infer_fn,
        loss_fn=loss_fn,
    )


def make_batch(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(BATCH, 1, 28, 28, generator=g)
    y = torch.randint(0, 10, (BATCH,), generator=g)
    return x.to(device), y.to(device)


def build_bench_warehouse(root: Path) -> Warehouse:
    torch.manual_seed(13)
    base = model_factory()
    blocks = decompose_boundary(base, [["fc1", "bn1"], "fc2", "fc3"])
    wh = Warehouse(root, blocks, num_variants=NUM_VARIANTS)
    initialize_variants(base, blocks, NUM_VARIANTS, wh, noise_std=1e-2, seed=0)
    return wh


# ------------------------------------------------------------ size accounting


@dataclass
class Sizes:
    """Exact byte sizes derived from the BlockSpec key lists (§6.1 symbols)."""

    block_bytes: Dict[int, int]   # w_b per block
    model_bytes: int              # W  = Σ w_b   (params + persistent buffers)
    param_bytes: int              # Wp = trainable-parameter bytes only
    pool_bytes: int               # N·W
    num_blocks: int
    num_variants: int


def compute_sizes(warehouse: Warehouse) -> Sizes:
    block_bytes = {
        s.id: state_dict_bytes(warehouse.load_variant(s.id, 0))
        for s in warehouse.block_specs
    }
    model_bytes = sum(block_bytes.values())
    param_bytes = sum(
        p.numel() * p.element_size() for p in model_factory().parameters()
    )
    return Sizes(
        block_bytes=block_bytes,
        model_bytes=model_bytes,
        param_bytes=param_bytes,
        pool_bytes=warehouse.num_variants * model_bytes,
        num_blocks=warehouse.num_blocks,
        num_variants=warehouse.num_variants,
    )


# ----------------------------------------------------------------- predictions


def predict_offline(sizes: Sizes) -> Dict[str, Dict[str, int]]:
    """Closed-form §6.1 counters for the offline/disk engine."""
    W, B = sizes.model_bytes, sizes.num_blocks
    K, T = len(INFER_CONFIGS), len(TRAIN_SCHEDULE)
    return {
        "infer": dict(assembles=K, block_swaps=K * B, pool_to_device_bytes=K * W,
                      device_to_pool_bytes=0, disk_read_bytes=K * W, disk_write_bytes=0),
        "train": dict(assembles=2 * T, block_swaps=2 * T * B,
                      pool_to_device_bytes=2 * T * W,        # ref + update, every step
                      device_to_pool_bytes=T * W,            # writeback of the update model
                      disk_read_bytes=2 * T * W, disk_write_bytes=T * W),
        "flush": dict(disk_write_bytes=0),                   # offline writes eagerly
    }


def predict_online(sizes: Sizes) -> Dict[str, Dict[str, int]]:
    """§6.1 counters for the online engine (pool resident, prefetched).

    pool→device is schedule-dependent: Σ over swaps of the touched block's
    bytes, replayed against the same active-variant map the engine keeps.
    Disk traffic during the phases is zero (pool prefetched); the deferred
    disk write equals the distinct dirty slices at flush.
    """
    W = sizes.model_bytes
    T = len(TRAIN_SCHEDULE)
    active: Dict[int, int] = {}

    def swap_bytes(cfg: Dict[int, int]) -> Tuple[int, int]:
        nbytes = swaps = 0
        for b in sorted(cfg):
            if active.get(b) != cfg[b]:
                nbytes += sizes.block_bytes[b]
                swaps += 1
                active[b] = cfg[b]
        return nbytes, swaps

    inf_bytes = inf_swaps = 0
    for cfg in INFER_CONFIGS:
        nb, sw = swap_bytes(cfg)
        inf_bytes += nb
        inf_swaps += sw

    tr_bytes = tr_swaps = 0
    dirty: set[Tuple[int, int]] = set()
    for cfg_update, cfg_ref in TRAIN_SCHEDULE:
        for cfg in (cfg_ref, cfg_update):                 # canonical order: ref first
            nb, sw = swap_bytes(cfg)
            tr_bytes += nb
            tr_swaps += sw
        dirty.update((b, v) for b, v in cfg_update.items())

    return {
        "infer": dict(assembles=len(INFER_CONFIGS), block_swaps=inf_swaps,
                      pool_to_device_bytes=inf_bytes, device_to_pool_bytes=0,
                      disk_read_bytes=0, disk_write_bytes=0),
        "train": dict(assembles=2 * T, block_swaps=tr_swaps,
                      pool_to_device_bytes=tr_bytes,
                      device_to_pool_bytes=T * W,
                      disk_read_bytes=0, disk_write_bytes=0),
        "flush": dict(disk_write_bytes=sum(sizes.block_bytes[b] for b, _ in dirty)),
    }


# ------------------------------------------------------------------ measuring


@dataclass
class PhaseMeasurement:
    counters: Dict[str, int]
    cuda_peak_bytes: Optional[int]    # None when device is not cuda
    cuda_base_bytes: Optional[int] = None   # allocated at phase start (pool, batch, libs)

    @property
    def cuda_delta_bytes(self) -> Optional[int]:
        """Phase-attributable VRAM: peak minus the standing baseline."""
        if self.cuda_peak_bytes is None or self.cuda_base_bytes is None:
            return None
        return self.cuda_peak_bytes - self.cuda_base_bytes


@dataclass
class ConfigMeasurement:
    mode: str
    pool: str
    device: str
    infer: PhaseMeasurement
    train: PhaseMeasurement
    flush: Dict[str, int]
    pool_resident_bytes: int
    sizes: Sizes
    cuda_base0_bytes: Optional[int] = None   # allocated after lib warmup, before the engine exists

    def infer_total(self) -> Optional[int]:
        """Engine-attributable VRAM at inference: peak − pre-engine baseline."""
        if self.infer.cuda_peak_bytes is None or self.cuda_base0_bytes is None:
            return None
        return self.infer.cuda_peak_bytes - self.cuda_base0_bytes

    def train_total(self) -> Optional[int]:
        """Engine-attributable VRAM during training: peak − pre-engine baseline."""
        if self.train.cuda_peak_bytes is None or self.cuda_base0_bytes is None:
            return None
        return self.train.cuda_peak_bytes - self.cuda_base0_bytes


def _cuda_peak_start(device: torch.device) -> Optional[int]:
    """Reset peak stats; return the standing baseline (allocated bytes)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        return int(torch.cuda.memory_allocated(device))
    return None


def _cuda_peak_read(device: torch.device) -> Optional[int]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return int(torch.cuda.max_memory_allocated(device))
    return None


def _cuda_warmup(device: torch.device, batch) -> None:
    """Pre-allocate library state (cuBLAS/cuDNN workspaces) outside the phases.

    Workspaces are allocated through the caching allocator at first use and
    held for the process lifetime; running one representative
    forward/backward/step beforehand moves them into the standing baseline so
    phase deltas measure only model-attributable memory.
    """
    if device.type != "cuda":
        return
    model = model_factory().to(device)
    user = make_user()
    out = user.infer_fn(model, batch)
    loss = user.loss_fn(out, batch)
    optim = torch.optim.Adam(model.parameters(), lr=LR, foreach=False)
    loss.backward()
    optim.step()
    del model, out, loss, optim
    torch.cuda.synchronize(device)


def measure_config(mode: str, pool_device: Optional[str], device: str,
                   root: Path) -> ConfigMeasurement:
    """Run the fixed schedules through a real engine and record everything."""
    dev = torch.device(device)
    wh = build_bench_warehouse(root)
    sizes = compute_sizes(wh)
    user = make_user()
    batch = make_batch(dev)
    _cuda_warmup(dev, batch)                         # library workspaces -> baseline
    base0 = _cuda_peak_start(dev)                    # pre-engine standing allocation

    engine: AssemblyEngine = make_engine(
        mode, wh, model_factory, device=dev, pool_device=pool_device
    )
    if mode == "online":
        engine.prefetch_pool()                       # deterministic pool footprint
    pool_resident = engine.pool_resident_bytes()
    engine.stats.reset()

    # ---- inference phase (A = 1) ----
    base = _cuda_peak_start(dev)
    for cfg in INFER_CONFIGS:
        model = engine.assemble(cfg, train=False)
        with torch.no_grad():
            user.infer_fn(model, batch)
        if mode == "offline":
            del model
    infer = PhaseMeasurement(engine.stats.snapshot(), _cuda_peak_read(dev), base)

    # ---- diversification training phase ----
    engine.stats.reset()
    base = _cuda_peak_start(dev)
    for step, (cfg_update, cfg_ref) in enumerate(TRAIN_SCHEDULE):
        diversification_step(engine, user, batch, cfg_update, cfg_ref,
                             lr=LR, weight=WEIGHT, step=step, writer_id="bench")
    train = PhaseMeasurement(engine.stats.snapshot(), _cuda_peak_read(dev), base)

    # ---- deferred persistence ----
    engine.stats.reset()
    engine.flush()
    flush = engine.stats.snapshot()
    engine.close()

    return ConfigMeasurement(mode=mode, pool=pool_device or ("disk" if mode == "offline" else "cpu"),
                             device=device, infer=infer, train=train, flush=flush,
                             pool_resident_bytes=pool_resident, sizes=sizes,
                             cuda_base0_bytes=base0)


# ------------------------------------------------------------------ verifying

COUNTER_KEYS = ("assembles", "block_swaps", "pool_to_device_bytes",
                "device_to_pool_bytes", "disk_read_bytes", "disk_write_bytes")

# CUDA bounds on engine-attributable VRAM (peak − pre-engine baseline; the
# baseline holds the batch and the cuBLAS/cuDNN workspaces from the warmup).
#
# Inference: one resident sub-model + transient activations ⇒ ≈ 1.3·W,
#   strictly < 2·W (a second resident model would cross that line).
# Training (canonical order): weights + grads + 2 Adam moments (4·Wp)
#   + saved activations (≈ 0.4·Wp at batch 128) + the single-tensor Adam
#   step transient (≈ largest tensor ≈ 0.6·Wp here) ⇒ ≈ 5.2·Wp measured.
#   A concurrent second resident sub-model adds a further +1.0·Wp ⇒ ≥ 6.2·Wp,
#   which the upper bound excludes. (W ≈ Wp here; buffers are ~8 KB of 5.3 MB.)
INFER_PEAK_LO, INFER_PEAK_HI = 0.95, 1.75      # × W,  + pool when the pool is on the GPU
TRAIN_PEAK_LO, TRAIN_PEAK_HI = 4.4, 5.95       # × Wp, + pool when the pool is on the GPU


def verify_config(meas: ConfigMeasurement) -> List[str]:
    """Compare measurement to prediction; return human-readable check lines.

    Raises AssertionError on the first mismatch.
    """
    pred = predict_offline(meas.sizes) if meas.mode == "offline" else predict_online(meas.sizes)
    lines: List[str] = []

    for phase, got in (("infer", meas.infer.counters), ("train", meas.train.counters)):
        for key in COUNTER_KEYS:
            want = pred[phase][key]
            assert got[key] == want, (
                f"{meas.mode}/{meas.pool} {phase}.{key}: measured {got[key]:,} != predicted {want:,}"
            )
        lines.append(f"  {phase:5s}: all 6 counters == prediction "
                     f"(pool->dev {got['pool_to_device_bytes']:,} B, "
                     f"dev->pool {got['device_to_pool_bytes']:,} B, "
                     f"disk r/w {got['disk_read_bytes']:,}/{got['disk_write_bytes']:,} B)")

    want_flush = pred["flush"]["disk_write_bytes"]
    assert meas.flush["disk_write_bytes"] == want_flush, (
        f"{meas.mode}/{meas.pool} flush.disk_write: {meas.flush['disk_write_bytes']:,} != {want_flush:,}"
    )
    lines.append(f"  flush: deferred disk write {meas.flush['disk_write_bytes']:,} B == prediction")

    if meas.mode == "online":
        assert meas.pool_resident_bytes == meas.sizes.pool_bytes, (
            f"pool_resident_bytes {meas.pool_resident_bytes:,} != N·W {meas.sizes.pool_bytes:,}"
        )
        lines.append(f"  pool : resident {meas.pool_resident_bytes:,} B == N·W exactly")
    else:
        assert meas.pool_resident_bytes == 0
        lines.append("  pool : on disk (0 resident bytes)")

    if meas.infer_total() is not None:
        W, Wp = meas.sizes.model_bytes, meas.sizes.param_bytes
        pool_on_gpu = meas.sizes.pool_bytes if meas.pool.startswith("cuda") else 0

        lo, hi = pool_on_gpu + INFER_PEAK_LO * W, pool_on_gpu + INFER_PEAK_HI * W
        got = meas.infer_total()
        assert lo <= got <= hi, (
            f"{meas.mode}/{meas.pool} inference VRAM {got:,} outside [{lo:,.0f}, {hi:,.0f}] "
            f"(= pool {pool_on_gpu:,} + [{INFER_PEAK_LO}, {INFER_PEAK_HI}]·W)"
        )
        lines.append(f"  VRAM : inference {(got - pool_on_gpu) / W:.2f}·W + pool({pool_on_gpu:,} B) "
                     f"— A=1, same footprint as the unwrapped model")

        lo, hi = pool_on_gpu + TRAIN_PEAK_LO * Wp, pool_on_gpu + TRAIN_PEAK_HI * Wp
        got = meas.train_total()
        assert lo <= got <= hi, (
            f"{meas.mode}/{meas.pool} training VRAM {got:,} outside [{lo:,.0f}, {hi:,.0f}] "
            f"(= pool {pool_on_gpu:,} + [{TRAIN_PEAK_LO}, {TRAIN_PEAK_HI}]·Wp)"
        )
        lines.append(f"  VRAM : training {(got - pool_on_gpu) / Wp:.2f}·Wp + pool({pool_on_gpu:,} B) "
                     "— weights+grads+2 Adam moments+act; NO concurrent second sub-model "
                     "(that would add +1.0·Wp ⇒ ≥ 6.2·Wp)")
    return lines


# ----------------------------------------------------------------------- main


def run_all(device: str = "cpu", include_cuda_pool: Optional[bool] = None) -> List[ConfigMeasurement]:
    """Measure + verify every applicable configuration; returns measurements."""
    if include_cuda_pool is None:
        include_cuda_pool = device.startswith("cuda")
    configs = [("offline", None), ("online", "cpu")]
    if include_cuda_pool:
        configs.append(("online", device))
    out: List[ConfigMeasurement] = []
    for mode, pool in configs:
        with tempfile.TemporaryDirectory() as td:
            meas = measure_config(mode, pool, device, Path(td) / "wh")
            label = f"{mode}/pool={meas.pool}/device={device}"
            print(f"\n[{label}]")
            for line in verify_config(meas):
                print(line)
            out.append(meas)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as td:
        sizes = compute_sizes(build_bench_warehouse(Path(td) / "wh"))
    print(f"BenchNet: W (params+buffers) = {sizes.model_bytes:,} B, "
          f"Wp (params) = {sizes.param_bytes:,} B, "
          f"B = {sizes.num_blocks} blocks {sorted(sizes.block_bytes.values())}, "
          f"N = {sizes.num_variants} variants, pool N·W = {sizes.pool_bytes:,} B")
    print(f"Schedules: {len(INFER_CONFIGS)} inference configs, "
          f"{len(TRAIN_SCHEDULE)} canonical training steps; device = {args.device}")

    run_all(args.device)

    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"\nHost peak RSS (informational): {rss_kb / 1024:.1f} MiB")
    print("ALL RESOURCE-MODEL ASSERTIONS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
