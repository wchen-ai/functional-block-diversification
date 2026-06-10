"""Two assembly engines behind one ``mode={"offline","online"}`` switch.

Both engines expose the identical public API (:class:`AssemblyEngine`)
and are required to be **numerically equivalent** for a fixed seed and
schedule: given the same warehouse content, the same sequence of
``assemble``/``writeback`` calls, and the same batches, the final
warehouse content (and content hashes) must be identical.

* :class:`OfflineEngine` (passive, pool on **disk**) — every ``assemble``
  constructs a fresh model via ``model_factory()`` and loads the chosen
  variants from disk; ``writeback`` writes touched blocks to disk
  immediately. Nothing engine-owned stays resident between calls.

* :class:`OnlineEngine` (active, pool resident on ``pool_device``) — one
  resident model lives on the compute device; ``assemble`` swaps only the
  blocks whose active variant differs, via ``load_state_dict(strict=False)``
  on that block's cached slice (in-place ``copy_``, so hooks and
  parametrizations are preserved); ``writeback`` updates the in-RAM slice
  cache and marks slices dirty; ``flush`` persists dirty slices.
  ``pool_device="cpu"`` (default) keeps the variant pool in host RAM —
  each swap moves one block across PCIe. ``pool_device="cuda"`` keeps the
  pool on the GPU — swaps are device-to-device and cross no bus, at the
  cost of the pool's VRAM (≈ N × model bytes).

Every engine maintains an :class:`EngineStats` counter of *logical* byte
movements (pool→active-model, active-model→pool, disk reads/writes, block
swaps). These are exact, deterministic, and are asserted against the
resource model of ``agent_specs/00_core_warehouse.md`` §6.1 by
``warehouse_adopt/bench_resource.py`` and ``tests/test_resource_model.py``.
A pool→device byte crosses PCIe exactly when the pool and the compute
device differ in type.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import nn

from .warehouse import Warehouse


def state_dict_bytes(state: Dict[str, torch.Tensor]) -> int:
    """Exact byte size of a state-dict slice (numel × element_size per tensor)."""
    return sum(t.numel() * t.element_size() for t in state.values())


@dataclass
class EngineStats:
    """Deterministic counters of logical byte movement (see module docstring).

    ``pool_to_device_bytes`` — slice bytes delivered into the active model by
    ``assemble`` (offline: the full model per assemble; online: only swapped
    blocks). ``device_to_pool_bytes`` — bytes captured from the active model
    by ``writeback``. ``disk_read/write_bytes`` — bytes that actually hit the
    filesystem. ``block_swaps`` — number of per-block activations performed.
    """

    assembles: int = 0
    block_swaps: int = 0
    pool_to_device_bytes: int = 0
    device_to_pool_bytes: int = 0
    disk_read_bytes: int = 0
    disk_write_bytes: int = 0

    def reset(self) -> None:
        for f in ("assembles", "block_swaps", "pool_to_device_bytes",
                  "device_to_pool_bytes", "disk_read_bytes", "disk_write_bytes"):
            setattr(self, f, 0)

    def snapshot(self) -> dict:
        return {
            "assembles": self.assembles,
            "block_swaps": self.block_swaps,
            "pool_to_device_bytes": self.pool_to_device_bytes,
            "device_to_pool_bytes": self.device_to_pool_bytes,
            "disk_read_bytes": self.disk_read_bytes,
            "disk_write_bytes": self.disk_write_bytes,
        }


class AssemblyEngine(abc.ABC):
    """Common interface of the offline and online assembly engines."""

    def __init__(
        self,
        warehouse: Warehouse,
        model_factory: Callable[[], nn.Module],
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.warehouse = warehouse
        self.model_factory = model_factory
        self.device = torch.device(device)
        self.stats = EngineStats()

    # -------------------------------------------------------------- interface

    @abc.abstractmethod
    def assemble(self, block_config: Dict[int, int], train: bool = False) -> nn.Module:
        """Return a model on ``self.device`` loaded with ``block_config``.

        ``block_config`` maps every block id to a variant id (full
        cover, validated). ``train`` sets the returned model's mode.
        """

    @abc.abstractmethod
    def writeback(
        self,
        model: nn.Module,
        block_config: Dict[int, int],
        *,
        blocks: Optional[List[int]] = None,
        writer_id: str = "local",
        round: int = 0,
    ) -> None:
        """Persist the model's current weights for ``blocks`` (default all).

        Offline: writes to disk immediately. Online: updates the slice
        cache and marks the slices dirty (persisted on ``flush``).
        """

    def flush(self) -> None:
        """Persist any deferred writes (no-op for the offline engine)."""

    def close(self) -> None:
        """Flush and release engine-held resources."""
        self.flush()

    def pool_resident_bytes(self) -> int:
        """Bytes of the variant pool held resident by the engine (0 = on disk)."""
        return 0

    # -------------------------------------------------------------- helpers

    def _validate_config(self, block_config: Dict[int, int]) -> None:
        expected = set(self.warehouse.block_ids())
        got = set(block_config.keys())
        if got != expected:
            raise ValueError(
                f"block_config must cover every block exactly once. "
                f"Missing: {sorted(expected - got)}; extra: {sorted(got - expected)}."
            )
        for b, v in block_config.items():
            if not (0 <= v < self.warehouse.num_variants):
                raise IndexError(
                    f"variant {v} for block {b} out of range [0, {self.warehouse.num_variants})"
                )

    def _slice_from_model(
        self, model: nn.Module, block_id: int, device: str | torch.device = "cpu"
    ) -> Dict[str, torch.Tensor]:
        sd = model.state_dict()
        keys = self.warehouse.block_by_id(block_id).state_dict_keys
        return {k: sd[k].detach().to(device, copy=True) for k in keys}


class OfflineEngine(AssemblyEngine):
    """Passive engine: pool on disk; fresh model + disk round-trips per call."""

    def assemble(self, block_config: Dict[int, int], train: bool = False) -> nn.Module:
        self._validate_config(block_config)
        model = self.model_factory()
        merged: Dict[str, torch.Tensor] = {}
        for block_id in sorted(block_config.keys()):
            slice_state = self.warehouse.load_variant(block_id, block_config[block_id])
            self.stats.disk_read_bytes += state_dict_bytes(slice_state)
            merged.update(slice_state)
        model.load_state_dict(merged, strict=True)
        self.stats.assembles += 1
        self.stats.block_swaps += len(block_config)
        self.stats.pool_to_device_bytes += state_dict_bytes(merged)
        model = model.to(self.device)
        model.train(train)
        return model

    def writeback(
        self,
        model: nn.Module,
        block_config: Dict[int, int],
        *,
        blocks: Optional[List[int]] = None,
        writer_id: str = "local",
        round: int = 0,
    ) -> None:
        self._validate_config(block_config)
        for block_id in sorted(blocks if blocks is not None else block_config.keys()):
            slice_state = self._slice_from_model(model, block_id, device="cpu")
            nbytes = state_dict_bytes(slice_state)
            self.stats.device_to_pool_bytes += nbytes
            self.stats.disk_write_bytes += nbytes
            self.warehouse.save_variant(
                block_id,
                block_config[block_id],
                slice_state,
                writer_id=writer_id,
                round=round,
            )


class OnlineEngine(AssemblyEngine):
    """Active engine: one resident model; pool resident on ``pool_device``."""

    def __init__(
        self,
        warehouse: Warehouse,
        model_factory: Callable[[], nn.Module],
        *,
        device: str | torch.device = "cpu",
        pool_device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(warehouse, model_factory, device=device)
        self.pool_device = torch.device(pool_device)
        if self.pool_device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("pool_device='cuda' requested but CUDA is not available.")
        self._model: Optional[nn.Module] = None
        self._active: Dict[int, int] = {}
        self._cache: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        self._dirty: Dict[Tuple[int, int], Tuple[str, int]] = {}

    def _resident(self) -> nn.Module:
        if self._model is None:
            self._model = self.model_factory().to(self.device)
            self._active = {}
        return self._model

    def _get_slice(self, block_id: int, variant_id: int) -> Dict[str, torch.Tensor]:
        key = (block_id, variant_id)
        if key not in self._cache:
            slice_state = self.warehouse.load_variant(
                block_id, variant_id, map_location=self.pool_device
            )
            self.stats.disk_read_bytes += state_dict_bytes(slice_state)
            self._cache[key] = slice_state
        return self._cache[key]

    def prefetch_pool(self) -> int:
        """Load every ``(block, variant)`` slice into the resident pool.

        Optional warm-up that makes the pool footprint deterministic
        (``pool_resident_bytes() == N × model bytes`` afterwards) and
        removes first-touch disk latency from the swap path. Returns the
        resident pool size in bytes.
        """
        for spec in self.warehouse.block_specs:
            for v in range(self.warehouse.num_variants):
                self._get_slice(spec.id, v)
        return self.pool_resident_bytes()

    def pool_resident_bytes(self) -> int:
        return sum(state_dict_bytes(s) for s in self._cache.values())

    def assemble(self, block_config: Dict[int, int], train: bool = False) -> nn.Module:
        self._validate_config(block_config)
        model = self._resident()
        for block_id in sorted(block_config.keys()):
            variant_id = block_config[block_id]
            if self._active.get(block_id) == variant_id:
                continue
            slice_state = self._get_slice(block_id, variant_id)
            # In-place copy_; never assigns new Parameter objects, so
            # hooks/parametrizations registered on the resident model survive.
            incompatible = model.load_state_dict(slice_state, strict=False)
            if incompatible.unexpected_keys:
                raise RuntimeError(
                    f"Slice for block {block_id} contains keys absent from the "
                    f"model: {incompatible.unexpected_keys[:5]}"
                )
            self.stats.block_swaps += 1
            self.stats.pool_to_device_bytes += state_dict_bytes(slice_state)
            self._active[block_id] = variant_id
        self.stats.assembles += 1
        model.train(train)
        return model

    def writeback(
        self,
        model: nn.Module,
        block_config: Dict[int, int],
        *,
        blocks: Optional[List[int]] = None,
        writer_id: str = "local",
        round: int = 0,
    ) -> None:
        self._validate_config(block_config)
        if model is not self._model:
            raise ValueError("OnlineEngine.writeback expects the resident model it assembled.")
        for block_id in sorted(blocks if blocks is not None else block_config.keys()):
            variant_id = block_config[block_id]
            if self._active.get(block_id) != variant_id:
                raise RuntimeError(
                    f"writeback for block {block_id} variant {variant_id} but the "
                    f"resident model holds variant {self._active.get(block_id)}."
                )
            slice_state = self._slice_from_model(model, block_id, device=self.pool_device)
            self.stats.device_to_pool_bytes += state_dict_bytes(slice_state)
            self._cache[(block_id, variant_id)] = slice_state
            self._dirty[(block_id, variant_id)] = (writer_id, round)

    def flush(self) -> None:
        for (block_id, variant_id) in sorted(self._dirty.keys()):
            writer_id, round_ = self._dirty[(block_id, variant_id)]
            slice_state = self._cache[(block_id, variant_id)]
            self.stats.disk_write_bytes += state_dict_bytes(slice_state)
            self.warehouse.save_variant(
                block_id,
                variant_id,
                slice_state,
                writer_id=writer_id,
                round=round_,
            )
        self._dirty.clear()

    def close(self) -> None:
        self.flush()
        self._model = None
        self._active = {}
        self._cache.clear()


ENGINE_MODES = ("offline", "online")
POOL_DEVICES = ("disk", "cpu", "cuda")


def make_engine(
    mode: str,
    warehouse: Warehouse,
    model_factory: Callable[[], nn.Module],
    *,
    device: str | torch.device = "cpu",
    pool_device: Optional[str | torch.device] = None,
) -> AssemblyEngine:
    """Factory for the engines; ``mode`` ∈ {"offline","online"}.

    ``pool_device`` selects where the N-variant pool lives:

    * ``None``    — mode default: offline → ``"disk"``, online → ``"cpu"``.
    * ``"disk"``  — offline engine only (its defining property).
    * ``"cpu"``   — online engine, pool in host RAM (per-swap PCIe transfer
      of one block when computing on GPU).
    * ``"cuda"``/``"cuda:K"``/`torch.device` — online engine, pool resident
      on the GPU (zero-bus swaps; costs ≈ N × model bytes of VRAM).

    Decision rule: scarce VRAM or few steps → offline/disk; abundant host
    RAM and many steps on one box → online/cpu; pool fits in VRAM and speed
    is critical → online/cuda. See agent_specs/00_core_warehouse.md §6.1.
    """
    if mode == "offline":
        if pool_device is not None and str(pool_device) != "disk":
            raise ValueError(
                f"OfflineEngine keeps its pool on disk; got pool_device={pool_device!r}. "
                "Use mode='online' for a cpu/cuda-resident pool."
            )
        return OfflineEngine(warehouse, model_factory, device=device)
    if mode == "online":
        if pool_device is None or str(pool_device) == "cpu":
            resolved: str | torch.device = "cpu"
        elif str(pool_device) == "disk":
            raise ValueError(
                "OnlineEngine keeps its pool resident; use mode='offline' for pool_device='disk'."
            )
        else:
            resolved = pool_device
        return OnlineEngine(warehouse, model_factory, device=device, pool_device=resolved)
    raise ValueError(f"Unknown engine mode {mode!r}; expected one of {ENGINE_MODES}.")
