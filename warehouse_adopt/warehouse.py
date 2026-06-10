"""On-disk warehouse: ``block_<i>/variant_<j>.pt`` + metadata + provenance.

Layout under ``root/``::

    root/
    ├── warehouse_metadata.json      # block specs, key lists, hashes, provenance
    ├── block_00/
    │   ├── variant_00.pt            # dict[str, Tensor]: exactly that block's keys
    │   └── variant_01.pt
    └── block_01/ …

Invariants
----------
* Variant ``0`` of every block is the *unperturbed clone* of the base
  checkpoint, so the original model is always reconstructable as the
  all-zeros configuration.
* Every ``save_variant`` appends a provenance entry
  ``(writer_id, round, content_hash)``; the *last* entry's hash always
  matches the bytes on disk (checked by :meth:`Warehouse.verify`).
* A variant file contains exactly its block's ``state_dict_keys`` —
  no more, no fewer (checked on save and on load).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from .decompose import BlockSpec

METADATA_FILENAME = "warehouse_metadata.json"
SCHEMA_VERSION = 1


@dataclass
class ProvenanceEntry:
    """One write event for a ``(block, variant)`` slice."""

    writer_id: str
    round: int
    content_hash: str


def tensor_content_hash(slice_state: Dict[str, torch.Tensor]) -> str:
    """Deterministic sha256 over a slice: key names, dtypes, shapes, bytes.

    Independent of file serialisation; two slices hash equal iff every
    tensor is bit-identical.
    """
    h = hashlib.sha256()
    for key in sorted(slice_state.keys()):
        t = slice_state[key].detach().cpu().contiguous()
        if t.dtype is torch.bfloat16:  # numpy has no bfloat16
            t = t.view(torch.int16)
        h.update(key.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return "sha256:" + h.hexdigest()


class Warehouse:
    """Persistent store of ``(block, variant)`` weight slices with provenance."""

    def __init__(
        self,
        root: str | Path,
        block_specs: List[BlockSpec],
        num_variants: int,
        *,
        exist_ok: bool = False,
    ) -> None:
        if num_variants < 1:
            raise ValueError(f"num_variants must be >= 1, got {num_variants}")
        if not block_specs:
            raise ValueError("block_specs must be non-empty")
        self.root = Path(root)
        if (self.root / METADATA_FILENAME).exists() and not exist_ok:
            raise FileExistsError(
                f"{self.root} already contains a warehouse "
                f"(pass exist_ok=True to overwrite metadata)."
            )
        self.root.mkdir(parents=True, exist_ok=True)
        self.block_specs = list(block_specs)
        self.num_variants = int(num_variants)
        self._provenance: Dict[str, List[dict]] = {}
        self._save_metadata()

    # ------------------------------------------------------------------ paths

    def variant_path(self, block_id: int, variant_id: int) -> Path:
        """Filesystem path of one ``(block, variant)`` slice."""
        return self.root / f"block_{block_id:02d}" / f"variant_{variant_id:02d}.pt"

    # ------------------------------------------------------------------ I/O

    def save_variant(
        self,
        block_id: int,
        variant_id: int,
        slice_state: Dict[str, torch.Tensor],
        *,
        writer_id: str = "local",
        round: int = 0,
    ) -> str:
        """Write a slice to disk, append a provenance entry, return its hash.

        The slice must contain exactly the block's ``state_dict_keys``.
        Tensors are stored on CPU with their original dtype.
        """
        spec = self.block_by_id(block_id)
        if not (0 <= variant_id < self.num_variants):
            raise IndexError(f"variant_id {variant_id} out of range [0, {self.num_variants})")
        expected = set(spec.state_dict_keys)
        got = set(slice_state.keys())
        if got != expected:
            raise ValueError(
                f"Slice for block {block_id} has wrong keys. "
                f"Missing: {sorted(expected - got)[:5]}; extra: {sorted(got - expected)[:5]}."
            )
        cpu_state = {k: v.detach().cpu().clone() for k, v in slice_state.items()}
        path = self.variant_path(block_id, variant_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cpu_state, path)
        content_hash = tensor_content_hash(cpu_state)
        self._provenance.setdefault(self._pv_key(block_id, variant_id), []).append(
            {"writer_id": str(writer_id), "round": int(round), "content_hash": content_hash}
        )
        self._save_metadata()
        return content_hash

    def load_variant(
        self,
        block_id: int,
        variant_id: int,
        map_location: str | torch.device = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """Read one slice from disk; validates the stored key set."""
        path = self.variant_path(block_id, variant_id)
        if not path.exists():
            raise FileNotFoundError(f"Variant not found on disk: {path}")
        state = torch.load(path, map_location=map_location, weights_only=True)
        expected = set(self.block_by_id(block_id).state_dict_keys)
        if set(state.keys()) != expected:
            raise RuntimeError(
                f"On-disk slice {path} key set disagrees with metadata for block {block_id}."
            )
        return state

    # ------------------------------------------------------------------ provenance

    @staticmethod
    def _pv_key(block_id: int, variant_id: int) -> str:
        return f"{block_id}/{variant_id}"

    def provenance(self, block_id: int, variant_id: int) -> List[ProvenanceEntry]:
        """Full write history of one slice (oldest first)."""
        raw = self._provenance.get(self._pv_key(block_id, variant_id), [])
        return [ProvenanceEntry(e["writer_id"], int(e["round"]), e["content_hash"]) for e in raw]

    def content_hash(self, block_id: int, variant_id: int) -> Optional[str]:
        """Hash recorded by the most recent write (None if never written)."""
        raw = self._provenance.get(self._pv_key(block_id, variant_id), [])
        return raw[-1]["content_hash"] if raw else None

    def variants_written_by(self, writer_id: str) -> List[Tuple[int, int]]:
        """All ``(block_id, variant_id)`` whose trace contains ``writer_id``."""
        out: List[Tuple[int, int]] = []
        for key, entries in self._provenance.items():
            if any(e["writer_id"] == writer_id for e in entries):
                b, v = key.split("/")
                out.append((int(b), int(v)))
        return sorted(out)

    def verify(self) -> List[str]:
        """Re-hash every written slice; return mismatch descriptions (empty = OK)."""
        problems: List[str] = []
        for key, entries in self._provenance.items():
            if not entries:
                continue
            b, v = (int(x) for x in key.split("/"))
            try:
                state = self.load_variant(b, v)
            except (FileNotFoundError, RuntimeError) as exc:
                problems.append(f"block {b} variant {v}: {exc}")
                continue
            actual = tensor_content_hash(state)
            if actual != entries[-1]["content_hash"]:
                problems.append(
                    f"block {b} variant {v}: hash mismatch "
                    f"(metadata {entries[-1]['content_hash'][:18]}…, disk {actual[:18]}…)"
                )
        return problems

    # ------------------------------------------------------------------ metadata

    def _save_metadata(self) -> None:
        meta = {
            "version": SCHEMA_VERSION,
            "num_variants": self.num_variants,
            "block_specs": [
                {
                    "id": s.id,
                    "name": s.name,
                    "state_dict_keys": s.state_dict_keys,
                    "num_params": s.num_params,
                }
                for s in self.block_specs
            ],
            "provenance": self._provenance,
        }
        (self.root / METADATA_FILENAME).write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, root: str | Path) -> "Warehouse":
        """Reconstruct a warehouse handle from a directory on disk."""
        root = Path(root)
        meta_path = root / METADATA_FILENAME
        if not meta_path.exists():
            raise FileNotFoundError(f"No {METADATA_FILENAME} under {root}; not a warehouse.")
        meta = json.loads(meta_path.read_text())
        specs = [
            BlockSpec(
                id=s["id"],
                name=s["name"],
                state_dict_keys=list(s["state_dict_keys"]),
                num_params=int(s["num_params"]),
            )
            for s in meta["block_specs"]
        ]
        wh = cls.__new__(cls)
        wh.root = root
        wh.block_specs = specs
        wh.num_variants = int(meta["num_variants"])
        wh._provenance = {k: list(v) for k, v in meta.get("provenance", {}).items()}
        return wh

    # ------------------------------------------------------------------ accessors

    @property
    def num_blocks(self) -> int:
        return len(self.block_specs)

    def block_ids(self) -> List[int]:
        return [s.id for s in self.block_specs]

    def num_configurations(self) -> int:
        """Distinct assemblable sub-models = num_variants ** num_blocks."""
        return self.num_variants ** self.num_blocks

    def block_by_id(self, block_id: int) -> BlockSpec:
        for spec in self.block_specs:
            if spec.id == block_id:
                return spec
        raise KeyError(f"No block with id={block_id}")

    def __repr__(self) -> str:
        return (
            f"Warehouse(root={str(self.root)!r}, blocks={self.num_blocks}, "
            f"variants={self.num_variants}, configurations={self.num_configurations()})"
        )


def initialize_variants(
    model: nn.Module,
    block_specs: List[BlockSpec],
    num_variants: int,
    warehouse: Warehouse,
    *,
    noise_std: float = 1e-3,
    seed: int = 0,
    writer_id: str = "init",
) -> None:
    """Materialise initial variants: variant 0 verbatim, others perturbed.

    ``variant_v[k] = base[k] + N(0, noise_std · mean|base[k]|)`` for
    ``v >= 1`` and floating-point tensors; integer tensors (e.g. BN
    ``num_batches_tracked``) are cloned verbatim. Deterministic: one
    CPU generator seeded with ``seed``, consumed in (block, variant,
    key) iteration order.
    """
    if num_variants != warehouse.num_variants:
        raise ValueError(
            f"num_variants ({num_variants}) != warehouse.num_variants ({warehouse.num_variants})"
        )
    base_state = model.state_dict()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for spec in block_specs:
        for variant_id in range(num_variants):
            slice_state: Dict[str, torch.Tensor] = {}
            for key in spec.state_dict_keys:
                w = base_state[key].detach().cpu()
                if variant_id == 0 or noise_std == 0.0 or not w.is_floating_point():
                    slice_state[key] = w.clone()
                    continue
                scale = noise_std * w.abs().mean().clamp_min(1e-8)
                noise = torch.randn(w.shape, generator=generator, dtype=w.dtype) * scale
                slice_state[key] = w + noise
            warehouse.save_variant(
                spec.id, variant_id, slice_state, writer_id=writer_id, round=0
            )
