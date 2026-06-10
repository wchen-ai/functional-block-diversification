"""Block decomposition — split a model's state_dict into blocks.

A *block* is a non-overlapping, exactly-covering subset of the model's
``state_dict`` keys (parameters AND persistent buffers). Keys are
ordered by ``named_modules()`` topological (registration) order, never
by dict insertion accidents.

Atomicity rules enforced here (the source-free gotchas):

* A *leaf module* (a module owning parameters/buffers directly) is an
  atomic group: its weight, bias, and norm-layer running statistics
  always travel in the same block.
* Tied/shared weights (two keys aliasing one storage, detected via
  ``Tensor.data_ptr()``) are merged into one atomic group and are never
  split across blocks; strategies that cannot honour this fail loudly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple, Union

import torch
from torch import nn


@dataclass
class BlockSpec:
    """A single block of the decomposition.

    Attributes
    ----------
    id : int
        Zero-based block index.
    name : str
        Human-readable label (e.g. ``"layer3"`` or ``"block_2"``).
    state_dict_keys : list[str]
        Ordered keys from ``model.state_dict()`` belonging to this block.
    num_params : int
        Total element count across all keys.
    """

    id: int
    name: str
    state_dict_keys: List[str] = field(default_factory=list)
    num_params: int = 0


# ------------------------------------------------------------------ internals


def _ordered_keys(model: nn.Module) -> List[str]:
    """State-dict keys in ``named_modules()`` topological order.

    Walks modules in registration order; within a module, parameters
    precede buffers, each in registration order. Non-persistent buffers
    are excluded (they are not part of ``state_dict``).
    """
    keys: List[str] = []
    for mod_name, mod in model.named_modules():
        prefix = f"{mod_name}." if mod_name else ""
        for pname, p in mod._parameters.items():
            if p is not None:
                keys.append(prefix + pname)
        for bname, b in mod._buffers.items():
            if b is None or bname in mod._non_persistent_buffers_set:
                continue
            keys.append(prefix + bname)
    sd_keys = set(model.state_dict().keys())
    if set(keys) != sd_keys:
        missing = sd_keys - set(keys)
        extra = set(keys) - sd_keys
        raise RuntimeError(
            "named_modules() traversal disagrees with state_dict(); "
            f"missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}. "
            "This model uses a non-standard state_dict hook; use decompose_manual."
        )
    return keys


def _atomic_groups(model: nn.Module) -> List[Tuple[str, List[str]]]:
    """Ordered atomic groups: one per leaf module, tied groups merged.

    Returns ``[(group_name, [keys...]), ...]`` in topological order.
    Two groups whose tensors share storage (``data_ptr``) are merged
    into the earlier group, so tied weights can never be split.
    """
    live = model.state_dict(keep_vars=True)
    groups: List[Tuple[str, List[str]]] = []
    for mod_name, mod in model.named_modules():
        prefix = f"{mod_name}." if mod_name else ""
        keys: List[str] = []
        for pname, p in mod._parameters.items():
            if p is not None:
                keys.append(prefix + pname)
        for bname, b in mod._buffers.items():
            if b is None or bname in mod._non_persistent_buffers_set:
                continue
            keys.append(prefix + bname)
        if keys:
            groups.append((mod_name if mod_name else "<root>", keys))

    # Merge groups linked by shared storage into the earliest group.
    ptr_to_group: Dict[int, int] = {}
    merged: List[Tuple[str, List[str]]] = []
    group_index_remap: Dict[int, int] = {}
    for gi, (gname, keys) in enumerate(groups):
        target = None
        for k in keys:
            ptr = live[k].data_ptr()
            if ptr in ptr_to_group:
                target = ptr_to_group[ptr]
                break
        if target is None:
            group_index_remap[gi] = len(merged)
            merged.append((gname, list(keys)))
        else:
            group_index_remap[gi] = target
            tname, tkeys = merged[target]
            tkeys.extend(k for k in keys if k not in tkeys)
        for k in keys:
            ptr_to_group.setdefault(live[k].data_ptr(), group_index_remap[gi])
    return merged


def _numel(model: nn.Module, keys: Sequence[str]) -> int:
    sd = model.state_dict()
    return sum(sd[k].numel() for k in keys)


def _make_blocks(model: nn.Module, named_key_groups: Sequence[Tuple[str, List[str]]]) -> List[BlockSpec]:
    return [
        BlockSpec(id=i, name=name, state_dict_keys=list(keys), num_params=_numel(model, keys))
        for i, (name, keys) in enumerate(named_key_groups)
    ]


# ------------------------------------------------------------------ strategies


def decompose_balanced(model: nn.Module, num_blocks: int) -> List[BlockSpec]:
    """Partition into ``num_blocks`` roughly parameter-balanced blocks.

    Greedily packs atomic groups (leaf modules; tied groups pre-merged)
    in topological order until each block reaches ``total/num_blocks``
    elements. May return fewer than ``num_blocks`` blocks for tiny models.
    """
    if num_blocks < 1:
        raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
    groups = _atomic_groups(model)
    if not groups:
        raise ValueError("Model has no parameters or buffers in its state_dict")
    sd = model.state_dict()
    sizes = [sum(sd[k].numel() for k in keys) for _, keys in groups]
    total = sum(sizes)
    target = max(1, total // num_blocks)

    out: List[Tuple[str, List[str]]] = []
    cur_keys: List[str] = []
    cur_count = 0
    for (gname, keys), size in zip(groups, sizes):
        cur_keys.extend(keys)
        cur_count += size
        if cur_count >= target and len(out) < num_blocks - 1:
            out.append((f"block_{len(out)}", cur_keys))
            cur_keys, cur_count = [], 0
    if cur_keys:
        out.append((f"block_{len(out)}", cur_keys))
    blocks = _make_blocks(model, out)
    validate_decomposition(model, blocks)
    return blocks


def decompose_top_level(model: nn.Module) -> List[BlockSpec]:
    """Every direct child of ``model`` is a block.

    Keys owned by the root module itself (rare) are appended to the
    last block. Fails loudly if tied weights span two children.
    """
    children = [name for name, _ in model.named_children()]
    if not children:
        raise ValueError("Model has no direct children to use as blocks")
    ordered = _ordered_keys(model)
    buckets: Dict[str, List[str]] = {name: [] for name in children}
    leftovers: List[str] = []
    for key in ordered:
        head = key.split(".", 1)[0]
        if head in buckets:
            buckets[head].append(key)
        else:
            leftovers.append(key)
    named = [(name, buckets[name]) for name in children if buckets[name]]
    if leftovers:
        named[-1] = (named[-1][0], named[-1][1] + leftovers)
    blocks = _make_blocks(model, named)
    validate_decomposition(model, blocks)
    return blocks


def decompose_manual(
    model: nn.Module,
    patterns: Sequence[str],
    *,
    allow_module_split: bool = False,
) -> List[BlockSpec]:
    """Assign each key to the first matching regex; one block per pattern.

    Raises if any key matches no pattern (add a final catch-all
    ``'.*'``), or if a leaf module / tied group is split across blocks
    (override the module check with ``allow_module_split=True``;
    tied-weight splits always fail).
    """
    compiled = [re.compile(p) for p in patterns]
    if not compiled:
        raise ValueError("decompose_manual requires at least one pattern")
    ordered = _ordered_keys(model)
    buckets: List[List[str]] = [[] for _ in compiled]
    for key in ordered:
        for i, pat in enumerate(compiled):
            if pat.search(key):
                buckets[i].append(key)
                break
        else:
            raise ValueError(
                f"State-dict key {key!r} matched no block pattern. "
                "Add a catch-all pattern such as '.*' as the last entry."
            )
    named = [(f"block_{i}", keys) for i, keys in enumerate(buckets) if keys]
    blocks = _make_blocks(model, named)
    validate_decomposition(model, blocks, allow_module_split=allow_module_split)
    return blocks


def decompose_boundary(
    model: nn.Module,
    boundaries: Sequence[Union[str, Sequence[str]]],
) -> List[BlockSpec]:
    """Explicit boundary list: each entry is one block, given as a module
    prefix or a list of prefixes.

    Example (ResNet): ``[["conv1", "bn1"], "layer1", "layer2", "layer3",
    "layer4", ["avgpool", "fc"]]`` → stem / 4 residual stages / head.
    Every state-dict key must be covered by exactly one entry.
    """
    if not boundaries:
        raise ValueError("decompose_boundary requires at least one boundary entry")
    norm: List[List[str]] = [
        [b] if isinstance(b, str) else list(b) for b in boundaries
    ]
    ordered = _ordered_keys(model)
    named: List[Tuple[str, List[str]]] = []
    claimed: Dict[str, str] = {}
    for prefixes in norm:
        keys: List[str] = []
        for key in ordered:
            head_match = any(key == p or key.startswith(p + ".") for p in prefixes)
            if head_match:
                if key in claimed:
                    raise ValueError(
                        f"Key {key!r} claimed by two boundary entries "
                        f"({claimed[key]!r} and {'+'.join(prefixes)!r})."
                    )
                claimed[key] = "+".join(prefixes)
                keys.append(key)
        if not keys:
            raise ValueError(f"Boundary entry {'+'.join(prefixes)!r} matched no state-dict keys.")
        named.append(("+".join(prefixes), keys))
    uncovered = [k for k in ordered if k not in claimed]
    if uncovered:
        raise ValueError(
            f"Boundary list does not cover {len(uncovered)} state-dict keys, "
            f"e.g. {uncovered[:5]}. Add entries (or a final group) covering them."
        )
    blocks = _make_blocks(model, named)
    validate_decomposition(model, blocks)
    return blocks


# ------------------------------------------------------------------ validation


def validate_decomposition(
    model: nn.Module,
    blocks: Sequence[BlockSpec],
    *,
    allow_module_split: bool = False,
) -> None:
    """Strict invariants; raises ``ValueError`` on the first violation.

    1. Exact cover: the blocks' keys partition ``state_dict()`` keys —
       no missing, no extra, no duplicates.
    2. Tied weights: keys sharing one storage (``data_ptr``) all live in
       the same block. Never overridable.
    3. Module atomicity: a leaf module's keys (weights + norm buffers)
       live in one block — unless ``allow_module_split=True``.
    """
    sd_keys = list(model.state_dict().keys())
    seen: Dict[str, int] = {}
    for spec in blocks:
        for k in spec.state_dict_keys:
            if k in seen:
                raise ValueError(f"Key {k!r} appears in blocks {seen[k]} and {spec.id}.")
            seen[k] = spec.id
    missing = [k for k in sd_keys if k not in seen]
    extra = [k for k in seen if k not in set(sd_keys)]
    if missing or extra:
        raise ValueError(
            f"Decomposition does not exactly cover the state_dict. "
            f"Missing: {missing[:5]}{'…' if len(missing) > 5 else ''}; "
            f"extra: {extra[:5]}{'…' if len(extra) > 5 else ''}."
        )

    live = model.state_dict(keep_vars=True)
    ptr_block: Dict[int, Tuple[str, int]] = {}
    for k in sd_keys:
        ptr = live[k].data_ptr()
        if ptr in ptr_block:
            first_key, first_block = ptr_block[ptr]
            if seen[k] != first_block:
                raise ValueError(
                    f"Tied weights split across blocks: {first_key!r} (block "
                    f"{first_block}) and {k!r} (block {seen[k]}) share storage. "
                    "Tied keys must be assigned to one block (use decompose_manual "
                    "or decompose_boundary with both prefixes in one entry)."
                )
        else:
            ptr_block[ptr] = (k, seen[k])

    if not allow_module_split:
        for gname, keys in _atomic_groups(model):
            owners = {seen[k] for k in keys}
            if len(owners) > 1:
                raise ValueError(
                    f"Leaf module {gname!r} is split across blocks {sorted(owners)}; "
                    "its parameters and buffers (e.g. norm running stats) must stay "
                    "together. Pass allow_module_split=True only if you know better."
                )


# ------------------------------------------------------------------ CLI grammar


def parse_decomposition(spec: str):
    """Parse a CLI decomposition spec into ``(callable, description)``.

    Grammar:

    * ``balanced:K``                      → :func:`decompose_balanced`
    * ``top-level``                       → :func:`decompose_top_level`
    * ``manual:<regex>;<regex>;…``        → :func:`decompose_manual`
    * ``boundary:p1+p2,p3,p4+p5,…``       → :func:`decompose_boundary`
      (commas separate blocks; ``+`` joins prefixes within one block)
    """
    if spec.startswith("balanced:"):
        k = int(spec.split(":", 1)[1])
        return (lambda model: decompose_balanced(model, k)), f"balanced:{k}"
    if spec == "top-level":
        return decompose_top_level, "top-level"
    if spec.startswith("manual:"):
        patterns = [p for p in spec.split(":", 1)[1].split(";") if p]
        return (lambda model: decompose_manual(model, patterns)), f"manual:{len(patterns)} patterns"
    if spec.startswith("boundary:"):
        entries: List[Union[str, List[str]]] = []
        for group in spec.split(":", 1)[1].split(","):
            group = group.strip()
            if not group:
                continue
            parts = [p.strip() for p in group.split("+") if p.strip()]
            entries.append(parts if len(parts) > 1 else parts[0])
        return (lambda model: decompose_boundary(model, entries)), f"boundary:{len(entries)} blocks"
    raise ValueError(
        f"Unsupported decomposition spec {spec!r}. Use 'balanced:K', 'top-level', "
        "'manual:<regex>;<regex>;…', or 'boundary:p1+p2,p3,…'."
    )
