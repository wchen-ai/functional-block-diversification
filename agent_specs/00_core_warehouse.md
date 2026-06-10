# 00 — Core Warehouse Specification (shared by all versions)

**Audience:** an AI coding agent implementing one of the three versions
(`01_uncertainty_local.md`, `02_federated_fbd.md`, `03_active_learning.md`).
Read this file plus exactly one version spec; together they are self-contained.

**Status: the core described here is ALREADY IMPLEMENTED and validated** in
`warehouse_adopt/warehouse_adopt/` with a passing smoke test
(`warehouse_adopt/tests/test_smoke_mnist.py`, 5 tests, CPU, ~5 s).
Your first action is to run it:

```bash
cd warehouse_adopt && python -m pytest tests/test_smoke_mnist.py -q   # expect: 5 passed
```

Do **not** rewrite the core. Verify it, then build your version on top of it.
This document is the normative description of that core: if you ever find a
divergence between this spec and the code, the spec wins — fix the code and
keep the smoke test green. Reference implementations for many semantics also
exist in `SASWISE/src/adapters/` (single-machine origin) and `fbd_transfer/`
(federated origin); they are *read-only inspiration*, never imports.

---

## 1. The source-free principle (non-negotiable)

The system never edits a model's source code. It requires exactly two things:

* **(a) a constructor** for the architecture — a `model_factory()` callable,
  or an auto-construction spec (`torchvision:<name>`, `timm:<name>`,
  `hf:<repo>`); and
* **(b) the checkpoint's `state_dict`**.

"Blocks" are groups of `state_dict` keys. "Assembling" a sub-model means
composing a `state_dict` from per-block variant slices and calling
`load_state_dict` on a model built by the constructor. The model's **own,
unmodified `forward`** runs every pass. No subclassing user models, no
monkey-patching `forward`, no FX tracing, no graph surgery.

The only unsupported input is a bare tensor dump with no obtainable
constructor — detect this (constructor missing/unresolvable) and fail with an
actionable message.

## 2. Package layout

```
warehouse_adopt/                      # project root (this directory)
├── pyproject.toml                    # name=warehouse-adopt, console script
├── agent_specs/                      # the documents you are reading
├── warehouse_adopt/                  # the python package
│   ├── __init__.py                   # re-exports the public API (§9)
│   ├── contract.py                   # §3 user-module contract + auto-construction
│   ├── decompose.py                  # §4 decomposition strategies + validation
│   ├── warehouse.py                  # §5 disk store + provenance + hashes
│   ├── engine.py                     # §6 offline/online assembly engines
│   ├── training.py                   # §7 diversification + ConsistencyBalance
│   ├── inference.py                  # §8 Welford ensemble inference
│   └── cli.py                        # §10 CLI: init | finetune | infer
└── tests/
    ├── conftest.py                   # sys.path bootstrap (no install needed)
    └── test_smoke_mnist.py           # the acceptance smoke test
```

Version specs add **subpackages** (`warehouse_adopt/uncertainty/`,
`warehouse_adopt/federated/`, `warehouse_adopt/active/`) and CLI subcommands;
they never modify the core modules except where a version spec explicitly
says so.

## 3. User-module contract & auto-construction (`contract.py`)

A *user module* is one `.py` file with five top-level callables (one optional
sixth). This is the same contract as `SASWISE/examples/adopt_mnist_mlp.py`.

```python
@dataclass
class UserModule:
    model_factory:        Callable[[], nn.Module]                 # fresh, randomly-init model; structure only — weights always overwritten by assembly
    load_base_checkpoint: Callable[[nn.Module, str], None]        # load pretrained weights into a model_factory() instance, in place
    get_dataloaders:      Callable[[], tuple[Any, Any]]           # () -> (train_loader, val_loader)
    infer_fn:             Callable[[nn.Module, Any], Tensor]      # one forward pass; batch is whatever the loader yields
    loss_fn:              Callable[[Tensor, Any], Tensor]         # scalar task loss for out = infer_fn(model, batch)
    consistency_fn:       Optional[Callable[[Tensor, Tensor], Tensor]] = None  # optional; default MSE (§7)

    @classmethod
    def from_namespace(cls, ns) -> "UserModule"                   # build from any object with the attributes; raises AttributeError listing missing names

def load_user_module(path: str | Path) -> UserModule              # import a .py file, validate the contract, wrap it

def resolve_model_factory(spec: str) -> Callable[[], nn.Module]
    # Auto-construction fallbacks when the user has no model code:
    #   torchvision:<name>[?k=v&k2=v2]  -> torchvision.models.<name>(**kwargs)
    #   timm:<name>[?...]               -> timm.create_model(name, pretrained=False, **kwargs)
    #   hf:<repo>[?...]                 -> AutoModel.from_config(AutoConfig.from_pretrained(repo, **kwargs))   # constructor-only, never downloads weights
    # kwarg values are ast.literal_eval'd, falling back to str.
    # Missing provider library -> ImportError naming the extra to install.

def load_checkpoint_state_dict(path, *, strip_prefixes=("module.",), key=None) -> dict[str, Tensor]
    # torch.load(weights_only=True); unwraps {"state_dict": ...}/{"model": ...}; strips DataParallel prefixes.

def to_device(batch, device) -> Any                               # recursively moves tensors in tensors/lists/tuples/dicts; passes through everything else
```

## 4. Block decomposition (`decompose.py`)

A **block** is a non-overlapping subset of `state_dict` keys (parameters AND
persistent buffers). All blocks together cover every key **exactly once**.
Key order inside a block follows `named_modules()` topological (registration)
order — never incidental dict order.

```python
@dataclass
class BlockSpec:
    id: int                       # zero-based block index
    name: str                     # human label, e.g. "layer3" or "block_2"
    state_dict_keys: list[str]    # ordered keys belonging to this block
    num_params: int               # total element count across keys

decompose_balanced(model, num_blocks) -> list[BlockSpec]
    # Greedily packs *atomic groups* (see below) in topological order into
    # ~equal element-count blocks. May return fewer blocks for tiny models.
decompose_top_level(model) -> list[BlockSpec]
    # One block per direct child of the root; root-owned stragglers go to the
    # last block. Natural for ResNets/U-Nets.
decompose_manual(model, patterns: Sequence[str], *, allow_module_split=False) -> list[BlockSpec]
    # Each key -> first matching regex; one block per pattern; a key matching
    # no pattern raises (tell the user to add '.*' as catch-all).
decompose_boundary(model, boundaries: Sequence[str | Sequence[str]]) -> list[BlockSpec]
    # Explicit boundary list; each entry = one block given as module prefix(es).
    # ResNet example: [["conv1","bn1"],"layer1","layer2","layer3","layer4",["avgpool","fc"]]
    # Uncovered or doubly-claimed keys raise with the offending keys listed.
parse_decomposition(spec: str) -> (callable, description)
    # CLI grammar: 'balanced:K' | 'top-level' | 'manual:<re>;<re>;…' | 'boundary:p1+p2,p3,…'
validate_decomposition(model, blocks, *, allow_module_split=False) -> None
    # Strict invariants; ValueError on first violation. Called by every strategy.
```

**Atomic groups.** A *leaf module* (a module owning params/buffers directly)
is atomic: its weight, bias, and norm running statistics always land in the
same block. Groups whose tensors share storage (`Tensor.data_ptr()` on
`state_dict(keep_vars=True)`) are merged into the earliest group, so **tied
weights are never split**.

**`validate_decomposition` invariants (all strategies must pass):**

1. *Exact cover* — keys partition `state_dict()`: no missing, extra, duplicate.
2. *Tied weights* — keys sharing a storage live in one block. Never overridable;
   violation raises naming both keys and both blocks.
3. *Module atomicity* — a leaf module's keys live in one block, unless
   `allow_module_split=True` (manual only; tied-weight rule still applies).

## 5. The warehouse (`warehouse.py`)

On-disk layout (one `.pt` per `(block, variant)`):

```
root/
├── warehouse_metadata.json
├── block_00/variant_00.pt      # dict[str, Tensor] — exactly block 0's keys
├── block_00/variant_01.pt
└── block_01/…
```

`warehouse_metadata.json` schema (`SCHEMA_VERSION = 1`):

```json
{
  "version": 1,
  "num_variants": 4,
  "block_specs": [
    {"id": 0, "name": "block_0", "state_dict_keys": ["fc1.weight", "..."], "num_params": 51264}
  ],
  "provenance": {
    "0/0": [{"writer_id": "init", "round": 0, "content_hash": "sha256:…"}],
    "0/1": [{"writer_id": "init", "round": 0, "content_hash": "sha256:…"},
             {"writer_id": "client_3", "round": 17, "content_hash": "sha256:…"}]
  }
}
```

Provenance keys are `"<block_id>/<variant_id>"`; entries are append-only,
oldest first; the **last** entry's hash always matches the bytes on disk.

```python
class Warehouse:
    def __init__(self, root, block_specs, num_variants, *, exist_ok=False)  # creates dirs + metadata; refuses to clobber an existing warehouse unless exist_ok
    @classmethod
    def load(cls, root) -> "Warehouse"                                      # reconstruct handle from metadata on disk
    def variant_path(self, block_id, variant_id) -> Path                    # root/block_<i:02d>/variant_<j:02d>.pt
    def save_variant(self, block_id, variant_id, slice_state, *, writer_id="local", round=0) -> str
        # Validates slice keys == spec keys exactly; stores CPU tensors with original dtype;
        # appends a provenance entry; persists metadata; returns the content hash.
    def load_variant(self, block_id, variant_id, map_location="cpu") -> dict[str, Tensor]
        # weights_only torch.load; validates the stored key set against metadata.
    def provenance(self, block_id, variant_id) -> list[ProvenanceEntry]     # full write history, oldest first
    def content_hash(self, block_id, variant_id) -> str | None              # last write's hash (None if never written)
    def variants_written_by(self, writer_id) -> list[tuple[int, int]]       # all (block, variant) whose trace contains writer_id — the unlearning query
    def verify(self) -> list[str]                                           # re-hash every slice vs metadata; [] means intact
    num_blocks: int;  num_variants: int;  block_specs: list[BlockSpec]
    def block_ids(self) -> list[int]
    def num_configurations(self) -> int                                     # num_variants ** num_blocks
    def block_by_id(self, block_id) -> BlockSpec

def tensor_content_hash(slice_state) -> str
    # sha256 over sorted (key, dtype, shape, raw bytes); bfloat16 hashed via int16 view.
    # File-format independent: equal hash <=> bit-identical tensors.

def initialize_variants(model, block_specs, num_variants, warehouse, *,
                        noise_std=1e-3, seed=0, writer_id="init") -> None
    # variant_0[k] = base[k] verbatim;
    # variant_v[k] = base[k] + N(0, noise_std·mean|base[k]|) for v>=1, floating tensors only
    # (integer tensors like BN num_batches_tracked are cloned verbatim).
    # Deterministic: ONE cpu torch.Generator(seed), consumed in (block, variant, key) order.
```

**Invariant — variant 0 is the unperturbed clone:** assembling the all-zeros
configuration reproduces the original checkpoint bit-for-bit, forever (until
training overwrites it; federated/unlearning versions that need a pristine
copy must either never write variant 0 or snapshot it first — see the version
specs).

## 6. Two assembly engines, one switch (`engine.py`)

```python
def state_dict_bytes(state: dict[str, Tensor]) -> int   # exact slice bytes: Σ numel·element_size

@dataclass
class EngineStats:           # deterministic counters of LOGICAL byte movement
    assembles: int           # assemble() calls
    block_swaps: int         # per-block activations performed
    pool_to_device_bytes: int  # slice bytes delivered into the active model by assemble
    device_to_pool_bytes: int  # bytes captured from the active model by writeback
    disk_read_bytes: int     # bytes actually read from the filesystem
    disk_write_bytes: int    # bytes actually written to the filesystem
    def reset(self) -> None; def snapshot(self) -> dict
    # A pool→device byte crosses PCIe exactly when pool and compute device differ in type.

class AssemblyEngine(ABC):
    def __init__(self, warehouse, model_factory, *, device="cpu")   # .stats: EngineStats
    def assemble(self, block_config: dict[int, int], train: bool = False) -> nn.Module
        # Return a model on self.device loaded with block_config (full cover, validated).
    def writeback(self, model, block_config, *, blocks: list[int] | None = None,
                  writer_id="local", round=0) -> None
        # Persist the model's current weights for `blocks` (default: all blocks in config).
    def flush(self) -> None                 # persist deferred writes (offline: no-op)
    def close(self) -> None                 # flush + release resident state
    def pool_resident_bytes(self) -> int    # bytes of the variant pool held resident (0 = on disk)

class OnlineEngine(AssemblyEngine):
    def __init__(self, warehouse, model_factory, *, device="cpu", pool_device="cpu")
    def prefetch_pool(self) -> int          # load every slice into the resident pool;
                                            # afterwards pool_resident_bytes() == N·W exactly

ENGINE_MODES  = ("offline", "online")
POOL_DEVICES  = ("disk", "cpu", "cuda")
def make_engine(mode, warehouse, model_factory, *, device="cpu",
                pool_device=None) -> AssemblyEngine
    # pool_device — WHERE THE N-VARIANT POOL LIVES (first-class knob, §6.1):
    #   None    mode default: offline → "disk", online → "cpu"
    #   "disk"  offline only (its defining property; online+disk raises)
    #   "cpu"   online, pool in host RAM — each swap moves one block over PCIe
    #   "cuda"[:K] / torch.device — online, pool resident on the GPU:
    #           zero-bus swaps, costs ≈ N·W of VRAM (offline+cuda raises)
```

**Offline (passive, pool on disk).** `assemble` builds a *fresh*
`model_factory()` model, merges all requested slices from disk,
`load_state_dict(strict=True)` (full cover ⇒ strict is the correct check),
moves to device. `writeback` slices `model.state_dict()` per block and saves
to disk immediately. Nothing engine-owned stays resident between calls.

**Online (active, pool resident on `pool_device`).** ONE resident model on
the compute device, built lazily on first `assemble`. Per block, if the
active variant differs from the request, the engine
`load_state_dict(strict=False)`s only that block's slice from the resident
slice cache (filled lazily from disk onto `pool_device`; `prefetch_pool()`
warms it deterministically). `load_state_dict` performs in-place `copy_`
(never `assign=True`), so **hooks and parametrizations registered on the
resident model survive activation**. `writeback` clones the block's tensors
into the cache (on `pool_device`) and marks `(block, variant)` dirty; `flush`
persists dirty slices (sorted order) via `Warehouse.save_variant`; `close`
flushes and drops the resident model + cache. Reads of a dirty slice hit the
cache, never stale disk.

| | offline (pool=disk) | online (pool=cpu, default) | online (pool=cuda) |
|---|---|---|---|
| Resident engine state | none between calls | 1 model + pool in host RAM | 1 model + pool in VRAM |
| Cost per `assemble` | full model: disk read + load | per-block diff swap over PCIe | per-block diff swap on-device |
| Cost per `writeback` | full disk write per touched block | RAM clone; disk deferred to `flush` | VRAM clone; disk deferred to `flush` |
| Crash safety | every step on disk | dirty slices lost unless `flush_every` | dirty slices lost unless `flush_every` |
| Quantified footprint/transfer | **see §6.1** | **see §6.1** | **see §6.1** |

**Numerical-equivalence requirement (MUST, enforced by the smoke test):**
with identical starting warehouse content and an identical
`assemble`/`writeback` call sequence on identical batches, the two engines —
under any `pool_device` — produce **bit-identical** final variant tensors and
equal content hashes (fp32 host↔device transfers are exact; arithmetic runs
on `device` either way). Provenance *granularity* may differ (offline logs
every step's write; online logs at flush) — content equality is what is
required.

## 6.1 Resource model — measured, not asserted

Symbols: **P** = trainable parameter count; **Wp** = parameter bytes
(≈ 4P fp32); **W** = model state bytes = Wp + persistent-buffer bytes
(W ≈ Wp for typical nets); **w_b** = block *b* bytes, Σ w_b = W; **B** =
blocks; **N** = colours/variants; **T** = training steps; **K** = inference
configurations; **act** = forward/backward activation working set;
**A** = *concurrently resident* sub-models.

**Where the active set costs live (identical across all three engines):**

* **Inference: A = 1.** One assembled sub-model + activations — the GPU
  footprint of the *unwrapped original model*. Sampling more configurations
  costs time, never memory (configurations are evaluated sequentially).
* **Diversification training touches TWO sub-models per step** — the update
  colour M_k and the frozen reference M_j (§7) — but the **shipped canonical
  order time-multiplexes them: M_j is assembled, evaluated under `no_grad`,
  and released *before* M_k is assembled, so weight residency stays A = 1.**
  The factor 2 appears in per-step *transfer* (two full configurations are
  loaded each step), not in residency. An implementation that keeps both
  resident concurrently (e.g. `SASWISE/src/adapters/finetune.py`, or any
  variant that interleaves the two forwards) pays A = 2: one extra +W of
  VRAM. The audit prompt's "A = 2 during training" describes that concurrent
  ordering; the bench below *proves* the shipped core runs at A = 1.
* **Training-only extras on the trained model M_k:** gradients (+Wp) and
  Adam moments (+2Wp; the "+2P" optimizer multiplier), allocated fresh each
  step (§7), plus a transient step temporary of O(largest tensor)
  (`foreach=False` is pinned in the canonical step precisely so this stays
  O(one tensor) instead of ≈ +2Wp). M_j, when resident, is weights-only.
  These costs are independent of the pool-placement choice.

**Steady-state footprint and per-step transfer by pool placement:**

| | offline (pool=disk) | online (pool=cpu) | online (pool=cuda) |
|---|---|---|---|
| GPU VRAM, inference (A=1) | W + act | W + act | **N·W** + W + act |
| GPU VRAM, training (canonical) | W + 3Wp + act | W + 3Wp + act | **N·W** + W + 3Wp + act |
| GPU VRAM, training (concurrent A=2) | + W | + W | + W |
| CPU RAM, steady | ≈ 0 (≤ 2W transient staging per assemble) | **N·W** (pool) | ≈ 0 |
| Disk | N·W across B·N files | N·W | N·W |
| Transfer per training step | 2W disk-read, 2W pool→device, W device→pool, W disk-write | Σ swapped w_b ≤ 2W over PCIe, W device→host; disk 0 (flush: Σ distinct dirty ≤ T·W) | same bytes as online/cpu but device-to-device; **PCIe ≈ 0** after `prefetch_pool` |
| Transfer per inference config | W disk-read + W pool→device | Σ swapped w_b over PCIe | Σ swapped w_b on-device |

**Measured demonstration** (`python -m warehouse_adopt.bench_resource`;
asserted by `tests/test_resource_model.py`; BenchNet W = 5,351,472 B, B = 3,
N = 3, batch 128, 4 inference configs + 4 canonical training steps, CUDA):

* Counters equal the closed-form/replayed predictions **exactly** in all
  three placements: offline training moved 2T·W = 42,811,776 B pool→device
  and wrote T·W eagerly; online moved only the swapped blocks —
  26,757,360 B (0.63 × offline) — wrote nothing during the phase, and
  flushed 10,702,944 B of distinct dirty slices (< T·W: deferred writes
  deduplicate).
* `pool_resident_bytes() == N·W` exactly (16,054,416 B) for online after
  `prefetch_pool` — in host RAM (pool=cpu) or VRAM (pool=cuda); 0 for offline.
* Engine-attributable VRAM is **byte-identical** for offline and online/cpu —
  inference 1.34·W, training 5.21·Wp (= 4·Wp weights+grads+moments
  + 0.4·Wp activations + 0.6·Wp Adam step temporary) — and online/cuda adds
  the pool term. Training stays far below the ≥ 6.2·Wp a concurrent second
  sub-model would cost: **A = 1, demonstrated.**

**Decision rule for `pool_device`:** scarce VRAM or few steps (or federated
round-trips, where the warehouse lives server-side anyway) → **offline/disk**;
abundant host RAM and many steps on one box → **online/cpu** (default); pool
fits in VRAM and swap latency matters → **online/cuda**. In all cases
inference costs the same GPU as the unwrapped model (plus the pool only if
you put it on the GPU).

## 7. Diversification training (`training.py`)

The **canonical step** — both engines MUST execute exactly this order (this
is what makes them equivalent):

1. `ref = engine.assemble(cfg_ref, train=False)`; `out_ref = infer_fn(ref, batch)`
   under `torch.no_grad()`, then `.detach()`; release `ref`.
2. `model = engine.assemble(cfg_update, train=True)`; `out = infer_fn(model, batch)`.
3. `loss = loss_fn(out, batch) + λ · consistency_fn(out, out_ref)`
   (`consistency_fn` defaults to `default_consistency_fn` = MSE; classification
   users should supply softmax-KL).
4. One step of a **fresh `torch.optim.Adam(model.parameters(), lr,
   foreach=False)`** — persistent optimizer state across configurations would
   defeat block isolation (never cache optimizers), and `foreach=False` keeps
   the step-time temporaries at O(largest tensor) instead of ≈ +2·Wp
   (identical arithmetic; the §6.1 resource model assumes it).
5. `engine.writeback(model, cfg_update, writer_id=…, round=step)` — only the
   update configuration's blocks are persisted; the reference is never written.

**Two-sub-models mechanics (memory view of steps 1–4).** Every step evaluates
two sub-models: M_k (update) ultimately carries weights + gradients + two
Adam moments ≈ Wp + Wp + 2Wp = 4 parameter-copies ≈ 16·P bytes fp32; M_j
(reference) is evaluated under `no_grad` in eval mode and carries weights
only (its forward retains just the output tensor, which is detached and kept
as `out_ref`). Both *logically* participate in the loss, but the canonical
order above releases M_j **before** M_k exists, so they are never resident
simultaneously — weight residency stays at one sub-model and the second
model's cost surfaces as per-step transfer instead (2 full configurations
loaded per step; quantified and measured in §6.1). Keeping both resident
concurrently is a legal variant costing one extra +W of device memory; it is
NOT what the core implements. This residency behaviour is independent of the
pool-placement choice (`pool_device`).

```python
@dataclass
class ConsistencyBalance:           # λ schedule for L = L_task + λ·L_cons
    adaptive: bool = True           # adaptive: λ = (r/(1−r))·EMA[L_task]/EMA[L_cons], clipped
    target_ratio: float = 0.1       # r — consistency's target share of total loss
    fixed_weight: float = 0.1       # λ when adaptive=False
    warmup_steps: int = 100         # linear 0→target ramp; 0 disables
    ema_decay: float = 0.9
    min_weight: float = 0.0
    max_weight: float = 100.0

class ConsistencyBalancer:          # lag-1: current_weight() reads EMAs from steps 0..t-1,
    def __init__(self, config)      # so a loss never scales its own gradient
    def current_weight(self) -> float
    def observe(self, task_loss: float, cons_loss: float) -> None

def default_consistency_fn(out_update, out_ref) -> Tensor       # F.mse_loss
def sample_distinct_configs(block_ids, num_variants, rng: random.Random) -> (cfg_update, cfg_ref)
    # Two full configs differing in >=1 block; consumes rng in fixed order (reproducible schedule).

@dataclass
class StepStats: step; total_loss; task_loss; consistency_loss; weight; cfg_update; cfg_ref

def diversification_step(engine, user, batch, cfg_update, cfg_ref, *,
                         lr, weight, step=0, writer_id="local") -> StepStats
    # Executes the canonical step above; batch must already be on engine.device.

def finetune_warehouse(warehouse, user, *, mode="offline", epochs=1, max_steps=None,
                       lr=1e-4, consistency=None, device="cpu", pool_device=None,
                       seed=0, writer_id="local", flush_every=0, log_every=0) -> list[StepStats]
    # Loop: sample (cfg_update, cfg_ref) from random.Random(seed) ONLY, move batch,
    # run the canonical step, balancer.observe AFTER the gradient step.
    # consistency: ConsistencyBalance | float (fixed λ, no warm-up) | None (adaptive default).
    # flush_every>0 flushes the online engine periodically; engine.close() in a finally block.

def recalibrate_bn(model, loader, user, *, num_batches=50, device="cpu", reset=True) -> nn.Module
    # Optional BN repair after cross-config assembly: reset running stats, forward
    # num_batches in train mode under no_grad (only running stats change).
```

## 8. Ensemble inference (`inference.py`)

```python
@dataclass
class EnsemblePrediction:
    mean: Tensor                          # (N, …) Welford running mean over configurations
    std: Tensor                           # (N, …) per-element std across configurations
    num_configurations: int
    predictive_entropy: Tensor | None     # (N,) entropy of softmax(mean) — classification only
    block_configurations: list[dict[int, int]]

def sample_configurations(warehouse, num_configurations, seed=0) -> list[dict[int, int]]
    # Enumerates the full product if num_configurations >= total, else samples
    # distinct configs without replacement from random.Random(seed).

def infer_with_uncertainty(warehouse, user, *, mode="offline", num_configurations=16,
                           task="auto", device="cpu", pool_device=None, seed=0,
                           show_progress=False) -> EnsemblePrediction
    # Per config: engine.assemble -> no_grad infer_fn over val_loader -> fold into
    # Welford mean/M2. Never stores per-config outputs (O(output_size) memory).
    # Configurations run SEQUENTIALLY: compute footprint = ONE sub-model (A=1,
    # §6.1) = the unwrapped model, regardless of num_configurations.
    # task="classification" adds predictive entropy; "auto" = rank-2 output.
```

**The std-vs-entropy caveat (must survive into every version):** for
classification, the uncertainty scalar is `predictive_entropy`, NOT the
logit `std` (measured on MNIST: AUROC of misclassification 0.92 for entropy
vs 0.44 for std). `std` is correct for regression/segmentation.

## 9. Public API

Everything above is re-exported flat from `warehouse_adopt/__init__.py`:

`UserModule, load_user_module, resolve_model_factory, load_checkpoint_state_dict,
to_device, BlockSpec, decompose_balanced, decompose_top_level, decompose_manual,
decompose_boundary, parse_decomposition, validate_decomposition, Warehouse,
ProvenanceEntry, initialize_variants, tensor_content_hash, AssemblyEngine,
OfflineEngine, OnlineEngine, make_engine, ENGINE_MODES, POOL_DEVICES,
EngineStats, state_dict_bytes, ConsistencyBalance, ConsistencyBalancer,
default_consistency_fn, sample_distinct_configs, StepStats,
diversification_step, finetune_warehouse, recalibrate_bn, EnsemblePrediction,
sample_configurations, infer_with_uncertainty`

The resource-model demonstration lives in `warehouse_adopt/bench_resource.py`
(runnable: `python -m warehouse_adopt.bench_resource [--device cuda]`) with
its measurement/prediction helpers (`measure_config`, `predict_offline`,
`predict_online`, `verify_config`, `compute_sizes`).

## 10. CLI skeleton (`cli.py`)

Console script `warehouse-adopt` (also `python -m warehouse_adopt.cli`), three
subcommands shared by all versions; version specs ADD subcommands:

```
warehouse-adopt init     (--user-module F | --model torchvision:<n>|timm:<n>|hf:<repo>)
                         [--checkpoint F] --output-dir D
                         [--decomposition balanced:6] [--variants 4] [--noise-std 1e-3] [--seed 0]
warehouse-adopt finetune --user-module F --warehouse D [--mode offline|online]
                         [--pool-device disk|cpu|cuda[:K]]
                         [--epochs 1] [--max-steps N] [--lr 1e-4] [--device auto]
                         [--seed 0] [--flush-every 0] [--log-every 50]
                         [--consistency-mode adaptive|fixed] [--target-ratio 0.1]
                         [--warmup-steps 100] [--ema-decay 0.9] [--fixed-weight 0.1]
warehouse-adopt infer    --user-module F --warehouse D [--mode offline|online]
                         [--pool-device disk|cpu|cuda[:K]]
                         [--num-configurations 16] [--task auto] [--device auto]
                         [--seed 0] [--output predictions.pt] [--quiet]
```

`init` requires exactly one of `--user-module`/`--model`; with `--model` and a
checkpoint it goes through `resolve_model_factory` +
`load_checkpoint_state_dict` (fully source-free). `infer` saves
`{"mean","std","num_configurations","block_configurations"[,"predictive_entropy"]}`
via `torch.save`.

## 11. Source-free gotchas → acceptance items

Each of these MUST hold in any implementation built on the core; the smoke
test already checks the starred ones for the core itself.

| # | Gotcha | Required behaviour |
|---|--------|--------------------|
| G1* | Norm-layer buffers | `running_mean/var/num_batches_tracked` are state_dict keys; they travel in the same block as their module's weights (atomic groups). Optional repair: `recalibrate_bn`. |
| G2* | Tied/shared weights | Detected via `data_ptr()` on `state_dict(keep_vars=True)`; merged into one block; any decomposition that would split them **fails loudly** naming the keys. |
| G3* | Key ordering | Group by `named_modules()` topological order (`_ordered_keys`), never raw dict order; traversal is cross-checked against `state_dict()` and raises on disagreement. |
| G4* | Hooks/parametrizations (online) | Activation uses `load_state_dict(strict=False)` with default `assign=False` (in-place `copy_`); the engine never rebinds Parameter objects, so hooks survive. |
| G5 | dtype/device | Slices stored on CPU with original dtype; `assemble` loads to the engine device; `copy_` handles cross-device/dtype-preserving copies. bfloat16 hashes via int16 view. |
| G6* | Strict key coverage | Decomposition partitions the state_dict exactly (`validate_decomposition`); offline assembly uses `strict=True`; save/load validate the slice key set against metadata. |
| G7* | Deterministic variant init | One CPU generator, fixed (block, variant, key) consumption order; same seed ⇒ bit-identical warehouse. Integer tensors never perturbed. |
| G8 | Non-persistent buffers | Excluded everywhere (they are not in `state_dict`); never invent keys for them. |

## 12. Build order (already executed for the core; replay if rebuilding)

1. `contract.py` — everything else consumes `UserModule`/`to_device`.
2. `decompose.py` — needs nothing but torch; `BlockSpec` feeds the warehouse.
3. `warehouse.py` — needs `BlockSpec`; provides storage for engines.
4. `engine.py` — needs `Warehouse`; offline first (it is the semantics
   reference), then online, then assert equivalence.
5. `training.py` — needs engines + contract; canonical step before the loop.
6. `inference.py` — needs engines + contract.
7. `cli.py` — needs all of the above; thin argparse only.
8. `tests/test_smoke_mnist.py` — must pass before any version work starts.
9. `bench_resource.py` + `tests/test_resource_model.py` — the §6.1
   demonstration; counters must equal predictions exactly before trusting
   any resource claim elsewhere.

## 13. Acceptance checklist (pytest-style; all green today)

```
tests/test_smoke_mnist.py::test_end_to_end[offline]            init→finetune→infer, shapes,
tests/test_smoke_mnist.py::test_end_to_end[online]               provenance, verify()==[], BN counters advanced
tests/test_smoke_mnist.py::test_engines_numerically_equivalent  bit-identical slices + equal hashes + equal predictions
tests/test_smoke_mnist.py::test_variant0_reproduces_base        all-zeros config == base checkpoint, torch.equal per key
tests/test_smoke_mnist.py::test_tied_weights_never_split        embed↔head tie lands in one block

tests/test_resource_model.py::test_offline_counters_match_formula    2T·W reads, T·W eager writes — exact
tests/test_resource_model.py::test_online_cpu_counters_match_replay  swap bytes < 2T·W, pool == N·W exact, disk 0
tests/test_resource_model.py::test_predictions_are_engine_agnostic_in_totals
tests/test_resource_model.py::test_cuda_peaks_offline                 [GPU] infer < 1.75·W; train in [4.4, 5.95]·Wp
tests/test_resource_model.py::test_cuda_peaks_online_cpu_pool         [GPU] identical bounds to offline
tests/test_resource_model.py::test_cuda_peaks_online_cuda_pool        [GPU] + N·W pool term, pool residency proven
```

Any change to the core must keep all eleven green (the three `[GPU]` entries
skip cleanly on CPU-only boxes; the counter equalities run everywhere).
Version specs add their own checklists on top.

## 14. Common pitfalls

* **Stale online reads** — after `writeback`, a later `assemble` requesting the
  same `(block, variant)` must see the cached (dirty) slice, not the disk copy.
* **Reference leakage** — never `writeback` the reference configuration; only
  `cfg_update`'s blocks are persisted (step 5 of §7).
* **Optimizer reuse** — a cached Adam carries moments estimated on *other*
  variants' weights; always create it fresh per step.
* **RNG discipline** — the config schedule comes from `random.Random(seed)`
  only. `model_factory()` consumes global torch RNG (offline calls it per
  assemble, online once); nothing in the core may depend on global RNG after
  assembly, or the engines drift apart.
* **`strict=False` abuse** — `strict=False` is for *partial* per-block loads in
  the online engine only; full-cover offline assembly must use `strict=True`
  and version code should too.
* **Editing tensors of an assembled model in place without `writeback`** —
  the warehouse is the single source of truth; un-written changes are lost by
  design (offline) or on `close` (online).
* **Hash drift** — `save_variant` clones to CPU before hashing; if you add a
  new write path, hash exactly what you serialize.
* **`pool_device="cuda"` without headroom** — the pool costs a permanent
  N·W of VRAM on top of the active model (§6.1); check
  `N × state_dict_bytes(model.state_dict())` against free memory first, and
  prefer `prefetch_pool()` at start-up so the cost is paid (and visible)
  before training begins, not mid-run.
* **Misreading the counters** — `EngineStats` counts *logical* slice
  movement; a pool→device byte is PCIe traffic only when pool and compute
  device differ in type (online/cuda swaps are device-to-device and cross no
  bus). Don't sum counters across `reset()` boundaries.
* **Reintroducing foreach/fused optimizers** — they allocate ≈ +2·Wp of
  step-time temporaries and break the §6.1 training bound; the canonical
  step pins `foreach=False` deliberately.
