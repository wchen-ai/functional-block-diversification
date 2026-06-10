# 03 — Version Spec: Uncertainty-Driven Active Learning

**Prerequisite reading:** `00_core_warehouse.md` (core already implemented and
smoke-tested — run `python -m pytest tests/test_smoke_mnist.py -q` first;
expect 5 passed). This file + the core spec are self-contained.

**Goal.** An acquisition loop that uses the warehouse ensemble's uncertainty
(predictive entropy and/or inter-variant disagreement) as the acquisition
function: train the warehouse on the labelled pool, score the unlabelled
pool, select the top-`b` most informative samples, label them with a
simulated oracle, repeat. Must work with **both** engines (`mode="offline"`
and `"online"`).

## 1. What this version adds over the core

| Core already provides | This version adds |
|---|---|
| `finetune_warehouse` (train on a loader) | pool management (labelled/unlabelled index sets) |
| `infer_with_uncertainty` (mean/std/entropy) | acquisition scoring incl. disagreement (needs per-config vote tracking) |
| both engines | the acquisition loop + simulated oracle + learning curves |
| CLI `init/finetune/infer` | `warehouse-adopt al run` + loop-state/curve files |

New subpackage `warehouse_adopt/active/`; core modules unmodified — with one
extension: acquisition scoring needs per-configuration *argmax votes*, which
`EnsemblePrediction` does not retain. Implement `score_pool` in the new
subpackage by iterating engine-assembled configurations directly (same
pattern as `infer_with_uncertainty`; do NOT modify `inference.py`).

## 2. New public API (`warehouse_adopt/active/`)

```python
# active/pool.py
@dataclass
class Pool:
    labelled: list[int]            # indices into the base train dataset
    unlabelled: list[int]
    history: list[dict]            # one entry per completed acquisition round (schema §4)
    def assert_disjoint(self) -> None       # labelled ∩ unlabelled == ∅, raises otherwise
    def save(self, path) -> None; @classmethod load(cls, path) -> "Pool"

def make_initial_pool(num_samples: int, initial_labelled: int, *, seed: int = 0) -> Pool
    # Random (seeded) initial labelled set; the rest unlabelled.

# active/oracle.py
class SimulatedOracle:
    def __init__(self, dataset)            # any dataset whose __getitem__ -> (x, y)
    def label(self, indices: list[int]) -> list[int]
    # Returns the ground-truth labels for `indices`. The ONLY component allowed
    # to read labels of unlabelled samples — everything else must treat the
    # unlabelled pool as unlabelled (invariant §5.2).

# active/acquisition.py
def score_pool(warehouse, user_factory_loader, *, mode="offline",
               num_configurations=8, strategy="entropy", device="cpu",
               pool_device=None, seed=0) -> torch.Tensor
    # Scores the CURRENT unlabelled pool, one float per pool sample
    # (higher = acquire first). Iterates `num_configurations` sampled configs
    # (core sample_configurations) SEQUENTIALLY — the A=1 inference footprint
    # of core §6.1: one assembled sub-model at a time, i.e. ONE unwrapped-
    # model forward pass worth of VRAM per colour/config, regardless of
    # num_configurations. (A batched variant scoring j configs concurrently
    # costs j× the model VRAM; if you add it, gate it behind an explicit
    # `concurrent_configs=j` argument defaulting to 1.) Per config,
    # engine.assemble -> softmax probs over the pool loader; accumulates:
    #   mean probs (Welford or running sum), and per-config argmax votes.
    # strategy:
    #   "entropy"      -> entropy of the mean softmax (predictive entropy)
    #   "disagreement" -> 1 - (vote count of the modal class / num_configurations)
    #                     (variation-ratio across sub-model votes)
    #   "margin"       -> negative (p1 - p2) margin of the mean softmax
    #   "random"       -> uniform random scores from random.Random(seed) (baseline)

def select_top_b(scores: torch.Tensor, pool_unlabelled: list[int], b: int) -> list[int]
    # Deterministic: ties broken by ascending dataset index. Returns dataset indices.

# active/loop.py
@dataclass
class ALConfig:
    rounds: int = 8                # acquisition rounds R
    budget_per_round: int = 64     # b
    initial_labelled: int = 128
    strategy: str = "entropy"      # acquisition strategy name (see score_pool)
    mode: str = "offline"          # engine mode, both must work
    finetune_epochs: int = 1
    num_variants: int = 3
    num_configurations: int = 8    # configs used for scoring
    lr: float = 1e-3
    noise_std: float = 1e-2
    seed: int = 0
    device: str = "cpu"

def run_active_learning(user_template, base_train_dataset, val_loader,
                        decompose_fn, cfg: ALConfig, *, workdir: Path) -> Pool
    # Round t = 0..R-1:
    #  1. Build the labelled loader from Pool.labelled (shuffle with a
    #     torch.Generator seeded (cfg.seed, t)).
    #  2. REBUILD the warehouse from the SAME base checkpoint each round
    #     (re-init under workdir/round_<t>/warehouse with cfg.num_variants,
    #     cfg.noise_std, seed=cfg.seed) and finetune_warehouse on the
    #     labelled loader (mode=cfg.mode, epochs=cfg.finetune_epochs,
    #     lr=cfg.lr, seed=hash((cfg.seed, t)) & 0xFFFFFFFF).
    #     Rationale: retrain-from-base each round is the standard AL protocol
    #     and keeps rounds comparable; warm-starting is a documented variant,
    #     not the default.
    #  3. Evaluate: mean-ensemble accuracy on val_loader -> history.
    #  4. score_pool on the unlabelled pool (strategy=cfg.strategy);
    #     select_top_b; SimulatedOracle.label; move indices unlabelled->labelled.
    #  5. Pool.assert_disjoint(); append history entry; Pool.save(workdir/pool.json).
    # Returns the final Pool (R history entries + final accuracy entry).

def learning_curve(pool: Pool) -> list[tuple[int, float]]   # [(num_labelled, val_acc), ...]
```

`user_template` is a `UserModule` whose `get_dataloaders` is IGNORED by the
loop (the loop owns the data); document that only `model_factory`,
`infer_fn`, `loss_fn`, `consistency_fn` are used. Provide
`make_loop_user(user_template, labelled_loader, val_loader) -> UserModule`
(dataclasses.replace with a closure for `get_dataloaders`) so the core
`finetune_warehouse` signature is reused unchanged.

## 3. CLI addition

```
warehouse-adopt al run --user-module F --workdir D
    [--strategy entropy|disagreement|margin|random] [--mode offline|online]
    [--rounds 8] [--budget 64] [--initial 128] [--variants 3]
    [--num-configurations 8] [--epochs 1] [--lr 1e-3] [--seed 0] [--device auto]
    [--curves curves.csv]
```

Runs `run_active_learning` with the user module's train dataset as the pool
source (the user module's `get_dataloaders()[0].dataset` is the base
dataset; document this requirement) and writes `pool.json` + `curves.csv`.

## 4. Data schemas

`pool.json` (written every round; restartable):

```json
{
  "labelled": [3, 17, 42],
  "unlabelled": [0, 1, 2],
  "history": [
    {"round": 0, "num_labelled": 128, "val_accuracy": 0.74,
     "strategy": "entropy", "acquired": [501, 502], "mode": "offline",
     "seed": 0}
  ]
}
```

`curves.csv`: header `num_labelled,val_accuracy,strategy,seed,mode`, one row
per (round, run).

## 5. Invariants

1. **Budget accounting:** after round t, `len(labelled) ==
   initial_labelled + t·budget_per_round`; an acquired index appears in
   exactly one round's `acquired` list.
2. **No label leakage:** only `SimulatedOracle.label` touches labels of
   unlabelled indices; `score_pool` consumes images only (its loader yields
   `(x, idx_or_dummy)`); grep-level check: `active/acquisition.py` contains
   no access to `dataset.targets`/labels.
3. **Determinism:** identical `ALConfig` (incl. seed) ⇒ identical acquisition
   sequence and identical `pool.json` (both engine modes; the engines are
   numerically equivalent by core §6, so the acquired indices must match
   between `mode="offline"` and `mode="online"` runs with equal seeds).
4. Pool partition stays a partition: `assert_disjoint` after every mutation.

## 6. Build plan (dependency order)

1. `active/__init__.py`, `active/pool.py` — pure python, unit-test
   save/load/disjointness first.
2. `active/oracle.py` — trivial but isolates label access (invariant 5.2).
3. `active/acquisition.py::score_pool` — the only torch-heavy piece; verify
   on a hand-built 2-config toy that "entropy" and "disagreement" rank an
   ambiguous sample above a confident one before wiring the loop.
4. `active/loop.py` — consumes 1–3 + core `finetune_warehouse`; implement
   `make_loop_user` first, then the round loop, then `learning_curve`.
5. CLI `al run` + `curves.csv` writer.
6. Acceptance test (§7).

## 7. Acceptance test (write as `tests/test_active_learning.py`)

Benchmark: MNIST (already under `~/data`; fall back to FashionMNIST download
if missing). Keep it CPU-feasible: base dataset = 4 000-sample fixed subset,
val = 1 000 samples, model = the smoke test's `MLPBN`,
`decompose_boundary([["fc1","bn1"],"fc2","fc3"])`, `num_variants=3`.

```python
def test_warehouse_uncertainty_beats_random(tmp_path):
    cfgs = [ALConfig(strategy=s, rounds=5, budget_per_round=128,
                     initial_labelled=128, seed=seed, mode="offline")
            for s in ("entropy", "random") for seed in (0, 1, 2)]
    # run all six; aggregate learning_curve per strategy (mean over seeds)
    # MATCHED LABELLING BUDGET comparison:
    assert mean_final_acc["entropy"] > mean_final_acc["random"]      # final budget point
    dominated = sum(mean_curve["entropy"][k] >= mean_curve["random"][k]
                    for k in range(1, len(curve)))                    # skip round 0 (identical pools)
    assert dominated / (len(curve) - 1) >= 0.6                       # curve dominance at >=60% of checkpoints
    # and print/persist the two curves (curves.csv) — the learning-curve
    # comparison is the deliverable, not just the asserts.

def test_engine_modes_agree(tmp_path):
    # ALConfig(seed=0, rounds=2, strategy="entropy") run with mode="offline"
    # and mode="online" -> identical acquired index sequences (invariant 5.3).

def test_budget_accounting_and_disjointness(tmp_path):
    # rounds=2 on a tiny synthetic pool: invariant 5.1 + 5.4 hold; pool.json
    # round-trips through Pool.load.
```

**Definition of done:** core smoke test still 5-passed; the three tests above
green (the comparison test may be marked `@pytest.mark.slow`; it must still
run in <~15 min CPU); `warehouse-adopt al run --help` works and a 2-round CLI
run writes `pool.json` + `curves.csv` matching §4.

## 8. Common pitfalls (this version)

* **Scoring with a shuffled pool loader** — scores must align with pool
  indices; build the pool loader with `shuffle=False` over
  `Subset(dataset, pool.unlabelled)` and map positions back to dataset
  indices explicitly.
* **`std` as the classification acquisition score** — same caveat as the
  core (§8): use predictive entropy or vote disagreement, never raw logit std.
* **Warm-starting silently** — reusing last round's warehouse changes the
  comparison semantics; the default is retrain-from-base (step 2). If you add
  warm-start as an option, it must be off by default and recorded in history.
* **Random baseline reusing the model's RNG** — give "random" its own
  `random.Random(seed)`; otherwise the baseline correlates with training noise.
* **Round-0 comparisons** — all strategies share the same initial pool;
  curves only diverge from round 1. Skip round 0 in dominance counts.
* **Tiny `num_configurations`** — disagreement needs ≥ ~8 votes to be
  non-degenerate with `V=3, B=3`; entropy works with fewer. Document the
  trade-off in `--num-configurations` help text.
* **Online-engine state across rounds** — the loop rebuilds the warehouse
  each round; always `engine.close()` (the core `finetune_warehouse` already
  does) so a stale resident model never leaks into the next round's scoring.
* **Resource expectations (core §6.1)** — pool scoring and round evaluation
  are A = 1 inference: the unwrapped model's VRAM, sequentially per config.
  The per-round retrain is the canonical training step (weights + grads +
  2 Adam moments ≈ 4·Wp on the update model, reference time-multiplexed).
  An AL loop's many short phases favour `mode="online"` with `pool_device=
  "cpu"` on a workstation; the per-round warehouse rebuild makes
  offline/disk equally correct, just slower.
