# 01 — Version Spec: Single-Machine Sub-Model-Ensemble Uncertainty

**Prerequisite reading:** `00_core_warehouse.md` (the core is already
implemented and smoke-tested; run `python -m pytest tests/test_smoke_mnist.py -q`
first — expect 5 passed). This file + the core spec are self-contained.

**Goal.** The SASWISE use case: a user has one pretrained checkpoint on one
machine and wants calibrated *uncertainty* for its predictions. The warehouse
turns one checkpoint into `V^B` sub-models; their disagreement is the
uncertainty signal. This version adds the **evaluation and reporting layer**
that turns `EnsemblePrediction` into decisions: misclassification-detection
AUROC, risk–coverage analysis, and a selective-prediction report.

## 1. What this version adds over the core

| Core already provides | This version adds |
|---|---|
| `init/finetune/infer` pipeline, both engines | `uncertainty` subpackage: metrics + report |
| `EnsemblePrediction(mean, std, predictive_entropy)` | AUROC(misclassification \| uncertainty), risk–coverage curve, accuracy@coverage table, ECE |
| `predictions.pt` payload | `uncertainty_report.json` + printed table |
| std-vs-entropy caveat (core §8) | the caveat enforced in code: classification reports use entropy; std-based report requires `--allow-std` |

No core module is modified. New files only.

## 2. New public API

Create `warehouse_adopt/uncertainty/` with:

```python
# warehouse_adopt/uncertainty/metrics.py
def collect_labels(user: UserModule, *, device="cpu") -> torch.Tensor
    # Iterate user.get_dataloaders()[1] once; return the concatenated integer
    # label vector (N,). Batches are (x, y) or dicts with "label"; raise a
    # clear error otherwise telling the user to expose labels in val batches.

def misclassification_auroc(scores: torch.Tensor, correct: torch.Tensor) -> float
    # AUROC of `scores` (higher = more uncertain) for predicting `~correct`.
    # Implement rank-based (Mann-Whitney) AUROC in pure torch — no sklearn
    # dependency; ties handled by midranks.

def risk_coverage(scores: torch.Tensor, correct: torch.Tensor,
                  coverages: Sequence[float] = (1.0, 0.9, 0.8, 0.7, 0.5)) -> list[dict]
    # Sort by ascending uncertainty; for each coverage c keep the c·N most
    # certain samples; return [{"coverage": c, "kept": int, "accuracy": float,
    # "risk": 1-accuracy}, ...]. coverage=1.0 row equals plain accuracy.

def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor,
                               num_bins: int = 15) -> float
    # Standard ECE over softmax(mean) confidences, equal-width bins.

# warehouse_adopt/uncertainty/report.py
@dataclass
class UncertaintyReport:
    num_samples: int
    num_configurations: int
    accuracy: float
    auroc_entropy: float            # AUROC(misclassification | predictive_entropy)
    auroc_std: float                # AUROC(misclassification | mean per-class logit std) — reported for the caveat
    ece: float
    risk_coverage: list[dict]       # rows as returned by risk_coverage()

def build_report(pred: EnsemblePrediction, labels: torch.Tensor) -> UncertaintyReport
    # correct = (pred.mean.argmax(-1) == labels); entropy score from
    # pred.predictive_entropy (REQUIRED — raise if None: task must be
    # classification); std score = pred.std.mean(dim=-1).
def render_report(report: UncertaintyReport) -> str        # aligned text table for the console
def save_report(report: UncertaintyReport, path) -> None   # json.dump(asdict(report), indent=2)
```

## 3. CLI additions (`warehouse_adopt/cli.py` — add ONE subcommand)

```
warehouse-adopt uncertainty-report
    --user-module F --warehouse D
    [--mode offline|online] [--num-configurations 32] [--device auto] [--seed 0]
    [--predictions predictions.pt]      # reuse an existing infer output instead of re-running
    [--output uncertainty_report.json]
```

Behaviour: load predictions from `--predictions` if given, else call
`infer_with_uncertainty(task="classification")`; `collect_labels`;
`build_report`; print `render_report`; save JSON. Exit non-zero if
`predictive_entropy` is missing.

## 4. Data schemas

`uncertainty_report.json`:

```json
{
  "num_samples": 10000,
  "num_configurations": 32,
  "accuracy": 0.981,
  "auroc_entropy": 0.92,
  "auroc_std": 0.44,
  "ece": 0.013,
  "risk_coverage": [
    {"coverage": 1.0, "kept": 10000, "accuracy": 0.981, "risk": 0.019},
    {"coverage": 0.9, "kept": 9000,  "accuracy": 0.995, "risk": 0.005}
  ]
}
```

`predictions.pt` is unchanged from the core (`mean`, `std`,
`predictive_entropy`, `num_configurations`, `block_configurations`).

## 4.1 Resource footprint (core §6.1 applies verbatim)

Ensemble inference is the **A = 1** case of the core resource model:
configurations are evaluated **sequentially**, so the GPU footprint is one
assembled sub-model + activations — *the same VRAM as the unwrapped original
model* — no matter how large `--num-configurations` is. Cost of more
configurations is wall-clock and per-config transfer only (offline: W
disk-read per config; online: only the swapped blocks). Choose `--mode`/
`--pool-device` by the core decision rule: a one-shot report on a big model →
offline/disk; repeated reports on one box → online/cpu; pool fits in VRAM →
online/cuda (adds the N·W pool term but zero-bus swaps). Never parallelise
configurations across models to "speed up" the report without saying so —
that multiplies VRAM by the number of concurrent sub-models.

## 5. Invariants

1. **Entropy is the classification score.** `build_report` uses
   `predictive_entropy` for the headline AUROC; the std AUROC is computed
   only to *demonstrate* the caveat and must be labelled as such in output.
2. Risk–coverage is monotone in expectation: accuracy at coverage 0.5 ≥
   accuracy at coverage 1.0 for a useful uncertainty signal (do not assert in
   code; assert in the acceptance test for the trained warehouse).
3. The report never re-orders samples: `scores[i]`, `labels[i]`, `mean[i]`
   refer to the same validation sample (loader must not shuffle — document
   in the CLI help that val loaders should be `shuffle=False`).
4. Pure-torch metrics: no sklearn/scipy imports anywhere in the subpackage.

## 6. Build plan (dependency order)

1. `warehouse_adopt/uncertainty/__init__.py` — re-export the API of §2.
2. `metrics.py::misclassification_auroc` — pure function, unit-test against
   hand-computable cases (e.g. perfect separation = 1.0, anti-separation = 0.0,
   random ≈ 0.5) before anything depends on it.
3. `metrics.py::risk_coverage`, `expected_calibration_error`, `collect_labels`.
4. `report.py` — depends on metrics + core `EnsemblePrediction`.
5. CLI subcommand wiring in `cli.py` (import inside the subcommand function to
   keep core CLI import-light).
6. Acceptance test (§7).

## 7. Acceptance test (write as `tests/test_uncertainty_mnist.py`)

End-to-end on MNIST (CPU ok, GPU faster). Budget: a few minutes.

```python
def test_uncertainty_mnist_auroc_and_risk_coverage(tmp_path):
    # 1. Train a small base model on MNIST to >=95% val accuracy
    #    (e.g. the smoke test's MLPBN, 2 epochs, Adam lr=1e-3, full train set).
    # 2. init: decompose_boundary(base, [["fc1","bn1"], "fc2", "fc3"]),
    #    num_variants=3, noise_std=1e-3, seed=0.
    # 3. finetune_warehouse(mode="offline", epochs=1, lr=1e-4,
    #    consistency=ConsistencyBalance() (adaptive default), seed=0,
    #    full MNIST train loader, batch 128).
    # 4. pred = infer_with_uncertainty(num_configurations=27, task="classification")
    #    over the FULL 10k-test loader (shuffle=False).
    # 5. report = build_report(pred, collect_labels(user))
    assert report.accuracy >= 0.95
    assert report.auroc_entropy > 0.85          # REQUIRED acceptance threshold
    assert report.auroc_entropy > report.auroc_std   # the caveat, demonstrated
    rc = {row["coverage"]: row["accuracy"] for row in report.risk_coverage}
    assert rc[0.5] >= rc[1.0]                    # selective prediction helps
    # 6. Print render_report(report) so the run leaves a human-readable table.
```

Also keep a tiny pure-unit test for `misclassification_auroc` (3 hand cases)
and one for `risk_coverage` row arithmetic.

**Definition of done:** core smoke test still 5-passed; the two new test files
pass; `warehouse-adopt uncertainty-report --help` works; running the CLI on
the test warehouse prints the table and writes valid JSON matching §4.

## 8. Common pitfalls (this version)

* **Using `std` as the classification score** — the whole point of the
  caveat; entropy is the headline number (measured MNIST reference: entropy
  AUROC ≈ 0.92 vs std ≈ 0.44).
* **Shuffled val loader** — silently misaligns `labels` and `pred.mean`;
  document `shuffle=False`, and in `collect_labels` iterate the same loader
  object the inference used.
* **Too few configurations** — with `V=3, B=3` there are only 27 configs;
  `num_configurations=27` enumerates all (core `sample_configurations` does
  this automatically). Don't sample 64 from 27.
* **AUROC ties** — integer-valued or saturated entropy values produce ties;
  use midrank handling, not strict comparison counting.
* **BN buffers** (core G1) — they are inside the blocks; nothing extra to do,
  but if accuracy collapses after heavy diversification, offer
  `recalibrate_bn` before evaluation rather than retraining.
