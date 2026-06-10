# Human Guide — warehouse_adopt in ten minutes

This is the narrative version of the agent specs, for a person. It mirrors
the "Adopt any pretrained checkpoint" section of `SASWISE/README.md`, which
is where this design originated. (Agents: read `00_core_warehouse.md`
instead — that one is normative.)

## The idea

You have a pretrained model. We split its `state_dict` into **blocks**
(stem / stages / head, say), keep **V variants** of every block in an
on-disk **warehouse**, and treat every combination of variants as its own
sub-model — `V^B` models for the price of ~one. Sub-models are *assembled*,
never re-coded: a fresh model from your constructor + `load_state_dict` of
the chosen slices. Your model's own `forward` runs untouched — we never edit
your source. A short fine-tune with a consistency loss makes the variants
genuinely diverse without losing accuracy; after that, the ensemble's
disagreement is a working uncertainty signal, each block knows exactly who
wrote it (provenance), and removing a contributor is a file operation.

## Step 0 — describe your model in five callables

One Python file (see `SASWISE/examples/adopt_mnist_mlp.py` for a complete
MNIST example):

```python
def model_factory() -> nn.Module: ...                   # fresh instance of your architecture
def load_base_checkpoint(model, path) -> None: ...      # load your pretrained weights
def get_dataloaders() -> (train_loader, val_loader): ...
def infer_fn(model, batch) -> Tensor: ...               # one forward pass
def loss_fn(out, batch) -> Tensor: ...                  # task loss
def consistency_fn(out_a, out_b) -> Tensor: ...         # optional; default is MSE
```

No model code at all? Use `--model torchvision:resnet18?num_classes=10`
(or `timm:…` / `hf:…`) and skip `model_factory`.

## Step 1 — decompose and materialise the warehouse

```bash
cd warehouse_adopt
python -m warehouse_adopt.cli init \
    --user-module my_model.py --checkpoint base.pt --output-dir runs/demo \
    --decomposition boundary:conv1+bn1,layer1,layer2,layer3,layer4,avgpool+fc \
    --variants 4
```

This writes `runs/demo/warehouse/` with one `block_<i>/variant_<j>.pt` per
slice plus `warehouse_metadata.json` (block specs, content hashes, and a
provenance trace per slice). Variant 0 is always the *unperturbed* base, so
the original model stays reconstructable. Other decompositions:
`balanced:6`, `top-level`, `manual:<regex>;<regex>;…`.

## Step 2 — diversify

```bash
python -m warehouse_adopt.cli finetune \
    --user-module my_model.py --warehouse runs/demo/warehouse \
    --mode offline --epochs 2
```

Each step samples two random sub-models, trains one against
`task_loss + λ·consistency(loss vs the other)`, and persists only the
touched blocks. λ is auto-balanced (≈10 % of the total loss, 100-step
warm-up) — see `ConsistencyBalance` if you want manual control.

Two engines, same results (bit-identical, it's tested), and you choose where
the N-variant pool lives with `--pool-device`:

* `--mode offline` (pool on **disk**) — assemble from disk every step.
  Best for big models, little RAM, or few steps.
* `--mode online --pool-device cpu` (default) — one resident model,
  per-block hot-swaps from host RAM (one block over PCIe per swap), dirty
  blocks flushed to disk. Best for many steps on one box.
* `--mode online --pool-device cuda` — the whole pool lives in VRAM
  (≈ N × model size) and swaps cross no bus. Fastest, hungriest.

Costs, measured (`python -m warehouse_adopt.bench_resource`): **inference
always fits in the same GPU as your unwrapped model** — one sub-model at a
time, however many you ensemble. Training peaks at about 4× the parameter
bytes (weights + gradients + two Adam moments) plus activations — the frozen
reference sub-model is evaluated *before* the trained one each step, so a
second copy is never resident. The engines measure byte-identical GPU usage;
they differ only in where the pool sits and what each swap transfers.

## Step 3 — ensembled inference with uncertainty

```bash
python -m warehouse_adopt.cli infer \
    --user-module my_model.py --warehouse runs/demo/warehouse \
    --num-configurations 64 --task classification --output preds.pt
```

`preds.pt` holds the ensemble `mean`, per-element `std`, and (for
classification) `predictive_entropy`. **For classification use the entropy,
not the std** — on MNIST the entropy flags misclassifications at AUROC ≈ 0.92
while the logit std is worse than random (≈ 0.44). `std` is the right signal
for regression and segmentation.

## Where to go next

* **Uncertainty reports** (AUROC, risk–coverage, selective prediction):
  spec `01_uncertainty_local.md`.
* **Federated Fed-FBD** (clients own colours, architectural isolation,
  routing, surgical unlearning): spec `02_federated_fbd.md`.
* **Active learning** (entropy/disagreement acquisition loops): spec
  `03_active_learning.md`.

The three are specs for an AI coding agent — see `IMPLEMENT_WITH_AI.md` at
the repository root for how to hand them off. The shared core they build on
is already implemented and tested here:

```bash
python -m pytest tests/test_smoke_mnist.py -q     # 5 passed = you're good
```
