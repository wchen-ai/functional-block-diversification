# Functional Block Diversification (FBD)

**Source-free block-warehouse diversification of any pretrained PyTorch checkpoint.**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**Functional Block Diversification (FBD)** is a model mechanism: decompose a
model's `state_dict` into **blocks**, keep several diversified **variants** of
each block in a **warehouse**, and assemble complete sub-models on demand —
**without modifying the model's source code**. It needs only (a) a way to
*construct* the architecture (a tiny `model_factory`, or auto-construction via
`torchvision:` / `timm:` / `hf:`) and (b) the checkpoint. "Blocks" are
contiguous groups of `state_dict` keys; "assembling" composes a `state_dict`
and `load_state_dict`s it into a fresh model whose own, unmodified `forward`
runs every pass.

This repository implements the FBD mechanism as a block **warehouse** (the
`warehouse_adopt` package) — the model-agnostic generalisation of the
[SASWISE-UE](https://doi.org/10.1016/j.compbiomed.2025.110258) sub-model
warehouse and the Fed-FBD federated architecture.

**The mechanism is the point; three applications come as bonuses, each
specified for an AI agent to implement** (see
**[`IMPLEMENT_WITH_AI.md`](IMPLEMENT_WITH_AI.md)**):

- **Uncertainty** — a sub-model ensemble from one checkpoint, for calibrated
  uncertainty and risk–coverage.
- **Federated (Fed-FBD)** — block-level isolation from adversarial clients,
  contributor provenance, and surgical unlearning.
- **Active learning** — ensemble disagreement as the acquisition function.

## Two assembly engines

Both expose one API behind `mode={"offline","online"}`; they differ only in
where the *N-variant pool* lives and the per-step transfer that implies
(`agent_specs/00_core_warehouse.md` §6.1 has the measured resource model):

| | pool location | GPU VRAM | per-step transfer | best for |
|---|---|---|---|---|
| `offline` | disk | one sub-model (= the unwrapped model) | full sub-model load from disk | scarce VRAM, federated round-trips |
| `online`, `pool_device="cpu"` | host RAM | one sub-model | only the swapped block over PCIe | many steps on one box (default) |
| `online`, `pool_device="cuda"` | VRAM | pool + one sub-model | ≈ 0 (device-to-device) | small pool, speed-critical |

At **inference** the footprint is a single sub-model — the same GPU as the
unwrapped model. The transient 2× only appears during diversification
training (an update colour plus a frozen reference colour for the consistency
term); see the resource model for the exact `4P / 3P` breakdown, demonstrated
by `tests/test_resource_model.py`.

## Install

```bash
pip install -e .                 # core (torch only)
pip install -e ".[vision,test]"  # + torchvision and the test deps
```

## Quick start

```bash
pytest -q                        # run the smoke + resource tests

# CLI: decompose a checkpoint, fine-tune the warehouse, ensemble-infer
warehouse-adopt init    --help
warehouse-adopt finetune --help
warehouse-adopt infer   --help
```

A ten-minute narrative walk-through is in
[`agent_specs/HUMAN_GUIDE.md`](agent_specs/HUMAN_GUIDE.md).

## Repository layout

```
warehouse_adopt/        Implemented core package
  decompose.py          state_dict → BlockSpec (balanced / top-level / manual / boundary)
  warehouse.py          on-disk variant store + provenance + content hashes
  engine.py             OfflineEngine / OnlineEngine behind make_engine(mode, pool_device=…)
  training.py           diversification step + adaptive consistency balancing
  inference.py          Welford-streaming ensemble + uncertainty
  contract.py           the model-access user-module contract
  cli.py                warehouse-adopt init | finetune | infer
  bench_resource.py     measured GPU/CPU/transfer demonstration
agent_specs/            Specs for AI agents: 00 core + 01/02/03 versions + HUMAN_GUIDE
tests/                  Smoke (MNIST) + resource-model tests
IMPLEMENT_WITH_AI.md    Front door: hand this repo to an agent to build a version
```

## License

[MIT](LICENSE) © 2026 Weijie Chen.
