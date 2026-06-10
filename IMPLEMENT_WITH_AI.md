# Implement this with an AI coding agent

This repository implements **Functional Block Diversification (FBD)** — a
source-free mechanism for any pretrained PyTorch checkpoint: split a model's
`state_dict` into blocks, keep several diversified variants of each block in a
warehouse (the `warehouse_adopt` package), and assemble/train/evaluate
sub-models — without ever editing the model's source. The **shared core is
already implemented and tested** (`warehouse_adopt/`, `tests/`); the three
application *versions* are written as precise specs an AI agent can implement.

## How to use this repo with an agent

1. Open this repository in an AI coding agent (Claude Code, Codex, …).
2. Tell it: **"Read `agent_specs/00_core_warehouse.md` plus the one version
   spec I name, then implement that version against the existing
   `warehouse_adopt/` core. Run the smoke test first."**
3. Point it at one of:
   - [`agent_specs/01_uncertainty_local.md`](agent_specs/01_uncertainty_local.md)
   - [`agent_specs/02_federated_fbd.md`](agent_specs/02_federated_fbd.md)
   - [`agent_specs/03_active_learning.md`](agent_specs/03_active_learning.md)

Each version spec is self-contained when read together with the core spec:
it lists the extra API, data schemas, invariants, a dependency-ordered build
plan, and a runnable acceptance test the agent can self-verify against.

## Pick a version (5-line decision guide)

- **Uncertainty (01)** — one machine, one checkpoint; you want calibrated
  uncertainty / risk–coverage from a sub-model ensemble. Start here.
- **Federated (02)** — many clients; you need block-level isolation from
  adversarial clients, contributor provenance, and surgical unlearning.
- **Active learning (03)** — you have a large unlabelled pool and a labelling
  budget; use the ensemble's disagreement to choose what to label next.
- All three share the same warehouse, decomposition, and the two assembly
  engines (offline disk / online RAM or VRAM — see §6.1 of the core spec).
- For a human-readable tour instead of a spec, read
  [`agent_specs/HUMAN_GUIDE.md`](agent_specs/HUMAN_GUIDE.md).

## Verify the core before building anything

```bash
cd warehouse_adopt && pip install -e ".[test]" && pytest -q   # expect: all passed
```
