# 02 — Version Spec: Federated Fed-FBD (isolation, provenance, unlearning)

**Prerequisite reading:** `00_core_warehouse.md` (core already implemented and
smoke-tested — run `python -m pytest tests/test_smoke_mnist.py -q` first;
expect 5 passed). This file + the core spec are self-contained.

**Goal.** Federated training in which **colours are variants**: the core
warehouse's variant axis becomes the Fed-FBD colour axis. Each client may
write only the colours it owns; the server stores returned blocks by **direct
replacement** (never averaging) — that is exactly `Warehouse.save_variant` —
so a block never written by client *c* provably contains no information from
*c*. On top of that: per-contributor provenance, inference-time routing, and
surgical unlearning.

**Reference code (read-only inspiration, never imported):**
`fbd_transfer/fbd_record/fbd_settings.py` (`FBD_INFO`, `FBD_TRACE`),
`fbd_transfer/fbd_utils.py` (`FBDWarehouse.store_weights` = direct
replacement), `fbd_transfer/request_plan.json` / `update_plan.json` (plan
shapes), `fbd_transfer/test_routing_strategies.py` (routing strategies),
`fbd_transfer/run_e8_unlearning.py` (`unlearn_aggregate`, `unlearn_reinit`,
MIA harness).

## 1. What this version adds over the core

| Core already provides | This version adds |
|---|---|
| `Warehouse` (variants, provenance, hashes) | colour semantics + write-permission enforcement |
| direct-replacement `save_variant` | generalised plan generator `(B, N, C, r)` |
| engines + `diversification_step` | in-process federated simulation loop |
| `variants_written_by` | surgical unlearning (exact + aggregate) |
| `infer_with_uncertainty` | per-colour routing: mean / max-conf / top-k / learned / oracle |
| — | membership-inference (MIA) report pre/post unlearning |

New subpackage `warehouse_adopt/federated/`; core modules unmodified.

## 2. Plan generation (`federated/plans.py`)

Generalise the hard-coded `FBD_INFO` (which is the special case
`B=6, N=6, C=6, r=3`) to arbitrary `(blocks B, colours N, clients C,
redundancy r)`.

```python
@dataclass
class FBDPlan:
    num_blocks: int                     # B — from the warehouse decomposition
    num_colours: int                    # N — warehouse num_variants
    num_clients: int                    # C
    redundancy: int                     # r — owners per colour
    rounds: int                         # R — total communication rounds
    ownership: dict[int, list[int]]     # colour -> sorted owner client ids (len == r)
    clients: dict[int, list[int]]       # client -> owned colours (len == k = N*r/C)
    schedule: dict[int, dict[int, int]] # round -> {client -> colour to UPDATE}
    def colours_of(self, client_id) -> list[int]
    def owners_of(self, colour) -> list[int]
    def update_colour(self, round_, client_id) -> int
    def regularizer_colours(self, round_, client_id) -> list[int]   # owned minus update
    def save(self, path) -> None;  @classmethod load(cls, path) -> "FBDPlan"

def make_fbd_plan(num_blocks, num_colours, num_clients, redundancy, rounds, *, seed=0) -> FBDPlan
def validate_plan(plan) -> None      # raises ValueError on any violated invariant below
```

**Construction (deterministic given `seed`).** Require
`num_colours * redundancy % num_clients == 0`; let
`k = num_colours * redundancy // num_clients` (colours per client). Permute
the colour list with `random.Random(seed)`; client `i` owns the cyclic
interval `perm[(i*k + j) % N] for j in range(k)`. This gives every client
exactly `k` distinct colours and every colour exactly `r` owners (the C
length-`k` intervals tile `[0, N)` exactly `r` times). Schedule: round `t`
assigns client `c` its `own[c][t % k]` colour (rotation — mirrors the 3-round
rotation in `FBD_INFO["training_plan"]`).

**Plan invariants (checked by `validate_plan`):**

1. every colour has exactly `r` owners; every client owns exactly `k`
   distinct colours;
2. `schedule[t][c] ∈ clients[c]` for all `t, c`; over any `k` consecutive
   rounds each client updates each owned colour exactly once;
3. ownership and clients maps are mutually consistent inverses;
4. `r >= 2` is required for aggregate unlearning to have donors (warn at
   construction if `r == 1`: only exact unlearning will be possible).

**Shipping / request / update plans** are *derived views*, not stored state:

```python
def shipping_list(plan, round_, client_id) -> list[tuple[int, int]]
    # all (block_id, colour) for every owned colour — what the server sends.
def request_list(plan, round_, client_id) -> list[tuple[int, int]]
    # (block_id, update_colour) for every block — what the server takes back.
def update_plan_entry(plan, round_, client_id) -> dict
    # {"model_to_update": colour, "model_as_regularizer": [colours...]}
    # — same shape as fbd_transfer/update_plan.json, with colour ints
    #   instead of FBD_TRACE block-id strings.

def shipping_bytes(plan, warehouse, round_, client_id) -> int
    # Exact downlink bytes for the round: Σ over shipping_list of the block's
    # slice bytes (state_dict_bytes of the stored slice). With full-colour
    # shipping this is k·W per client (k = owned colours, W = model bytes).
def request_bytes(plan, warehouse, round_, client_id) -> int
    # Exact uplink bytes: Σ over request_list = W (one full colour) per client.
```

`FBDPlan.save` JSON schema:

```json
{
  "version": 1,
  "num_blocks": 6, "num_colours": 6, "num_clients": 6, "redundancy": 3, "rounds": 30,
  "ownership": {"0": [0, 2, 4], "1": [0, 3, 5]},
  "clients":   {"0": [0, 1, 2], "1": [3, 4, 5]},
  "schedule":  {"0": {"0": 0, "1": 3}, "1": {"0": 1, "1": 4}}
}
```

## 3. Server, clients, simulation (`federated/server.py`, `client.py`, `simulate.py`)

```python
# server.py
class FederatedServer:
    def __init__(self, warehouse: Warehouse, plan: FBDPlan)
        # Requires warehouse.num_variants == plan.num_colours and
        # warehouse.num_blocks == plan.num_blocks; snapshots variant 0? NO —
        # snapshots EVERY (block, colour) slice hash at construction for audit.
    def ship(self, round_, client_id) -> dict[tuple[int, int], dict[str, Tensor]]
        # {(block, colour): slice} for shipping_list(plan, round_, client_id).
    def receive(self, round_, client_id, slices: dict[tuple[int, int], dict[str, Tensor]]) -> None
        # WRITE-PERMISSION ENFORCEMENT: every (block, colour) key must satisfy
        # client_id in plan.owners_of(colour) AND colour == plan.update_colour(round_, client_id);
        # otherwise raise PermissionError and store NOTHING from this payload.
        # Accepted slices -> warehouse.save_variant(block, colour, slice,
        #                       writer_id=f"client_{client_id}", round=round_)
        # Direct replacement; no averaging anywhere on the server.

# client.py
@dataclass
class PoisonSpec:
    client_id: int
    kind: str                  # "label_flip" | "noise"
    noise_std: float = 0.5     # for kind="noise": N(0, noise_std) added to every received tensor before training
    # label_flip: client trains on (num_classes - 1 - y) labels.

class LocalClient:
    def __init__(self, client_id, user: UserModule, train_loader, *,
                 device="cpu", poison: PoisonSpec | None = None)
    def train_round(self, round_, plan, received, *, lr=1e-4, weight=0.1,
                    local_steps=None, seed=0) -> dict[tuple[int, int], dict[str, Tensor]]
        # 1. Build a PRIVATE in-memory Warehouse-like view from `received`
        #    (implementation detail: a temp Warehouse in a TemporaryDirectory,
        #    or an in-RAM dict engine — either is fine; it must contain ONLY
        #    the owned colours that were shipped).
        # 2. cfg_update = {b: update_colour for all b};
        #    cfg_ref    = {b: reg_colour}, reg_colour cycled per step from
        #    plan.regularizer_colours(round_, client_id).
        # 3. Run `local_steps` (default: one pass over train_loader) of the
        #    CORE canonical diversification_step with a per-client RNG —
        #    random.Random(hash((seed, round_, client_id)) & 0xFFFFFFFF) —
        #    and the client's OWN loader only.
        # 4. Return {(block, update_colour): trained slice} for ALL blocks.
        # Poisoning (if self.poison): label_flip remaps labels in loss_fn's
        # batch; noise perturbs the received update-colour slices before step 1.
        # POISONING NEVER TOUCHES THE RETURN KEY SET — the server's permission
        # check is what guarantees isolation, not client honesty.

# simulate.py
def partition_data(dataset, num_clients, *, scheme="iid", alpha=1.0, seed=0) -> list[torch.utils.data.Subset]
    # "iid": shuffled equal split. "dirichlet": per-class Dirichlet(alpha) shares
    # (reference: fbd_transfer uses the same scheme; smaller alpha = more skew).

@dataclass
class RoundRecord: round: int; client_updates: dict[int, int]; colour_hashes: dict[str, str]

def run_federated_simulation(user, warehouse, plan, *, make_client_loader,
                             rounds=None, mode="offline", device="cpu", lr=1e-4,
                             weight=0.1, local_steps=None, seed=0,
                             poison: PoisonSpec | None = None,
                             absent_clients: frozenset[int] = frozenset()) -> list[RoundRecord]
    # Sequential in-process loop (no Flower dependency; a Flower adapter can be
    # added later following fbd_transfer/fbd_strategy.py):
    #   for t in range(rounds or plan.rounds):
    #     for c in range(plan.num_clients):                # fixed ascending order
    #       if c in absent_clients: continue
    #       payload = client_c.train_round(t, plan, server.ship(t, c), ...)
    #       server.receive(t, c, payload)
    #   record per-round content hashes of every (block, colour).
    # DETERMINISM CONTRACT: all randomness is derived from (seed, round, client);
    # clients never share RNG, loaders are per-client with shuffle driven by a
    # per-client torch.Generator. This is what makes the isolation test below
    # a bit-identity test rather than a statistical one.
```

## 3.1 Resource accounting (core §6.1 applied to a federated round)

A federated round-trip **is the offline/disk pattern of core §6.1** played
over the network: the warehouse lives server-side on disk (N·W bytes across
B·N files); each round, client *c* receives `shipping_bytes(plan, wh, t, c)`
= k·W (its k owned colours) and returns `request_bytes` = W (the one update
colour) — direct replacement on receipt, no server-side model ever
assembled. Quantify and log both per round: `history.json` rounds gain
`"bytes_down": {client: int}, "bytes_up": {client: int}` (exact, from
`state_dict_bytes`, mirroring the core's `EngineStats` discipline).

Client-side, `train_round` runs the core canonical step on its private view:
weight residency A = 1 (the regularizer colour is time-multiplexed before
the update colour, core §7), training extras +3Wp on the update model only,
transient host staging ≤ k·W for the received slices. Server-side evaluation
(routing, §4) is A = 1 inference: one colour assembled at a time = the
unwrapped model's footprint.

## 4. Routing at inference (`federated/routing.py`)

```python
def colour_outputs(warehouse, user, *, colours=None, mode="offline",
                   device="cpu") -> tuple[dict[int, Tensor], Tensor]
    # For each colour m: engine.assemble({b: m for b in blocks}) -> softmax
    # probs over the val loader -> (probs[m] of shape (Nval, K), labels (Nval,)).

def route_mean(probs)                         -> Tensor   # average all colours
def route_max_confidence(probs)               -> Tensor   # per sample, the colour with highest max-softmax
def route_top_k(probs, k=2)                   -> Tensor   # per sample, mean of the k most confident colours
def route_oracle(probs, labels)               -> Tensor   # per sample, first colour predicting correctly (fallback: max-conf)
def fit_learned_router(probs_val, labels_val) -> "Router" # logistic regression on the concatenated per-colour
                                                          #   probability vectors (input dim N*K, output N colour logits),
                                                          #   trained on a held-out HALF of val; pure torch, Adam, 200 steps
def route_learned(router, probs)              -> Tensor

def evaluate_routing(warehouse, user, *, k=2, device="cpu", mode="offline") -> dict
    # Returns {"mean": {"acc": .., "auc": ..}, "max_conf": …, "top_k": …,
    #          "learned": …, "oracle": …} — multiclass AUC = macro one-vs-rest,
    # computed with a pure-torch rank AUROC (no sklearn).
```

Caveats to encode in docstrings: the **oracle** maximises per-sample
correctness, which upper-bounds accuracy but does **not** upper-bound AUC;
the learned router must be fit on a split disjoint from the reported split.

## 5. Surgical unlearning (`federated/unlearning.py`) and MIA (`federated/mia.py`)

```python
def find_client_blocks(warehouse, plan, client_id) -> list[tuple[int, int]]
    # Union of (a) plan-implied: every (block, colour) with client in owners_of(colour),
    # and (b) provenance-audited: warehouse.variants_written_by(f"client_{client_id}").
    # (a) is ground truth; (b) must be a subset of (a) — assert it, this is the audit.

def unlearn_aggregate(warehouse, plan, client_id, *, round_=None) -> UnlearnReport
    # For every owned colour m of the client, for every block b:
    #   donors = [(b, m') for m' not owned by client]   # plan guarantees len >= 1 when r < N·…; raise if empty
    #   replacement[key] = mean over donors of slice[key]      (floating tensors)
    #   replacement[key] = donors[0] slice[key]                (integer tensors, e.g. num_batches_tracked)
    #   warehouse.save_variant(b, m, replacement, writer_id=f"unlearn:client_{client_id}", round=round_ or 0)
    # Mirrors fbd_transfer/run_e8_unlearning.py::unlearn_aggregate, plus provenance.

def unlearn_exact(warehouse, plan, client_id, base_snapshot_dir, *, round_=None) -> UnlearnReport
    # For (block, colour) EXCLUSIVELY written by the client (r==1 colours, or
    # provenance shows no other writer): restore the pristine init-time slice
    # from `base_snapshot_dir` (a copy of the warehouse made right after
    # initialize_variants — the fed CLI's init step MUST create it).
    # Exact deletion: the restored bytes predate any client data.

@dataclass
class UnlearnReport:
    client_id: int; method: str
    replaced: list[tuple[int, int]]       # (block, colour) actually rewritten
    donor_counts: dict[str, int]          # "block/colour" -> number of donors averaged
    utility_before: float | None; utility_after: float | None   # ensemble accuracy, filled by the CLI

# mia.py — confidence-thresholding membership inference (Yeom-style; the
# shadow-model attack of fbd_transfer/run_e8_unlearning.py is an allowed
# upgrade but not required for acceptance)
def mia_scores(model, loader, user, *, device="cpu") -> Tensor      # per-sample max-softmax confidence
def mia_auc(warehouse, user, member_loader, nonmember_loader, *,
            colours=None, mode="offline", device="cpu") -> dict
    # For each colour and for the mean-ensemble: AUROC(member vs non-member |
    # confidence) using the pure-torch midrank AUROC. Returns
    # {"per_colour": {m: auc}, "ensemble": auc}.
```

**Unlearning invariants:**

1. After `unlearn_aggregate(c)`, no rewritten slice equals its pre-unlearning
   bytes (hash must change) unless all donors were already identical.
2. Provenance after unlearning shows the `unlearn:client_<c>` writer as the
   last entry of every replaced slice — deletion is auditable.
3. Blocks NOT owned by `c` are untouched: their hashes before/after the call
   are identical.
4. `r == 1` colours cannot be aggregate-unlearned (no donors) — `unlearn_exact`
   with the base snapshot is the only path; raise a clear error otherwise.

## 6. CLI additions (one `fed` subcommand group in `cli.py`)

```
warehouse-adopt fed make-plan --blocks B --colours N --clients C --redundancy r
                              --rounds R [--seed 0] --output plan.json
warehouse-adopt fed simulate  --user-module F --warehouse D --plan plan.json
                              [--mode offline|online] [--rounds R] [--lr 1e-4]
                              [--partition iid|dirichlet] [--alpha 1.0] [--seed 0]
                              [--poison-client C --poison-kind label_flip|noise]
                              [--absent-clients 2,5] [--history history.json]
warehouse-adopt fed route     --user-module F --warehouse D [--k 2] [--output routing.json]
warehouse-adopt fed unlearn   --user-module F --warehouse D --plan plan.json
                              --client C [--method aggregate|exact]
                              [--base-snapshot D2] [--output unlearn_report.json]
warehouse-adopt fed mia       --user-module F --warehouse D --plan plan.json
                              --client C [--output mia.json]
                              # member set = client C's training shard;
                              # non-member set = the validation set
```

`fed simulate` must create `<warehouse>/../base_snapshot/` (full copy of the
freshly-initialised warehouse) before round 0 if it does not exist — exact
unlearning depends on it.

`history.json` schema: `{"rounds": [{"round": 0, "client_updates": {"0": 2},
"colour_hashes": {"0/0": "sha256:…"}, "bytes_down": {"0": 32108832},
"bytes_up": {"0": 5351472}}], "final_eval": {...}}` — the byte fields are
exact per-client traffic from `shipping_bytes`/`request_bytes` (§3.1).

## 7. Build plan (dependency order)

1. `federated/__init__.py` + `plans.py` (`make_fbd_plan`, `validate_plan`,
   derived shipping/request/update views) — pure python, fully unit-testable
   without torch; everything else consumes the plan.
2. `federated/server.py` — needs core `Warehouse` + plans; write-permission
   check FIRST (it is the isolation mechanism).
3. `federated/client.py` — needs core engines/`diversification_step`;
   the private per-client view (temp warehouse) before poisoning support.
4. `federated/simulate.py` — needs 1–3; determinism contract before features.
5. Isolation acceptance test (§8 test A) — gate: do not proceed until green.
6. `federated/routing.py` — needs only core + a trained warehouse.
7. `federated/unlearning.py` then `federated/mia.py` (mia needs nothing from
   unlearning; the report combines them).
8. CLI group + `history.json`/report writers.
9. Remaining acceptance tests (§8 B, C).

## 8. Acceptance tests

Write as `tests/test_federated_fbd.py`; CPU, minutes-scale. Use a small
config so bit-identity is fast: the smoke test's `MLPBN`,
`decompose_boundary(model, [["fc1","bn1"], "fc2", "fc3"])` (B=3), `N=3`
colours, `C=3` clients, `r=2` (k=2 colours/client), MNIST subset (e.g. 1 500
train / 512 val), 3 rounds, `local_steps=4`.

```python
# A. THE ISOLATION INVARIANT (the central test — must be bit-identity)
def test_isolation_invariant(tmp_path):
    plan = make_fbd_plan(3, 3, 3, 2, rounds=3, seed=0); validate_plan(plan)
    target = 0                                  # the adversarial client
    runs = {}
    for tag, poison, absent in [
        ("honest", None, frozenset()),
        ("poisoned", PoisonSpec(target, "label_flip"), frozenset()),
        ("absent", None, frozenset({target})),
    ]:
        wh = build_warehouse(tmp_path / tag)    # identical seeds → identical init (assert hashes equal)
        run_federated_simulation(..., seed=11, poison=poison, absent_clients=absent)
        runs[tag] = wh
    safe_colours = [m for m in range(3) if target not in plan.owners_of(m)]
    for m in safe_colours:
        for b in range(3):
            h = [runs[t].content_hash(b, m) for t in ("honest", "poisoned", "absent")]
            assert h[0] == h[1] == h[2]         # bit-identical across all three runs
    poisoned_colours = plan.colours_of(target)
    assert any(runs["honest"].content_hash(b, m) != runs["poisoned"].content_hash(b, m)
               for m in poisoned_colours for b in range(3))   # the attack actually did something

# B. WRITE-PERMISSION ENFORCEMENT
def test_server_rejects_non_owned_write(...):
    # server.receive(round, c, {(b, colour_not_owned_by_c): slice}) raises
    # PermissionError and warehouse hashes are unchanged afterwards.

# C. UNLEARNING: MIA -> 0.5 at <1% utility cost
def test_unlearning_mia_and_utility(...):
    # Train the small federated setup a bit longer (e.g. 6 rounds).
    # acc_before = mean-ensemble accuracy on val.
    # mia_before = mia_auc(member=client0 shard, nonmember=val)["ensemble"]
    # unlearn_aggregate(wh, plan, client_id=0)
    # acc_after, mia_after = ...
    assert abs(mia_after - 0.5) <= 0.05                     # MIA at chance after unlearning
    assert abs(mia_after - 0.5) <= abs(mia_before - 0.5) + 0.02   # never gets worse
    assert acc_before - acc_after < 0.01                    # <1% absolute utility cost
    # plus invariants: untouched-block hashes unchanged; provenance shows
    # "unlearn:client_0" as last writer of every replaced slice.

# D. plan unit tests (no torch): ownership counts, k-rotation schedule,
#    N*r % C != 0 raises, r==1 warns, save/load round-trip equality.
# E. routing smoke: evaluate_routing returns all five strategies with finite
#    acc/auc, and oracle accuracy >= every other strategy's accuracy.
```

**Definition of done:** core smoke test still 5-passed; tests A–E green;
`warehouse-adopt fed make-plan/simulate/route/unlearn/mia --help` all work;
a full small simulation via the CLI produces `history.json`, a routing
report, and an unlearn report matching the schemas above.

## 9. Common pitfalls (this version)

* **Averaging on receive** — the moment the server averages instead of
  replacing, the isolation proof dies. `Warehouse.save_variant` is already
  direct replacement; never wrap it in FedAvg-style aggregation.
* **Shared RNG across clients** — if client b's batches/configs depend on the
  global RNG, removing client a changes b's updates and test A can never be
  bit-identical. Derive every stream from `(seed, round, client)`.
* **Trusting the client for isolation** — the permission check lives in
  `FederatedServer.receive`; a poisoned client may return anything, including
  forged keys. Test B exists because of this.
* **Variant 0 is NOT special here** — colour 0 is a trainable colour like any
  other; the pristine copy for exact unlearning is the separate
  `base_snapshot/` directory, created before round 0.
* **Integer buffers in aggregate unlearning** — `num_batches_tracked` must not
  be float-averaged into a corrupted dtype (the fbd_transfer reference has
  this bug: it `.float()`s everything); copy a donor's integer tensor.
* **BN statistics after replacement** — aggregate-unlearned colours mix BN
  stats; if post-unlearning utility drops more than expected, run the core's
  `recalibrate_bn` on the affected colours before measuring (and say so in
  the report).
* **Oracle routing as an AUC bound** — it bounds accuracy only (core §8 of
  the routing literature pitfall); never assert `oracle_auc >= mean_auc`.
* **Plan/provenance drift** — `find_client_blocks` asserts provenance ⊆ plan;
  if that fails, a write slipped past the permission check — treat as a bug,
  not data to be cleaned.
