# GradMesh: scheduling heterogeneous consumer GPUs for collaborative training

This document states the model GradMesh implements, why each policy is there,
what is claimed, what is not, and how to measure it. It is written to be the
methodology section of a paper and to be checkable against the code.

Implementation: [`engine/coordinator/scheduler.py`](engine/coordinator/scheduler.py).
Assertions: `npm run test:scheduler`.

---

## 1. The problem

Federated and round-synchronised training assume, usually silently, that workers
are interchangeable. Split a dataset into `N` equal shards, train one local epoch
each, average the results, repeat.

On owned, matched hardware that assumption holds. On a mesh of consumer devices
it fails immediately, and it fails in a way that gets worse as the mesh grows.

A synchronised round completes when its slowest worker completes:

```
T_round = max_i (s_i / r_i) + T_aggregate
```

where `s_i` is worker `i`'s shard size and `r_i` its throughput in samples per
second. With an equal split, `s_i = |D| / N`, so:

```
T_round = |D| / (N * min_i r_i) + T_aggregate
```

The round is governed by the **slowest** device, and the fast devices sit idle
for the difference. Three consequences follow.

1. **Adding a machine can make the mesh slower.** Introducing a device with
   `r_new < min_i r_i` reduces every existing shard by a factor of `N/(N+1)` but
   raises the makespan floor to `|D| / ((N+1) * r_new)`. Whenever
   `r_new < min_i r_i * N / (N+1)`, the round gets longer.
2. **Utilisation collapses with heterogeneity.** Mean utilisation is
   `mean_i(r_i) / max_i(r_i)`. A 6x spread between the fastest and slowest
   device wastes most of the mesh's capacity.
3. **The marginal contributor is disincentivised.** The person most likely to
   lend a GPU is the person with the weaker one, and under an equal split their
   contribution actively hurts.

This is the reason ad-hoc GPU sharing does not happen, and it is a scheduling
problem, not a systems-engineering one.

---

## 2. Signals

Every policy is derived from five per-node signals, each normalised to `[0, 1]`
against the current mesh.

| Signal | Symbol | Source |
|---|---|---|
| Compute | `C_i` | Measured FP32 matmul throughput, relative to the mesh peak |
| Memory | `M_i` | Free device memory, relative to the mesh peak |
| Health | `H_i` | Heartbeat freshness, decayed over the liveness window |
| Reliability | `T_i` | Learned from completed versus failed rounds |
| Latency | `L_i` | Control-plane round trip, relative to the mesh worst |

`C_i` is the load-bearing one and the reason it is **measured, not declared**.
Vendor specifications are not comparable across NVIDIA, Intel and CPU backends,
and a device's name says nothing about thermal state, driver version, or what
else is running on it. Each worker runs a short probe at startup
([`engine/probe.py`](engine/probe.py)): a warmed 2048x2048 FP32 matmul timed over
twelve iterations, plus a device-memory copy benchmark. It costs a few hundred
milliseconds and produces one number that is comparable across every backend.

The scheduling score is their weighted combination:

```
G_i = a*C_i + b*M_i + g*H_i + l*T_i - d*L_i
```

with defaults `a=0.45, b=0.20, g=0.15, l=0.15, d=0.05`. The weights are exposed
in the dashboard and persisted, so an ablation is a slider move rather than a
code change.

---

## 3. Policy 1: admission control

A device is measured, not trusted. `admit()` returns one of three tiers with a
written reason.

- **Rejected** when the device physically cannot do the job: offline, no
  supported accelerator, device memory below the floor, or fitness far enough
  below the mesh median that a full shard would dominate makespan.
- **Probation** when it can run but sits well below the mesh median. It still
  receives work, capped at `probation_shard_cap` of the dataset, so its
  gradients still enter aggregation without it setting the pace.
- **Full** otherwise.

Probation exists because rejecting weak devices outright would be both socially
wrong for a volunteer mesh and statistically wrong: a small shard from a weak
device still contributes gradient diversity, and its cost is bounded by the cap.

Every decision carries a human-readable reason, surfaced in the dashboard. A
contributor whose machine is not receiving work can see why.

---

## 4. Policy 2: predictive proportional sharding

Minimise makespan subject to covering the dataset:

```
minimise   max_i (s_i / r_i)
subject to sum_i s_i = |D|,  s_i >= 0
```

The optimum equalises every `s_i / r_i`, giving:

```
s_i = |D| * r_i / sum_j r_j
```

and a predicted makespan of `|D| / sum_j r_j`. Every admitted worker is
predicted to finish at the same instant, so the idle time that an equal split
creates is designed out rather than tolerated.

Three corrections are applied on top.

**Skew cap.** A device 40x slower than the fastest would receive a shard too
small for its gradient to mean anything. Rates are floored at
`max_j(r_j) / max_shard_skew` before proportioning, trading a little makespan
for gradient diversity.

**Probation cap with water-filling.** Capped nodes are clipped to their cap and
the remaining budget is re-proportioned over the uncapped nodes, iterated until
no cap binds. This is standard water-filling and it terminates in at most `N`
passes.

**Integer repair.** Floor, then distribute the deficit by largest fractional
remainder, so shards sum to exactly `|D|` with no image trained twice or missed.

Throughput is the measured EWMA once a worker has finished a round, and the
startup probe before that:

```
r_i(t+1) = alpha * observed + (1 - alpha) * r_i(t),   alpha = 0.35
```

So the first round is a calibrated guess and every round after is grounded in
what the mesh actually did.

**Sharding is randomised per round with a seeded shuffle.** Most YOLO exports
are ordered by capture session, so a contiguous split hands each worker a
class-correlated slice and biases the local gradients before aggregation sees
them. A per-round seed keeps shards unbiased while staying reproducible.

---

## 5. Policy 3: straggler mitigation

Prediction gives the barrier a deadline for free. Each shard carries:

```
soft = max(min_deadline, predicted * 1.6)
hard = max(2 * min_deadline, predicted * 3.0)
```

A supervisor evaluates every in-flight shard on a two-second tick.

- **Inside the soft deadline:** wait. The prediction has slack.
- **Past soft, an idle peer available:** speculate. A duplicate of the shard
  starts on the idle machine; whichever result lands first wins and the other is
  accepted and discarded. This is Dean and Ghemawat's backup-task idea applied
  to a federated round, and it is cheap here because the duplicate only runs
  when a machine would otherwise be idle.
- **Past hard:** drop. The barrier releases without that shard, the worker's
  reliability takes a multiplicative penalty, and its samples return to the next
  round's plan.

A round that has lost more than `max_dropped_fraction` of its samples is aborted
and replanned rather than aggregated. Aggregating over a third of a missing
dataset is not a valid FedAvg step, and producing a number anyway would be the
kind of quiet wrongness that invalidates a result.

Reliability is additive-increase, multiplicative-decrease, bounded to `[0, 1]`:
`+0.06` per clean round, `x0.75` per failure. Recovering trust takes many more
rounds than losing it, which is the correct asymmetry for a volunteer mesh.

---

## 6. Policy 4: contribution-weighted aggregation

Unequal shards make the unweighted mean the wrong estimator. GradMesh uses the
original FedAvg weighting, with a reliability term:

```
w_global = sum_i (n_i * T_i / sum_j (n_j * T_j)) * w_i
```

where `n_i` is the number of samples worker `i` actually trained on. The
reliability factor `T_i` means a machine with a history of dropped rounds cannot
pull the global model as hard as a proven one, which is a cheap partial defence
against an unreliable, though not a malicious, worker.

Tensors whose shapes disagree with the reference, such as a detection head after
a class-count change, are skipped rather than crashing the round, and the
remaining weights are renormalised so a missing tensor does not silently scale
the average down. This mirrors the shape-compatible filter the v3 worker already
used when loading a checkpoint.

---

## 7. What is claimed

`npm run test:scheduler` asserts each of these against the implementation.

1. **Shards equalise predicted finish time.** Spread across a 6x heterogeneous
   mesh stays within 5% of the makespan.
2. **Every sample is assigned exactly once.** Shards sum to `|D|` for arbitrary
   throughput vectors and worker counts.
3. **Adding a weak machine never raises predicted makespan.** This is the direct
   negation of the equal-split failure mode, and it is what makes an open mesh
   safe to join.
4. **Admission is legible.** Every decision carries a reason, and hard physical
   limits reject before relative ones.
5. **Straggler escalation is ordered.** Wait, then speculate only when a backup
   exists, then drop.
6. **Aggregation weights are proportional to contribution**, sum to one, and are
   damped by reliability.
7. **The throughput model converges.** The first observation is taken at face
   value; later ones are smoothed.
8. **Degenerate meshes degrade safely.** Empty, all-offline and zero-sample
   inputs plan nothing rather than dividing by zero.

---

## 8. What is not claimed

- **This is not step-level gradient synchronisation.** Workers train a local
  epoch and the coordinator averages weights. On a single machine with multiple
  GPUs, PyTorch DDP over NCCL is strictly better and GradMesh is the wrong tool.
- **Convergence is not proven.** Weight averaging across non-IID unequal shards
  is FedAvg, which has known convergence caveats. GradMesh reduces the non-IID
  problem with per-round reshuffling but does not eliminate it. Accuracy
  equivalence has to be measured per dataset, not assumed.
- **Speedup is not linear.** Communication, serialisation, checkpoint transfer
  and the barrier all cost real time, and they are all reported per round rather
  than folded away.
- **Workers are not verified.** A malicious worker can return poisoned weights.
  Reliability scoring catches failure, not lying.

---

## 9. Measurement protocol

The dashboard's run form supports the controlled comparison directly.

**Held constant:** dataset, base checkpoint, image size, per-machine batch size,
round count, optimiser and evaluation procedure.

**Varied:** mode only. `solo` plans onto the single strongest machine; `mesh`
plans across every admitted machine.

**Reported per round**, written to `.gradmesh/runs/<id>/summary.json`:

| Metric | Meaning |
|---|---|
| `makespan_seconds` | Slowest shard in the round |
| `fastest_seconds` | Quickest shard in the round |
| `straggler_gap_seconds` | The difference, which is idle capacity. Lower means a better split |
| `aggregation_seconds` | Coordinator-side weighted averaging cost |
| `wall_clock_seconds` | Barrier open to barrier close, including transfer |
| `serial_estimate_seconds` | Sum of observed shard times, the same work done one machine at a time |
| `speedup` | `serial_estimate / wall_clock` |
| `efficiency` | `speedup / workers` |
| `dropped_shards` | Shards lost to deadline or disconnection |

Per shard, the record also carries predicted versus actual seconds. **Prediction
error is the honest measure of whether the scheduler works**: as the EWMA
converges, error should shrink and the straggler gap with it. A mesh whose
straggler gap is not falling across rounds is a mesh whose scheduler is not
learning, and that is visible directly on the run page.

Report speedup alongside the exact hardware, driver versions, image size, batch
size and network. Speedup without that context is not a result.

---

## 10. Relationship to prior work

- **FedAvg** (McMahan et al., 2017) supplies the aggregation estimator. GradMesh
  uses it as specified, including the sample weighting that unequal shards
  require, and adds a reliability damping term.
- **Backup tasks** (Dean and Ghemawat, 2004) supply speculative re-execution.
  GradMesh applies it inside a synchronised federated round rather than a
  MapReduce phase, triggered by a per-shard prediction rather than a global
  progress percentile.
- **Heterogeneity-aware federated learning** (FedProx, and the asynchronous
  federated line of work) attacks the same problem from the optimisation side,
  by changing the local objective or relaxing the barrier. GradMesh keeps the
  barrier and the objective and attacks the **work assignment** instead, which
  is complementary rather than competing: proportional sharding and a proximal
  term could be used together.
- **Volunteer computing** (BOINC and descendants) solved contribution and trust
  at scale for embarrassingly parallel work. Synchronised training is not
  embarrassingly parallel, which is exactly why the straggler policy has to be
  predictive rather than purely reactive.

The contribution is the combination: measured capability as a scheduling
primitive, makespan-optimal proportional sharding, prediction-derived deadlines
driving speculation, and contribution-weighted aggregation, all running per
round on hardware nobody controls.

---

## 11. Open questions

1. **Optimal skew.** The cap trades makespan against gradient diversity. The
   right value is almost certainly dataset-dependent and is currently a constant.
2. **Compression.** Weight transfer dominates the round for larger models.
   Gradient or delta compression would move the ceiling on model size.
3. **Partial-round credit.** A worker dropped at the hard deadline loses all of
   its work. Checkpointing mid-epoch would recover it.
4. **Verification.** Redundant assignment plus cross-checking would let a mesh
   run across a network you do not control, which is the step from a lab tool to
   a marketplace.
5. **Cross-subnet meshes.** NAT traversal and transport security are out of
   scope for a LAN prototype and are the next real systems problem.
