"""GradMesh heterogeneity-aware scheduler.

The v3 coordinator split a dataset into equal shards and then waited for the
slowest machine. On a mesh of consumer hardware that is the wrong policy: a
round costs max_i(t_i), so an equal split makes the weakest GPU set the pace for
everyone and the mesh gets *slower* as you add machines.

This module replaces that with four cooperating policies.

1. Admission control. A device is measured, not trusted. admit() decides whether
   a node may hold a shard at all, and at which tier.
2. Predictive proportional sharding. plan_round() sizes each shard so every
   admitted worker is predicted to finish at the same instant, which minimises
   round makespan instead of round variance.
3. Straggler mitigation. deadline_for() derives a soft and hard deadline from
   the same prediction. A soft miss triggers speculative re-execution on an idle
   fast worker; a hard miss drops the shard from the barrier and returns its
   samples to the next round's plan.
4. Contribution-weighted aggregation. aggregation_weights() weights each
   worker's state dict by the samples it actually trained on, damped by that
   worker's reliability, which is the correct FedAvg estimator for unequal
   shards.

Everything here is a pure function over plain dicts so the policy can be unit
tested and reported in the paper independently of FastAPI.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------


@dataclass
class MeshPolicy:
    """Tunable scheduling policy. Surfaced in the dashboard settings page."""

    # Fitness weights: G_i = a*C_i + b*M_i + g*H_i + l*T_i - d*L_i
    w_compute: float = 0.45
    w_memory: float = 0.20
    w_health: float = 0.15
    w_reliability: float = 0.15
    w_latency: float = 0.05

    # Admission gate.
    min_gflops: float = 25.0            # below this a device cannot hold a full shard
    min_memory_mb: int = 1800           # below this YOLO at imgsz 512 will not fit
    min_fitness: float = 0.08           # relative floor against the mesh median
    probation_fitness: float = 0.22     # admitted, but capped to a micro shard
    probation_shard_cap: float = 0.15   # max fraction of the dataset on probation

    # Sharding.
    min_shard_samples: int = 8          # smaller shards cost more to ship than to run
    max_shard_skew: float = 6.0         # cap fastest:slowest ratio for gradient diversity

    # Straggler mitigation.
    soft_deadline_factor: float = 1.6   # predicted * this -> speculative backup
    hard_deadline_factor: float = 3.0   # predicted * this -> drop from the barrier
    min_deadline_seconds: float = 45.0  # never chase a worker that just started
    cold_start_grace_seconds: float = 240.0  # a worker's first round pays setup costs
    max_dropped_fraction: float = 0.34  # abort the round if more than this is lost

    # Learning.
    throughput_ewma_alpha: float = 0.35
    reliability_reward: float = 0.06
    reliability_penalty: float = 0.25

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "MeshPolicy":
        if not data:
            return cls()
        defaults = cls()
        known = {name: data[name] for name in vars(defaults) if name in data}
        return cls(**known)


DEFAULT_POLICY = MeshPolicy()


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_signal(node: dict, peak_gflops: float) -> float:
    """C_i: measured throughput relative to the strongest device in the mesh."""
    gflops = float((node.get("capability") or {}).get("gflops") or 0.0)
    return _clamp(_safe_div(gflops, peak_gflops))


def memory_signal(node: dict, peak_memory_mb: float) -> float:
    """M_i: free device memory relative to the largest device in the mesh."""
    total = float(node.get("gpu_memory_mb") or 0)
    allocated = float(node.get("allocated_memory_mb") or 0)
    free = max(0.0, total - allocated)
    return _clamp(_safe_div(free, peak_memory_mb))


def health_signal(node: dict, now: Optional[float] = None, timeout: float = 20.0) -> float:
    """H_i: heartbeat freshness decayed over the liveness window."""
    now = now if now is not None else time.time()
    age = now - float(node.get("last_seen") or 0.0)
    if age <= 0:
        return 1.0
    return _clamp(1.0 - _safe_div(age, timeout))


def reliability_signal(node: dict) -> float:
    """T_i: learned trust from completed vs failed or abandoned rounds."""
    return _clamp(float(node.get("reliability", 0.7)))


def latency_signal(node: dict, worst_latency_ms: float) -> float:
    """L_i: normalised control-plane round trip. Higher is worse."""
    latency = float(node.get("latency_ms") or 0.0)
    return _clamp(_safe_div(latency, worst_latency_ms))


def fitness(node: dict, mesh: dict, policy: MeshPolicy = DEFAULT_POLICY) -> float:
    """G_i, the scheduling score from the GradMesh model."""
    score = (
        policy.w_compute * compute_signal(node, mesh["peak_gflops"])
        + policy.w_memory * memory_signal(node, mesh["peak_memory_mb"])
        + policy.w_health * health_signal(node, mesh["now"])
        + policy.w_reliability * reliability_signal(node)
        - policy.w_latency * latency_signal(node, mesh["worst_latency_ms"])
    )
    return max(0.0, score)


def mesh_reference(nodes: Sequence[dict], now: Optional[float] = None) -> dict:
    """Normalisation reference. Floors keep a single-node mesh well defined."""
    now = now if now is not None else time.time()
    gflops = [float((n.get("capability") or {}).get("gflops") or 0.0) for n in nodes]
    memory = [float(n.get("gpu_memory_mb") or 0.0) for n in nodes]
    latency = [float(n.get("latency_ms") or 0.0) for n in nodes]
    return {
        "now": now,
        "peak_gflops": max(gflops) if gflops and max(gflops) > 0 else 1.0,
        "peak_memory_mb": max(memory) if memory and max(memory) > 0 else 1.0,
        "worst_latency_ms": max(latency) if latency and max(latency) > 0 else 1.0,
    }


# --------------------------------------------------------------------------
# Admission control
# --------------------------------------------------------------------------

TIER_FULL = "full"
TIER_PROBATION = "probation"
TIER_REJECTED = "rejected"

# Partitioning strategies. PROPORTIONAL is the contribution of this work;
# EQUAL is the naive baseline it has to beat, and the benchmark harness runs
# both over identical hardware so the comparison is not merely asserted.
PARTITION_PROPORTIONAL = "proportional"
PARTITION_EQUAL = "equal"


@dataclass
class Admission:
    node_id: str
    tier: str
    fitness: float
    reason: str
    shard_cap: float = 1.0

    def as_dict(self) -> dict:
        return asdict(self)


def admit(
    nodes: Sequence[dict],
    policy: MeshPolicy = DEFAULT_POLICY,
    now: Optional[float] = None,
) -> Dict[str, Admission]:
    """Decide, per node, whether it may hold a shard and how large a one.

    This is the answer to "should the coordinator trust this GPU". A device is
    rejected when it physically cannot run the job (no accelerator, not enough
    device memory, no measured throughput) and put on probation when it can run
    but sits far enough below the mesh median that a full shard would make it
    the straggler. Probation nodes still contribute gradients, they just carry
    less of the dataset.
    """
    mesh = mesh_reference(nodes, now)
    scores = {n["node_id"]: fitness(n, mesh, policy) for n in nodes}
    ranked = sorted(scores.values(), reverse=True)
    median = ranked[len(ranked) // 2] if ranked else 0.0

    decisions: Dict[str, Admission] = {}
    for node in nodes:
        node_id = node["node_id"]
        score = scores[node_id]
        capability = node.get("capability") or {}
        gflops = float(capability.get("gflops") or 0.0)
        memory_mb = int(node.get("gpu_memory_mb") or 0)

        if not node.get("active"):
            decisions[node_id] = Admission(node_id, TIER_REJECTED, score, "node is offline")
            continue
        if not node.get("supports_training", True):
            decisions[node_id] = Admission(
                node_id, TIER_REJECTED, score, "no supported accelerator was detected"
            )
            continue
        if memory_mb < policy.min_memory_mb:
            decisions[node_id] = Admission(
                node_id,
                TIER_REJECTED,
                score,
                "%d MB of device memory is below the %d MB floor" % (memory_mb, policy.min_memory_mb),
            )
            continue
        if gflops and gflops < policy.min_gflops:
            decisions[node_id] = Admission(
                node_id,
                TIER_PROBATION,
                score,
                "measured %.0f GFLOP/s is below the %.0f GFLOP/s floor" % (gflops, policy.min_gflops),
                shard_cap=policy.probation_shard_cap,
            )
            continue
        if median > 0 and score < policy.min_fitness * median:
            decisions[node_id] = Admission(
                node_id,
                TIER_REJECTED,
                score,
                "fitness is far below the mesh median and would dominate makespan",
            )
            continue
        if median > 0 and score < policy.probation_fitness * median:
            decisions[node_id] = Admission(
                node_id,
                TIER_PROBATION,
                score,
                "fitness is below the probation threshold, so the shard is capped",
                shard_cap=policy.probation_shard_cap,
            )
            continue

        decisions[node_id] = Admission(node_id, TIER_FULL, score, "admitted")
    return decisions


# --------------------------------------------------------------------------
# Predictive proportional sharding
# --------------------------------------------------------------------------


@dataclass
class ShardAssignment:
    node_id: str
    shard_index: int
    samples: int
    fitness: float
    tier: str
    throughput_sps: float
    predicted_seconds: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class RoundPlan:
    total_samples: int
    assignments: List[ShardAssignment] = field(default_factory=list)
    rejected: List[dict] = field(default_factory=list)
    predicted_makespan_seconds: float = 0.0
    predicted_serial_seconds: float = 0.0
    strategy: str = PARTITION_PROPORTIONAL

    @property
    def predicted_speedup(self) -> float:
        return _safe_div(self.predicted_serial_seconds, self.predicted_makespan_seconds)

    @property
    def predicted_imbalance(self) -> float:
        """Coefficient of variation of predicted finish times.

        Zero means every worker is expected to finish at the same instant, which
        is the goal. This is the headline number for the partitioning ablation,
        and it is predicted rather than observed, so it can be compared against
        the measured spread after the round to show the model was right.
        """
        times = [a.predicted_seconds for a in self.assignments]
        if len(times) < 2:
            return 0.0
        mean = sum(times) / len(times)
        if mean <= 0:
            return 0.0
        variance = sum((value - mean) ** 2 for value in times) / len(times)
        return (variance ** 0.5) / mean

    def as_dict(self) -> dict:
        return {
            "total_samples": self.total_samples,
            "assignments": [a.as_dict() for a in self.assignments],
            "rejected": self.rejected,
            "strategy": self.strategy,
            "predicted_makespan_seconds": round(self.predicted_makespan_seconds, 2),
            "predicted_serial_seconds": round(self.predicted_serial_seconds, 2),
            "predicted_speedup": round(self.predicted_speedup, 3),
            "predicted_imbalance": round(self.predicted_imbalance, 4),
        }


def probe_throughput(node: dict) -> float:
    """Samples per second implied by the startup probe alone.

    A YOLOv8n step at imgsz 640 is roughly 9 GFLOP of useful work per image once
    backward and optimiser traffic are included, and real utilisation on
    consumer parts sits near 12 percent of peak FP32 matmul. The absolute
    constant matters less than consistency, because sharding uses ratios.
    """
    gflops = float((node.get("capability") or {}).get("gflops") or 0.0)
    return max(0.05, gflops * 0.12 / 9.0)


def calibration_factor(nodes: Sequence[dict]) -> float:
    """Scale probe estimates onto the same scale as measured throughput.

    The probe estimates steady-state compute. A measured round also contains
    everything Ultralytics does once per call: the AMP check, dataloader
    construction, model setup and final validation. On a small shard those fixed
    costs dominate, so a machine that has finished a round reports far fewer
    samples per second than its probe implies. Measured on this hardware the gap
    was 2.6 against 37, a factor of fourteen.

    Mixing the two scales in one plan is what does the damage: a node that had
    never run looked fourteen times faster than an identical node that had, so
    the planner handed it nearly the whole dataset and dropped the other for
    falling under the minimum shard size.

    So probe values are multiplied by the ratio this mesh actually achieves,
    taken as the median over nodes that have both numbers. With no measurements
    yet the factor is 1.0 and every node is on the probe scale together, which is
    equally consistent. This corrects the scale, not the shape: relative
    differences between devices still come from the probe.
    """
    ratios = [
        float(node.get("throughput_sps") or 0.0) / probe_throughput(node)
        for node in nodes
        if float(node.get("throughput_sps") or 0.0) > 0
    ]
    if not ratios:
        return 1.0
    ratios.sort()
    middle = len(ratios) // 2
    median = ratios[middle] if len(ratios) % 2 else (ratios[middle - 1] + ratios[middle]) / 2.0
    # Clamp so one pathological round cannot distort every future plan.
    return max(0.01, min(100.0, median))


def effective_throughput(
    node: dict,
    policy: MeshPolicy = DEFAULT_POLICY,
    calibration: float = 1.0,
) -> float:
    """Samples per second for this node, on a single consistent scale.

    Measured throughput wins once the node has finished a round. Before that the
    probe estimate is used, scaled by `calibration` so it is comparable with the
    measured values of its peers.
    """
    measured = float(node.get("throughput_sps") or 0.0)
    if measured > 0:
        return measured
    return max(0.01, probe_throughput(node) * calibration)


def _proportional_with_caps(rates: Dict[str, float], caps: Dict[str, float]) -> Dict[str, float]:
    """Water-filling: proportional shares, clipped by per-node caps, iterated."""
    remaining = dict(rates)
    shares: Dict[str, float] = {}
    budget = 1.0

    for _ in range(len(rates) + 1):
        if not remaining or budget <= 1e-9:
            break
        total_rate = sum(remaining.values())
        if total_rate <= 0:
            break
        clipped = False
        for node_id, rate in list(remaining.items()):
            proposed = budget * rate / total_rate
            cap = caps.get(node_id, 1.0)
            if proposed > cap:
                shares[node_id] = cap
                budget -= cap
                remaining.pop(node_id)
                clipped = True
        if not clipped:
            for node_id, rate in remaining.items():
                shares[node_id] = budget * rate / total_rate
            remaining.clear()
            budget = 0.0

    # Anything left over, meaning every node was capped, goes back proportionally.
    if budget > 1e-9 and shares:
        total = sum(shares.values()) or 1.0
        for node_id in list(shares):
            shares[node_id] += budget * shares[node_id] / total

    return shares


def plan_round(
    nodes: Sequence[dict],
    total_samples: int,
    policy: MeshPolicy = DEFAULT_POLICY,
    now: Optional[float] = None,
    strategy: str = PARTITION_PROPORTIONAL,
) -> RoundPlan:
    """Size every shard so all admitted workers finish together.

    With throughput r_i and shard size s_i, worker time is s_i / r_i. Makespan is
    minimised when every s_i / r_i is equal, which gives s_i proportional to
    r_i. We then apply the probation cap and the skew cap, redistribute any
    trimmed samples over the remaining headroom, and finally repair rounding so
    the shards sum back to exactly total_samples.

    Passing strategy=PARTITION_EQUAL gives every admitted worker the same number
    of samples instead. That is deliberately the wrong policy for heterogeneous
    hardware, and it exists so the benchmark harness can measure how wrong: it
    is the control arm for the load-balancing ablation. Admission, deadlines and
    aggregation are identical in both arms, so the only variable is shard size.
    """
    if total_samples <= 0:
        return RoundPlan(total_samples=0)

    decisions = admit(nodes, policy, now)
    by_id = {n["node_id"]: n for n in nodes}

    eligible = [
        by_id[node_id]
        for node_id, decision in decisions.items()
        if decision.tier in (TIER_FULL, TIER_PROBATION)
    ]
    rejected = [
        {"node_id": d.node_id, "reason": d.reason, "fitness": round(d.fitness, 4)}
        for d in decisions.values()
        if d.tier == TIER_REJECTED
    ]

    if not eligible:
        return RoundPlan(total_samples=total_samples, rejected=rejected, strategy=strategy)

    # One scale for every node, whether measured or merely probed.
    calibration = calibration_factor(eligible)
    rates = {n["node_id"]: effective_throughput(n, policy, calibration) for n in eligible}

    if strategy == PARTITION_EQUAL:
        # The control arm. Every admitted worker carries the same share, which
        # is what a scheduler that ignores capability does. Rates are still
        # measured, because the predicted finish times are what expose the
        # imbalance this arm is meant to demonstrate.
        share = 1.0 / len(rates)
        shares = {node_id: share for node_id in rates}
    else:
        # Skew cap: a device that is 40x slower still needs enough samples for
        # its gradient to mean something, so clamp the ratio before
        # proportioning.
        fastest = max(rates.values())
        floor_rate = fastest / policy.max_shard_skew
        rates = {node_id: max(rate, floor_rate) for node_id, rate in rates.items()}

        caps = {
            node_id: (
                policy.probation_shard_cap
                if decisions[node_id].tier == TIER_PROBATION
                else 1.0
            )
            for node_id in rates
        }

        shares = _proportional_with_caps(rates, caps)

    raw = {node_id: share * total_samples for node_id, share in shares.items()}
    samples = {node_id: int(math.floor(value)) for node_id, value in raw.items()}

    # Drop shards too small to be worth shipping, then re-proportion their samples.
    too_small = [
        node_id
        for node_id, count in samples.items()
        if count < policy.min_shard_samples
    ]
    for node_id in too_small:
        if len(samples) <= 1:
            break
        rejected.append(
            {
                "node_id": node_id,
                "reason": "shard of %d samples is below the %d sample floor"
                % (samples[node_id], policy.min_shard_samples),
                "fitness": round(decisions[node_id].fitness, 4),
            }
        )
        samples.pop(node_id)
        rates.pop(node_id, None)

    if not samples:
        return RoundPlan(total_samples=total_samples, rejected=rejected, strategy=strategy)

    # Largest-remainder repair so the shards sum to exactly total_samples.
    deficit = total_samples - sum(samples.values())
    if deficit:
        order = sorted(samples, key=lambda nid: raw.get(nid, 0.0) - samples[nid], reverse=True)
        step = 1 if deficit > 0 else -1
        index = 0
        guard = 0
        while deficit != 0 and order and guard < 10 * len(order) + 64:
            guard += 1
            node_id = order[index % len(order)]
            index += 1
            if step < 0 and samples[node_id] <= 1:
                continue
            samples[node_id] += step
            deficit -= step

    assignments: List[ShardAssignment] = []
    for shard_index, node_id in enumerate(sorted(samples, key=lambda nid: -samples[nid])):
        rate = rates[node_id]
        count = samples[node_id]
        assignments.append(
            ShardAssignment(
                node_id=node_id,
                shard_index=shard_index,
                samples=count,
                fitness=round(decisions[node_id].fitness, 4),
                tier=decisions[node_id].tier,
                throughput_sps=round(rate, 4),
                predicted_seconds=round(_safe_div(count, rate), 2),
            )
        )

    makespan = max((a.predicted_seconds for a in assignments), default=0.0)
    best_rate = max(rates.values()) if rates else 0.0
    serial = _safe_div(total_samples, best_rate)

    return RoundPlan(
        total_samples=total_samples,
        assignments=assignments,
        rejected=rejected,
        predicted_makespan_seconds=makespan,
        predicted_serial_seconds=serial,
        strategy=strategy,
    )


# --------------------------------------------------------------------------
# Straggler mitigation
# --------------------------------------------------------------------------


@dataclass
class Deadline:
    soft_seconds: float
    hard_seconds: float

    def as_dict(self) -> dict:
        return asdict(self)


def deadline_for(
    predicted_seconds: float,
    policy: MeshPolicy = DEFAULT_POLICY,
    cold: bool = False,
) -> Deadline:
    """Soft and hard deadlines derived from the prediction that sized the shard.

    A worker's first round is a different workload from every round after it. It
    pays for the checkpoint transfer, CUDA context creation, kernel autotuning
    and the dataset label cache, none of which the throughput model can see,
    because the startup probe measures steady-state matmul rather than any of
    that. Measured on an RTX 3050 the first round ran 4x its own second round.
    Deriving a deadline from the probe alone would therefore declare a perfectly
    healthy machine a straggler on the one round where it is most likely to be a
    new contributor watching to see whether this works.

    So a node with no measured history gets a flat grace period instead, and
    ordinary deadlines take over from its second round onward.
    """
    predicted = max(float(predicted_seconds), 1.0)
    soft = max(policy.min_deadline_seconds, predicted * policy.soft_deadline_factor)
    hard = max(policy.min_deadline_seconds * 2, predicted * policy.hard_deadline_factor)
    if cold:
        soft = max(soft, policy.cold_start_grace_seconds)
        hard = max(hard, policy.cold_start_grace_seconds * 2)
    return Deadline(soft_seconds=soft, hard_seconds=hard)


def straggler_action(elapsed_seconds: float, deadline: Deadline, has_idle_backup: bool) -> str:
    """One of wait, speculate, or drop.

    A soft miss means the worker is running slower than its own measured
    history, so we launch a duplicate of its shard on an idle fast node and keep
    whichever result lands first. A hard miss means the worker is effectively
    gone, so the barrier releases without it.
    """
    if elapsed_seconds >= deadline.hard_seconds:
        return "drop"
    if elapsed_seconds >= deadline.soft_seconds and has_idle_backup:
        return "speculate"
    return "wait"


def should_abort_round(
    dropped_samples: int,
    total_samples: int,
    policy: MeshPolicy = DEFAULT_POLICY,
) -> bool:
    """Aggregating over a third of a missing dataset is not a valid FedAvg step."""
    if total_samples <= 0:
        return False
    return _safe_div(dropped_samples, total_samples) > policy.max_dropped_fraction


# --------------------------------------------------------------------------
# Learning and aggregation
# --------------------------------------------------------------------------


def update_throughput(
    node: dict, samples: int, seconds: float, policy: MeshPolicy = DEFAULT_POLICY
) -> float:
    """EWMA of observed samples per second. Returns the new estimate."""
    if seconds <= 0 or samples <= 0:
        return float(node.get("throughput_sps") or 0.0)
    observed = samples / seconds
    previous = float(node.get("throughput_sps") or 0.0)
    if previous <= 0:
        return observed
    alpha = policy.throughput_ewma_alpha
    return alpha * observed + (1.0 - alpha) * previous


def update_reliability(node: dict, succeeded: bool, policy: MeshPolicy = DEFAULT_POLICY) -> float:
    """Additive-increase, multiplicative-decrease trust, bounded to [0, 1]."""
    current = float(node.get("reliability", 0.7))
    if succeeded:
        return _clamp(current + policy.reliability_reward)
    return _clamp(current * (1.0 - policy.reliability_penalty))


def aggregation_weights(results: Sequence[dict]) -> Dict[str, float]:
    """FedAvg weights for unequal shards.

    Each result contributes in proportion to the samples it actually trained on,
    scaled by the contributing node's reliability so a flaky worker cannot pull
    the global model as hard as a proven one. Weights sum to 1.
    """
    scored: Dict[str, float] = {}
    for result in results:
        samples = max(0, int(result.get("samples") or 0))
        reliability = _clamp(float(result.get("reliability", 1.0)), 0.1, 1.0)
        if samples:
            scored[result["batch_id"]] = samples * reliability

    total = sum(scored.values())
    if not total:
        # Degenerate case: no sample counts reported. Fall back to a plain mean.
        count = len(results) or 1
        return {result["batch_id"]: 1.0 / count for result in results}
    return {batch_id: value / total for batch_id, value in scored.items()}


# Rough activation cost of one YOLOv8n image at 640px, in MB, measured rather
# than derived: batch 8 at 640 sits near 3.2 GB on an RTX 3050. Activation
# memory scales with pixel count, so it is quadratic in image size.
ACTIVATION_MB_PER_IMAGE_AT_640 = 340.0
# Weights, gradients, optimiser state and the CUDA context, none of which
# depend on batch size.
FIXED_OVERHEAD_MB = 900.0
# Never fill a card completely: fragmentation and the display output need room.
USABLE_MEMORY_FRACTION = 0.75


def safe_batch_size(
    device_memory_mb: int,
    imgsz: int,
    requested: int,
    node_max: Optional[int] = None,
) -> int:
    """Largest batch this device can hold, never above what was asked for.

    This is the guardrail that matters most for a volunteer mesh. The person
    starting the run picks one batch size for everyone, but the machines are not
    the same: a batch of 8 at 640px is comfortable on a 12 GB card and an
    instant out-of-memory crash on a 4 GB laptop. Without this, one contributor
    with a smaller GPU fails every round, loses reliability for it, and
    eventually gets dropped from a mesh they were perfectly able to help with.

    Returns at least 1. A device that cannot hold even one image is caught
    earlier by the admission gate's memory floor, not here.
    """
    budget = max(0.0, device_memory_mb * USABLE_MEMORY_FRACTION - FIXED_OVERHEAD_MB)
    per_image = ACTIVATION_MB_PER_IMAGE_AT_640 * (max(32, imgsz) / 640.0) ** 2
    affordable = int(budget // per_image) if per_image > 0 else requested
    ceiling = max(1, min(requested, affordable if affordable > 0 else 1))
    if node_max:
        ceiling = min(ceiling, node_max)
    return max(1, ceiling)


def imbalance(seconds: Sequence[float]) -> float:
    """Coefficient of variation of observed shard times.

    The measured counterpart to RoundPlan.predicted_imbalance, and the metric
    the load-balancing ablation reports. Dimensionless, so it is comparable
    across dataset sizes and hardware, which raw seconds are not.
    """
    values = [float(value) for value in seconds if value and value > 0]
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean <= 0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return (variance ** 0.5) / mean


def efficiency(speedup: float, node_count: int) -> float:
    """Parallel efficiency, the headline number for the benchmark table."""
    return _safe_div(speedup, node_count)
