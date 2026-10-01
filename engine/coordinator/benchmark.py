"""The experiment harness.

This is the part that turns GradMesh from a thing that trains into a thing you
can publish. It expands a factorial design over node count, dataset size and
partitioning strategy, runs every cell the required number of times, and writes
results that survive contact with a reviewer.

Design decisions that matter, because each one is a question somebody will ask.

**Node subsets are chosen deterministically.** Going from four machines to two
means picking which two, and picking differently between repeats would confound
the node-count effect with a hardware effect. The default takes the strongest K
by measured fitness, so the 2-node cell is always the same two machines. A
random policy is available for deliberately sampling the space, and it is seeded
by repeat index so it still reproduces.

**Dataset sizes are subsets of one parent, never different datasets.** Held-out
validation images are identical across every size, so a change in accuracy is a
change in what was learned.

**Trials run strictly one at a time.** Two concurrent trials would share the
same GPUs and neither timing would mean anything.

**Everything is written to disk after every trial.** A full sweep is a day of
compute. It must survive a crash, a reboot and a closed laptop lid.

The matrix expansion and summarisation live here as pure functions so they can
be tested without a mesh. Execution lives in app.py, which owns the run loop.
"""

from __future__ import annotations

import csv
import itertools
import json
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from coordinator.scheduler import PARTITION_EQUAL, PARTITION_PROPORTIONAL

SELECTION_STRONGEST = "strongest"
SELECTION_RANDOM = "random"

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
STATUS_ABORTED = "aborted"


# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------


@dataclass
class SuiteConfig:
    """Everything a sweep needs, and nothing that changes while it runs."""

    name: str = "scaling-sweep"
    base_model: str = "yolov8n.pt"
    parent_dataset_id: Optional[str] = None
    dataset_sizes: List[int] = field(default_factory=lambda: [100, 1000])
    node_counts: List[int] = field(default_factory=list)  # empty means 1..available
    strategies: List[str] = field(default_factory=lambda: [PARTITION_PROPORTIONAL])
    repeats: int = 3
    # The single-machine cells are the denominator of every speedup in the
    # sweep, so their variance contaminates every number in the table. Leg 1
    # ran them three times and got 302.9 +/- 63.8 s, a 21% coefficient of
    # variation, which put the headline speedup anywhere between 1.26x and
    # 1.93x depending on which repeat you divided by. They get more repeats
    # than the rest of the design for that reason alone.
    baseline_repeats: int = 5
    # Two rounds of every trial are spent teaching the throughput estimator
    # what these machines can do, and the shard plan is poor until it knows.
    # At five rounds that warmup was 40% of the trial and the whole tax fell
    # on the proportional arm, which is the arm being judged.
    rounds: int = 10
    # Ultralytics defaults warmup_epochs to 3.0. A federated round trains for
    # one epoch, so with the default every round of every trial runs entirely
    # inside learning-rate warmup and never reaches the stable phase, with the
    # bias group held at warmup_bias_lr = 0.1 throughout. Rounds after the
    # first continue from aggregated weights and are not fresh training runs,
    # so they should not warm up at all. None restores the Ultralytics default.
    warmup_epochs: Optional[float] = 0.0
    imgsz: int = 640
    batch_size: int = 8
    node_selection: str = SELECTION_STRONGEST
    evaluate: bool = True
    # Free-text label for the network these trials ran on, so results gathered
    # on lab Ethernet and on a phone hotspot stay distinguishable in the data.
    network_label: str = "unspecified"
    notes: str = ""
    # Legs of one campaign share a campaign_id and an otherwise identical
    # design, differing only in network_label. That is what makes "the same
    # experiment on college Wi-Fi and on a phone hotspot" a controlled
    # comparison rather than two unrelated sweeps.
    campaign_id: Optional[str] = None
    leg: int = 1
    # Stop a trial that has clearly hung rather than losing the rest of the day.
    trial_timeout_seconds: int = 3600
    settle_seconds: float = 6.0

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "SuiteConfig":
        if not data:
            return cls()
        defaults = cls()
        known = {name: data[name] for name in vars(defaults) if name in data}
        return cls(**known)


@dataclass
class TrialSpec:
    trial_id: str
    index: int
    node_count: int
    sample_count: int
    strategy: str
    repeat: int
    # Filled in at execution time, because which machines are online can change
    # between planning the sweep and reaching a given cell.
    node_ids: List[str] = field(default_factory=list)
    dataset_id: Optional[str] = None
    # What the subset actually contained. A request for 1000 images against a
    # dataset holding 4 silently trains on 4, and leg 1 published a table whose
    # 100-image and 1000-image rows were the same four images. The requested
    # size stays in sample_count; this is what ran.
    effective_sample_count: Optional[int] = None
    # Distinct per repeat so the repeats of a cell actually differ.
    seed: int = 0

    def label(self) -> str:
        arm = "" if self.strategy == PARTITION_PROPORTIONAL else " equal-split"
        return "%d node%s, %d images%s, run %d" % (
            self.node_count,
            "" if self.node_count == 1 else "s",
            self.sample_count,
            arm,
            self.repeat + 1,
        )

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["label"] = self.label()
        return payload


def expand(config: SuiteConfig, available_nodes: int) -> List[TrialSpec]:
    """Expand the factorial design into an ordered list of trials.

    Order is deliberate on two counts.

    Node count varies slowest, then dataset size. A sweep abandoned halfway
    therefore still contains complete cells for the smaller node counts rather
    than one repeat of everything, and a partial result you can plot beats a
    complete result you never finished.

    Within a cell, strategy varies fastest and repeat varies second-fastest, so
    the partitioning arms alternate rather than running in blocks. Leg 1 ran
    all three proportional trials and then all three equal trials, and one
    machine happened to run 51% faster during the second block than the first.
    That drift landed entirely on the strategy variable and made the ablation
    uninterpretable. Alternating costs nothing and spreads any drift across
    both arms instead of into the contrast between them.
    """
    counts = [n for n in (config.node_counts or range(1, available_nodes + 1)) if 1 <= n <= available_nodes]
    counts = sorted(set(counts))
    sizes = sorted(set(size for size in config.dataset_sizes if size > 0))
    strategies = [s for s in config.strategies if s in {PARTITION_PROPORTIONAL, PARTITION_EQUAL}]
    if not strategies:
        strategies = [PARTITION_PROPORTIONAL]

    repeats = max(1, config.repeats)
    baseline_repeats = max(repeats, config.baseline_repeats)

    trials: List[TrialSpec] = []
    index = 0
    for node_count, sample_count, repeat, strategy in itertools.product(
        counts, sizes, range(max(repeats, baseline_repeats)), strategies
    ):
        # Equal and proportional partitioning are identical on one machine, so
        # running both would waste a slot and put a duplicate in the ablation.
        if node_count == 1 and strategy == PARTITION_EQUAL:
            continue
        # Only the baseline cells get the extra repeats.
        limit = baseline_repeats if node_count == 1 else repeats
        if repeat >= limit:
            continue
        trials.append(
            TrialSpec(
                trial_id=uuid.uuid4().hex[:10],
                index=index,
                node_count=node_count,
                sample_count=sample_count,
                strategy=strategy,
                repeat=repeat,
                seed=1000 + repeat * 17,
            )
        )
        index += 1
    return trials


def select_nodes(
    nodes: Sequence[dict],
    count: int,
    selection: str,
    seed: int,
) -> List[str]:
    """Choose which machines take part in a trial.

    `nodes` must already be sorted strongest first by the caller, which owns the
    fitness function.
    """
    if count >= len(nodes):
        return [node["node_id"] for node in nodes]
    if selection == SELECTION_RANDOM:
        import random

        pool = [node["node_id"] for node in nodes]
        return sorted(random.Random(seed).sample(pool, count))
    return [node["node_id"] for node in nodes[:count]]


def estimate_seconds(trials: Sequence[TrialSpec], config: SuiteConfig, throughput_sps: float) -> float:
    """Rough wall-clock estimate for the whole sweep.

    Deliberately crude and deliberately pessimistic. Its only job is to tell
    somebody whether they are starting a twenty-minute job or an overnight one
    before they walk away from the machine.
    """
    if throughput_sps <= 0:
        throughput_sps = 8.0
    total = 0.0
    for trial in trials:
        per_round = trial.sample_count / max(1.0, throughput_sps * trial.node_count)
        # Per-round overhead: shard build, transfer, aggregation, evaluation.
        overhead = 6.0 + (12.0 if config.evaluate else 0.0)
        total += config.rounds * (per_round + overhead) + config.settle_seconds
    return total


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def trial_result(spec: TrialSpec, run: dict, status: str, error: Optional[str] = None) -> dict:
    """Flatten a finished run into one row of the results table."""
    history = run.get("round_history") or []
    accuracy = run.get("accuracy_history") or []
    final = accuracy[-1] if accuracy else {}

    imbalances = [item.get("imbalance") for item in history if item.get("imbalance") is not None]
    comm_bytes = sum(int(item.get("comm_bytes") or 0) for item in history)
    comm_seconds = sum(float(item.get("comm_seconds") or 0.0) for item in history)
    train_seconds = sum(float(item.get("wall_clock_seconds") or 0.0) for item in history)
    serial_seconds = sum(float(item.get("serial_estimate_seconds") or 0.0) for item in history)

    return {
        "trial_id": spec.trial_id,
        "index": spec.index,
        "label": spec.label(),
        "status": status,
        "error": error,
        "run_id": run.get("id"),
        "node_count": spec.node_count,
        "node_ids": spec.node_ids,
        "sample_count": spec.sample_count,
        "effective_sample_count": spec.effective_sample_count,
        "strategy": spec.strategy,
        "repeat": spec.repeat,
        "rounds_completed": run.get("current_round", 0),
        "rounds_requested": run.get("rounds", 0),
        # Timing. train_seconds excludes evaluation by construction.
        "train_seconds": round(train_seconds, 2),
        "serial_estimate_seconds": round(serial_seconds, 2),
        "eval_seconds": round(float(run.get("eval_seconds_total") or 0.0), 2),
        "speedup": round(serial_seconds / train_seconds, 4) if train_seconds > 0 else None,
        "efficiency": round(serial_seconds / train_seconds / spec.node_count, 4)
        if train_seconds > 0 and spec.node_count
        else None,
        # Accuracy.
        "map50": final.get("map50"),
        "map50_95": final.get("map50_95"),
        "accuracy_history": accuracy,
        # Load balance, the ablation's headline.
        "mean_imbalance": round(statistics.fmean(imbalances), 4) if imbalances else None,
        "mean_straggler_gap_seconds": round(
            statistics.fmean([float(item.get("straggler_gap_seconds") or 0.0) for item in history]), 2
        )
        if history
        else None,
        # Communication.
        "comm_bytes": comm_bytes,
        "comm_seconds": round(comm_seconds, 2),
        "comm_fraction": round(comm_seconds / serial_seconds, 4) if serial_seconds > 0 else None,
        "rounds": history,
        "finished_at": time.time(),
    }


def time_to_accuracy(result: dict, target_map50: float) -> Optional[float]:
    """Training seconds until mAP50 first reached the target.

    The paper defines speedup against a fixed accuracy target rather than a
    fixed epoch count, and this is the measurement that supports that
    definition. Returns None when the target was never reached, which is itself
    a reportable outcome and must not be silently treated as zero.
    """
    for point in result.get("accuracy_history") or []:
        if (point.get("map50") or 0.0) >= target_map50:
            return float(point.get("train_seconds") or 0.0)
    return None


def annotate_baselines(results: Sequence[dict]) -> List[dict]:
    """Add speedup measured against the single-machine baseline.

    There are two different quantities here and conflating them is the easiest
    way to overclaim.

    `parallel_speedup` is the sum of shard times over round wall clock. It says
    how well work was overlapped *inside* a round, and on one machine it is
    slightly below 1 because orchestration is not free. It is a diagnostic.

    `speedup` is what the paper means and what a reader assumes: the wall clock
    of the same job on one machine, divided by the wall clock here. It needs the
    single-machine cell at the same dataset size, so it cannot be computed from
    one trial in isolation, which is why it is added in a second pass once the
    sweep has a baseline to compare against.

    Trials at a dataset size whose baseline never ran get None rather than a
    guess, because a speedup number with no baseline behind it is worse than a
    blank.

    The baseline is the median of its repeats rather than the mean, which is
    the more robust point estimate but is not on its own a fix. In leg 1 the
    three baseline repeats at 1000 images were 231.8, 321.8 and 355.1 seconds,
    because one machine alternated between roughly 15 and 29 images per second.
    That distribution has two modes, not one outlier, and no single summary of
    it is trustworthy: the mean gives 1.65x, the median 1.75x, the fastest
    repeat 1.26x and the slowest 1.93x, all for the same pair of measurements.

    So the point estimate travels with the range it came from.
    `speedup_low` and `speedup_high` are the same ratio recomputed against the
    slowest and fastest baseline repeats, and `baseline_spread` is max over min
    of those repeats. A sweep whose spread is near 1.0 can quote the point
    estimate plainly. One whose spread is 1.5, as leg 1's was, has to quote the
    range or be corrected by the first reviewer who reads the standard
    deviation printed next to the mean.
    """
    rows = [dict(result) for result in results]

    baselines: Dict[int, List[float]] = {}
    for row in rows:
        if (
            row.get("status") == STATUS_DONE
            and row.get("node_count") == 1
            and isinstance(row.get("train_seconds"), (int, float))
            and row["train_seconds"] > 0
        ):
            baselines.setdefault(row["sample_count"], []).append(float(row["train_seconds"]))

    medians = {size: statistics.median(values) for size, values in baselines.items()}
    # max/min over the baseline repeats: 1.0 means they agreed exactly, and
    # anything much above about 1.15 means the speedups built on it are soft.
    spreads = {
        size: (max(values) / min(values)) if min(values) > 0 else None
        for size, values in baselines.items()
    }

    for row in rows:
        # Keep the intra-round measure under its own name before overwriting.
        row["parallel_speedup"] = row.get("speedup")
        row["parallel_efficiency"] = row.get("efficiency")

        size = row.get("sample_count")
        baseline = medians.get(size)
        train = row.get("train_seconds")
        if baseline and isinstance(train, (int, float)) and train > 0:
            speedup = baseline / float(train)
            row["baseline_train_seconds"] = round(baseline, 2)
            row["baseline_runs"] = len(baselines.get(size) or [])
            row["baseline_spread"] = (
                round(spreads[size], 3) if spreads.get(size) is not None else None
            )
            row["speedup"] = round(speedup, 4)
            row["efficiency"] = round(speedup / max(1, row.get("node_count") or 1), 4)
            # The same ratio against the extreme baseline repeats. These are
            # the honest bounds on the headline number.
            repeats = baselines.get(size) or []
            row["speedup_low"] = round(min(repeats) / float(train), 4) if repeats else None
            row["speedup_high"] = round(max(repeats) / float(train), 4) if repeats else None
        else:
            row["baseline_train_seconds"] = None
            row["baseline_runs"] = 0
            row["baseline_spread"] = None
            row["speedup"] = None
            row["speedup_low"] = None
            row["speedup_high"] = None
            row["efficiency"] = None
    return rows


def node_stability(results: Sequence[dict]) -> List[dict]:
    """How steady each machine was across the sweep.

    A machine whose throughput wanders is not a measurement problem to be
    averaged away, it is the largest single source of variance in the result
    and it has to be visible. In leg 1 one node ranged from 10.4 to 31.7
    images per second across trials of the same cell while its partner held
    within 20%, and because nothing reported that, the drift was read as a
    difference between partitioning strategies.

    `spread` is max over min. Anything past about 1.5 means that node's numbers
    describe its background load as much as its hardware.
    """
    samples: Dict[str, Dict[str, Any]] = {}
    for row in results:
        for node in ((row.get("network") or {}).get("nodes") or []):
            rate = node.get("throughput_sps")
            if not isinstance(rate, (int, float)) or rate <= 0:
                continue
            entry = samples.setdefault(
                node.get("node_id") or "?",
                {"node_id": node.get("node_id"), "name": node.get("name"), "rates": []},
            )
            entry["rates"].append(float(rate))

    rows: List[dict] = []
    for entry in samples.values():
        rates = entry["rates"]
        if len(rates) < 2:
            continue
        mean = statistics.fmean(rates)
        stdev = statistics.pstdev(rates)
        rows.append(
            {
                "node_id": entry["node_id"],
                "name": entry["name"],
                "trials": len(rates),
                "throughput_mean_sps": round(mean, 2),
                "throughput_min_sps": round(min(rates), 2),
                "throughput_max_sps": round(max(rates), 2),
                "spread": round(max(rates) / min(rates), 2) if min(rates) > 0 else None,
                "coefficient_of_variation": round(stdev / mean, 3) if mean > 0 else None,
                "unstable": bool(min(rates) > 0 and max(rates) / min(rates) >= 1.5),
            }
        )
    return sorted(rows, key=lambda r: r.get("spread") or 0, reverse=True)


def aggregate_cells(results: Sequence[dict]) -> List[dict]:
    """Collapse repeats into mean and standard deviation per cell.

    Reviewers asked for mean plus standard deviation over at least three runs.
    A single run of anything on consumer hardware is a rumour.
    """
    cells: Dict[tuple, List[dict]] = {}
    for result in annotate_baselines(results):
        if result.get("status") != STATUS_DONE:
            continue
        key = (result["node_count"], result["sample_count"], result["strategy"])
        cells.setdefault(key, []).append(result)

    rows: List[dict] = []
    for (node_count, sample_count, strategy), group in sorted(cells.items()):
        row: Dict[str, Any] = {
            "node_count": node_count,
            "sample_count": sample_count,
            "strategy": strategy,
            "runs": len(group),
        }
        for metric in (
            "train_seconds",
            "speedup",
            "efficiency",
            "parallel_speedup",
            "map50",
            "map50_95",
            "mean_imbalance",
            "mean_straggler_gap_seconds",
            "comm_bytes",
            "comm_fraction",
        ):
            values = [r[metric] for r in group if isinstance(r.get(metric), (int, float))]
            if not values:
                row["%s_mean" % metric] = None
                row["%s_std" % metric] = None
                continue
            row["%s_mean" % metric] = round(statistics.fmean(values), 5)
            row["%s_std" % metric] = round(statistics.stdev(values), 5) if len(values) > 1 else 0.0
        rows.append(row)
    return rows


def accuracy_delta(cells: Sequence[dict]) -> List[dict]:
    """mAP of each cell minus the single-node cell at the same dataset size.

    This is the accuracy-difference column: what distribution cost in quality,
    measured against the baseline that shares its dataset.
    """
    baselines = {
        row["sample_count"]: row.get("map50_mean")
        for row in cells
        if row["node_count"] == 1 and row.get("map50_mean") is not None
    }
    rows = []
    for row in cells:
        baseline = baselines.get(row["sample_count"])
        value = row.get("map50_mean")
        rows.append(
            {
                **row,
                "baseline_map50": baseline,
                "delta_map50": round(value - baseline, 5)
                if baseline is not None and value is not None
                else None,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "trial_id",
    "index",
    "status",
    "node_count",
    "sample_count",
    "strategy",
    "repeat",
    "rounds_completed",
    "train_seconds",
    "baseline_train_seconds",
    "serial_estimate_seconds",
    "eval_seconds",
    "speedup",
    "efficiency",
    "parallel_speedup",
    "map50",
    "map50_95",
    "mean_imbalance",
    "mean_straggler_gap_seconds",
    "comm_bytes",
    "comm_seconds",
    "comm_fraction",
    "run_id",
    "error",
]


def write_suite(directory: Path, suite: dict) -> Path:
    """Write suite.json plus the flat CSV, atomically.

    Atomically because this is called after every trial, and a sweep interrupted
    mid-write would otherwise lose the whole day rather than the last trial.
    """
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "suite.json"
    temporary = directory / "suite.json.tmp"
    temporary.write_text(json.dumps(suite, indent=2, default=str), encoding="utf-8")
    temporary.replace(target)

    with (directory / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for result in annotate_baselines(suite.get("results", [])):
            writer.writerow({key: _flatten(result.get(key)) for key in CSV_COLUMNS})

    cells = aggregate_cells(suite.get("results", []))
    if cells:
        summary = accuracy_delta(cells)
        with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary)

    return target


def _flatten(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value)
    return value


def read_suite(directory: Path) -> Optional[dict]:
    target = directory / "suite.json"
    if not target.is_file():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return None


def campaign_summary(suites: Sequence[dict]) -> List[dict]:
    """Group sweeps into campaigns, newest first.

    A campaign is only interesting once it has more than one leg, but a
    single-leg campaign still appears so the next leg has something to attach
    to.
    """
    grouped: Dict[str, List[dict]] = {}
    for entry in suites:
        campaign = entry.get("campaign_id")
        if not campaign:
            continue
        grouped.setdefault(campaign, []).append(entry)

    campaigns = []
    for campaign_id, legs in grouped.items():
        legs.sort(key=lambda item: item.get("leg") or 0)
        campaigns.append(
            {
                "campaign_id": campaign_id,
                "name": legs[0].get("name"),
                "legs": legs,
                "networks": [leg.get("network_label") for leg in legs],
                "created_at": min((leg.get("created_at") or 0) for leg in legs),
                "complete_legs": sum(1 for leg in legs if leg.get("status") == "done"),
            }
        )
    campaigns.sort(key=lambda item: item["created_at"], reverse=True)
    return campaigns


def compare_networks(suites_with_results: Sequence[dict]) -> List[dict]:
    """One row per network label, for the cross-network comparison table.

    Averaged over every completed trial in that leg, which is the level the
    question is actually asked at: does this mesh behave differently on a
    different network. Per-cell detail stays in each leg's own results.
    """
    rows = []
    for suite in suites_with_results:
        results = [r for r in suite.get("results", []) if r.get("status") == STATUS_DONE]
        if not results:
            continue

        annotated = annotate_baselines(results)
        multi = [r for r in annotated if (r.get("node_count") or 1) > 1]

        def mean_of(source, key):
            values = [r[key] for r in source if isinstance(r.get(key), (int, float))]
            return round(statistics.fmean(values), 5) if values else None

        latencies = [
            node.get("latency_ms")
            for r in results
            for node in (r.get("network") or {}).get("nodes", [])
            if isinstance(node.get("latency_ms"), (int, float))
        ]

        rows.append(
            {
                "suite_id": suite.get("id"),
                "leg": suite.get("config", {}).get("leg", 1),
                "network_label": suite.get("config", {}).get("network_label"),
                "trials": len(results),
                "mean_latency_ms": round(statistics.fmean(latencies), 2) if latencies else None,
                "train_seconds": mean_of(results, "train_seconds"),
                "speedup": mean_of(multi, "speedup"),
                "efficiency": mean_of(multi, "efficiency"),
                "map50": mean_of(results, "map50"),
                "comm_fraction": mean_of(results, "comm_fraction"),
                "comm_bytes": mean_of(results, "comm_bytes"),
                "imbalance": mean_of(multi, "mean_imbalance"),
            }
        )
    rows.sort(key=lambda item: item["leg"])
    return rows


def list_suites(root: Path) -> List[dict]:
    """Lightweight index of every sweep on disk, newest first."""
    if not root.is_dir():
        return []
    entries = []
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        suite = read_suite(directory)
        if not suite:
            continue
        results = suite.get("results", [])
        entries.append(
            {
                "id": suite.get("id"),
                "name": suite.get("config", {}).get("name"),
                "status": suite.get("status"),
                "created_at": suite.get("created_at"),
                "finished_at": suite.get("finished_at"),
                "total_trials": len(suite.get("trials", [])),
                "completed_trials": sum(1 for r in results if r.get("status") == STATUS_DONE),
                "failed_trials": sum(1 for r in results if r.get("status") == STATUS_FAILED),
                "network_label": suite.get("config", {}).get("network_label"),
                "campaign_id": suite.get("config", {}).get("campaign_id"),
                "leg": suite.get("config", {}).get("leg", 1),
            }
        )
    entries.sort(key=lambda item: item.get("created_at") or 0, reverse=True)
    return entries
