"""Figures and tables from a finished sweep.

Run it as `npm run report` or directly:

    python report.py <suite-id-or-path> [--out DIR]

Everything it produces comes from `suite.json`, which is the source of truth. If
a figure and the JSON ever disagree, the JSON is right and this file has a bug.

Chart choices are made for a paper rather than for a dashboard: no gridlines
competing with the data, error bars everywhere a mean is shown, one idea per
figure, and captions that state what the reader should conclude. Every figure is
written both as PNG for slides and PDF for LaTeX.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ENGINE_DIR = Path(__file__).resolve().parent
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

# A colourblind-safe sequence. Dataset sizes are ordered, so the series read as
# a progression rather than as unrelated categories.
SERIES_COLOURS = ["#3b5bd9", "#22a2c4", "#8a63d2", "#d98b3b", "#4a9d6a"]
ARM_COLOURS = {"proportional": "#3b5bd9", "equal": "#c9563c"}


def require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        print(
            "matplotlib is not installed in this environment.\n"
            "  Windows: .venv\\Scripts\\python.exe -m pip install matplotlib\n"
            "  macOS, Linux: .venv/bin/python -m pip install matplotlib",
            file=sys.stderr,
        )
        raise SystemExit(2)


def style(plt) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "medium",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
        }
    )


# ---------------------------------------------------------------------------
# Data shaping
# ---------------------------------------------------------------------------


def load_suite(target: str) -> Dict[str, Any]:
    path = Path(target)
    if path.is_dir():
        path = path / "suite.json"
    if not path.is_file():
        # Treat a bare argument as a suite id under the default state directory.
        from coordinator import store

        path = store.STATE_DIR / "benchmarks" / target / "suite.json"
    if not path.is_file():
        raise SystemExit("No suite.json found for %r" % target)
    return json.loads(path.read_text(encoding="utf-8"))


def done_results(suite: Dict[str, Any]) -> List[dict]:
    """Completed trials, with baseline-relative speedup filled in.

    Figures must never plot the raw per-trial `speedup`, which before this pass
    is the intra-round overlap measure rather than the speedup against one
    machine.
    """
    from coordinator import benchmark

    annotated = benchmark.annotate_baselines(suite.get("results", []))
    return [r for r in annotated if r.get("status") == "done"]


def group(results: Sequence[dict], strategy: str) -> Dict[int, Dict[int, List[dict]]]:
    """sample_count -> node_count -> results, for one partitioning arm."""
    out: Dict[int, Dict[int, List[dict]]] = {}
    for result in results:
        if result.get("strategy") != strategy:
            continue
        out.setdefault(result["sample_count"], {}).setdefault(result["node_count"], []).append(result)
    return out


def mean_std(values: Sequence[float]) -> tuple:
    clean = [float(v) for v in values if isinstance(v, (int, float))]
    if not clean:
        return (None, 0.0)
    return (statistics.fmean(clean), statistics.stdev(clean) if len(clean) > 1 else 0.0)


def series(cells: Dict[int, List[dict]], metric: str) -> tuple:
    """Sorted node counts with mean and standard deviation of one metric."""
    counts = sorted(cells)
    means, errors = [], []
    for count in counts:
        mean, std = mean_std([r.get(metric) for r in cells[count]])
        means.append(mean)
        errors.append(std)
    keep = [i for i, value in enumerate(means) if value is not None]
    return ([counts[i] for i in keep], [means[i] for i in keep], [errors[i] for i in keep])


def save(plt, fig, out: Path, name: str) -> List[str]:
    written = []
    for suffix in ("png", "pdf"):
        path = out / ("%s.%s" % (name, suffix))
        fig.savefig(path)
        written.append(str(path))
    plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def figure_scaling(plt, suite, out: Path) -> List[str]:
    """Speedup and efficiency against node count. The centrepiece."""
    grouped = group(done_results(suite), "proportional")
    if not grouped:
        return []

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.1))

    for index, size in enumerate(sorted(grouped)):
        colour = SERIES_COLOURS[index % len(SERIES_COLOURS)]
        counts, means, errors = series(grouped[size], "speedup")
        if counts:
            left.errorbar(counts, means, yerr=errors, marker="o", capsize=3, color=colour,
                          label="%d images" % size)
        counts, means, errors = series(grouped[size], "efficiency")
        if counts:
            right.errorbar(counts, means, yerr=errors, marker="o", capsize=3, color=colour,
                           label="%d images" % size)

    all_counts = sorted({c for cells in grouped.values() for c in cells})
    if all_counts:
        # Linear speedup is the ceiling. Showing it turns "is 2.4x good?" into a
        # question the reader can answer at a glance.
        left.plot(all_counts, all_counts, linestyle="--", linewidth=1.1, color="#8a94a6",
                  label="linear speedup")
        right.axhline(1.0, linestyle="--", linewidth=1.1, color="#8a94a6", label="perfect efficiency")

    left.set_xlabel("Machines")
    left.set_ylabel("Speedup over single machine")
    left.set_title("Speedup against the one-machine baseline")
    left.legend()
    left.set_xticks(all_counts)

    right.set_xlabel("Machines")
    right.set_ylabel("Efficiency (speedup / machines)")
    right.set_title("Efficiency falls as machines are added")
    right.set_ylim(0, 1.15)
    right.legend()
    right.set_xticks(all_counts)

    fig.suptitle("Scaling behaviour of the mesh", y=1.02, fontsize=12)
    return save(plt, fig, out, "fig1_scaling")


def figure_partitioning(plt, suite, out: Path) -> List[str]:
    """The ablation: proportional against equal shards on identical hardware."""
    results = done_results(suite)
    proportional = group(results, "proportional")
    equal = group(results, "equal")
    if not equal:
        return []

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.1))
    width = 0.36

    counts = sorted({c for cells in list(proportional.values()) + list(equal.values()) for c in cells if c > 1})
    if not counts:
        plt.close(fig)
        return []
    positions = list(range(len(counts)))

    # Imbalance is dimensionless, so pooling dataset sizes is legitimate.
    # Wall clock is not: a 100-image run and a 1000-image run differ by an order
    # of magnitude, and averaging them would produce error bars that describe
    # the dataset rather than the strategy. So the timing panel fixes the
    # dataset at one size and says which.
    sizes = sorted({r["sample_count"] for r in results})
    timing_size = sizes[-1] if sizes else None

    def pooled(source, count, metric, only_size=None):
        values = []
        for size, cells in source.items():
            if only_size is not None and size != only_size:
                continue
            values.extend(r.get(metric) for r in cells.get(count, []))
        return mean_std(values)

    for offset, (arm, source) in enumerate((("proportional", proportional), ("equal", equal))):
        means, errors = [], []
        for count in counts:
            mean, std = pooled(source, count, "mean_imbalance")
            means.append(mean or 0.0)
            errors.append(std)
        left.bar([p + (offset - 0.5) * width for p in positions], means, width,
                 yerr=errors, capsize=3, color=ARM_COLOURS[arm], label=arm)

        means, errors = [], []
        for count in counts:
            mean, std = pooled(source, count, "train_seconds", only_size=timing_size)
            means.append(mean or 0.0)
            errors.append(std)
        right.bar([p + (offset - 0.5) * width for p in positions], means, width,
                  yerr=errors, capsize=3, color=ARM_COLOURS[arm], label=arm)

    for axis, ylabel, title in (
        (left, "Shard-time imbalance (lower is better)", "Capability-proportional shards finish together"),
        (right, "Training wall clock (s)", "and finish sooner (%s images)" % timing_size),
    ):
        axis.set_xticks(positions)
        axis.set_xticklabels([str(c) for c in counts])
        axis.set_xlabel("Machines")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.legend()

    fig.suptitle("Partitioning ablation, identical hardware in both arms", y=1.02, fontsize=12)
    return save(plt, fig, out, "fig2_partitioning")


def figure_accuracy(plt, suite, out: Path) -> List[str]:
    """Accuracy retention: does distributing cost quality?"""
    grouped = group(done_results(suite), "proportional")
    usable = {
        size: cells
        for size, cells in grouped.items()
        if any(r.get("map50") is not None for cells_list in cells.values() for r in cells_list)
    }
    if not usable:
        return []

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.1))

    for index, size in enumerate(sorted(usable)):
        colour = SERIES_COLOURS[index % len(SERIES_COLOURS)]
        counts, means, errors = series(usable[size], "map50")
        if not counts:
            continue
        left.errorbar(counts, means, yerr=errors, marker="o", capsize=3, color=colour,
                      label="%d images" % size)

        baseline = means[0] if counts and counts[0] == 1 else None
        if baseline is not None:
            right.errorbar(counts, [m - baseline for m in means], yerr=errors, marker="o",
                           capsize=3, color=colour, label="%d images" % size)

    left.set_xlabel("Machines")
    left.set_ylabel("mAP@50")
    left.set_title("Final accuracy by machine count")
    left.legend()

    right.axhline(0.0, linestyle="--", linewidth=1.1, color="#8a94a6")
    right.set_xlabel("Machines")
    right.set_ylabel("mAP@50 minus single-machine baseline")
    right.set_title("Accuracy difference from distributing")
    right.legend()

    fig.suptitle("Accuracy is the other half of speedup", y=1.02, fontsize=12)
    return save(plt, fig, out, "fig3_accuracy")


def figure_time_to_accuracy(plt, suite, out: Path) -> List[str]:
    """mAP against cumulative training time. Speedup at a fixed accuracy."""
    results = [r for r in done_results(suite) if r.get("strategy") == "proportional"]
    if not results:
        return []

    sizes = sorted({r["sample_count"] for r in results})
    largest = sizes[-1]
    curves = [r for r in results if r["sample_count"] == largest and r.get("accuracy_history")]
    if not curves:
        return []

    fig, axis = plt.subplots(figsize=(6.6, 4.2))
    by_count: Dict[int, List[dict]] = {}
    for result in curves:
        by_count.setdefault(result["node_count"], []).append(result)

    for index, count in enumerate(sorted(by_count)):
        colour = SERIES_COLOURS[index % len(SERIES_COLOURS)]
        # One representative repeat: the median final accuracy, so the curve is
        # a real run rather than an average of runs with different lengths.
        chosen = sorted(by_count[count], key=lambda r: r.get("map50") or 0.0)[len(by_count[count]) // 2]
        points = chosen.get("accuracy_history") or []
        axis.plot(
            [p.get("train_seconds", 0.0) for p in points],
            [p.get("map50", 0.0) for p in points],
            marker="o",
            color=colour,
            label="%d machine%s" % (count, "" if count == 1 else "s"),
        )

    axis.set_xlabel("Training wall clock (s), evaluation excluded")
    axis.set_ylabel("mAP@50")
    axis.set_title("Time to reach a given accuracy (%d images)" % largest)
    axis.legend()
    return save(plt, fig, out, "fig4_time_to_accuracy")


def figure_communication(plt, suite, out: Path) -> List[str]:
    """How much of a round is network, and how that grows with machines."""
    grouped = group(done_results(suite), "proportional")
    if not grouped:
        return []

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.1))
    plotted = False

    for index, size in enumerate(sorted(grouped)):
        colour = SERIES_COLOURS[index % len(SERIES_COLOURS)]
        counts, means, errors = series(grouped[size], "comm_fraction")
        if counts:
            left.errorbar(counts, [m * 100 for m in means], yerr=[e * 100 for e in errors],
                          marker="o", capsize=3, color=colour, label="%d images" % size)
            plotted = True
        counts, means, _ = series(grouped[size], "comm_bytes")
        if counts:
            right.plot(counts, [m / 1e6 for m in means], marker="o", color=colour,
                       label="%d images" % size)

    if not plotted:
        plt.close(fig)
        return []

    left.set_xlabel("Machines")
    left.set_ylabel("Communication share of worker time (%)")
    left.set_title("Communication overhead")
    left.legend()

    right.set_xlabel("Machines")
    right.set_ylabel("Total transferred (MB)")
    right.set_title("Bytes moved per run")
    right.legend()

    fig.suptitle("Communication cost, the limit on scaling past a LAN", y=1.02, fontsize=12)
    return save(plt, fig, out, "fig5_communication")


def figure_dataset_scaling(plt, suite, out: Path) -> List[str]:
    """Training time against dataset size, one line per machine count."""
    results = done_results(suite)
    sizes = sorted({r["sample_count"] for r in results})
    if len(sizes) < 2:
        return []

    by_count: Dict[int, Dict[int, List[dict]]] = {}
    for result in results:
        if result.get("strategy") != "proportional":
            continue
        by_count.setdefault(result["node_count"], {}).setdefault(result["sample_count"], []).append(result)

    fig, axis = plt.subplots(figsize=(6.6, 4.2))
    for index, count in enumerate(sorted(by_count)):
        colour = SERIES_COLOURS[index % len(SERIES_COLOURS)]
        xs, ys, errs = [], [], []
        for size in sizes:
            mean, std = mean_std([r.get("train_seconds") for r in by_count[count].get(size, [])])
            if mean is None:
                continue
            xs.append(size)
            ys.append(mean)
            errs.append(std)
        if xs:
            axis.errorbar(xs, ys, yerr=errs, marker="o", capsize=3, color=colour,
                          label="%d machine%s" % (count, "" if count == 1 else "s"))

    axis.set_xscale("log")
    axis.set_xlabel("Training images (log scale)")
    axis.set_ylabel("Training wall clock (s)")
    axis.set_title("Cost against dataset size")
    axis.legend()
    return save(plt, fig, out, "fig6_dataset_scaling")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def write_markdown(suite: Dict[str, Any], out: Path) -> str:
    from coordinator import benchmark

    results = suite.get("results", [])
    cells = benchmark.accuracy_delta(benchmark.aggregate_cells(results))
    config = suite.get("config", {})
    environment = suite.get("environment", {})

    lines: List[str] = []
    lines.append("# %s" % config.get("name", "GradMesh sweep"))
    lines.append("")
    lines.append("Sweep `%s`. %d of %d trials completed." % (
        suite.get("id"),
        sum(1 for r in results if r.get("status") == "done"),
        len(suite.get("trials", [])),
    ))
    lines.append("")

    lines.append("## Setup")
    lines.append("")
    dataset = environment.get("dataset", {})
    lines.append("| Item | Value |")
    lines.append("| --- | --- |")
    lines.append("| Dataset | %s, %s train / %s val images |" % (
        dataset.get("name"), dataset.get("train_count"), dataset.get("val_count")))
    lines.append("| Classes | %s |" % ", ".join(dataset.get("class_names") or []))
    lines.append("| Model | %s |" % config.get("base_model"))
    lines.append("| Image size | %s |" % config.get("imgsz"))
    lines.append("| Batch size (ceiling) | %s |" % config.get("batch_size"))
    lines.append("| Rounds per trial | %s |" % config.get("rounds"))
    lines.append("| Repeats per cell | %s |" % config.get("repeats"))
    lines.append("| Network | %s |" % config.get("network_label"))
    coordinator = environment.get("coordinator", {})
    lines.append("| Coordinator | %s, torch %s, ultralytics %s |" % (
        coordinator.get("platform"), coordinator.get("torch"), coordinator.get("ultralytics")))
    lines.append("")

    lines.append("### Machines")
    lines.append("")
    lines.append("| Name | GPU | Backend | Memory | Measured GFLOP/s |")
    lines.append("| --- | --- | --- | --- | --- |")
    for node in environment.get("nodes", []):
        capability = node.get("capability") or {}
        lines.append("| %s | %s | %s | %s MB | %s |" % (
            node.get("name"), node.get("gpu"), node.get("backend"),
            node.get("memory_mb"), capability.get("gflops")))
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append("Mean ± standard deviation over repeats. Training time excludes evaluation.")
    lines.append("")
    lines.append("| Machines | Images | Strategy | Runs | Train time (s) | Speedup | Efficiency | mAP@50 | Δ mAP@50 | Imbalance |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in cells:
        def cell(metric, digits=3):
            mean = row.get("%s_mean" % metric)
            std = row.get("%s_std" % metric)
            if mean is None:
                return "—"
            return "%.*f ± %.*f" % (digits, mean, digits, std or 0.0)

        delta = row.get("delta_map50")
        lines.append("| %d | %d | %s | %d | %s | %s | %s | %s | %s | %s |" % (
            row["node_count"], row["sample_count"], row["strategy"], row["runs"],
            cell("train_seconds", 1), cell("speedup", 2), cell("efficiency", 2),
            cell("map50", 4), ("%+.4f" % delta) if delta is not None else "—",
            cell("mean_imbalance", 3),
        ))
    lines.append("")

    failures = [r for r in results if r.get("status") not in {"done", None}]
    if failures:
        lines.append("## Trials that did not complete")
        lines.append("")
        lines.append("| Trial | Status | Reason |")
        lines.append("| --- | --- | --- |")
        for result in failures:
            lines.append("| %s | %s | %s |" % (result.get("label"), result.get("status"),
                                               (result.get("error") or "")[:160]))
        lines.append("")

    path = out / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build(target: str, out_dir: Optional[str] = None) -> Dict[str, Any]:
    suite = load_suite(target)
    out = Path(out_dir) if out_dir else Path(load_suite_dir(target))
    out.mkdir(parents=True, exist_ok=True)

    plt = require_matplotlib()
    style(plt)

    written: List[str] = []
    for builder in (
        figure_scaling,
        figure_partitioning,
        figure_accuracy,
        figure_time_to_accuracy,
        figure_communication,
        figure_dataset_scaling,
    ):
        try:
            written.extend(builder(plt, suite, out))
        except Exception as exc:
            print("  skipped %s: %s" % (builder.__name__, exc), file=sys.stderr)

    written.append(write_markdown(suite, out))
    return {"suite_id": suite.get("id"), "output_dir": str(out), "files": written}


def load_suite_dir(target: str) -> Path:
    path = Path(target)
    if path.is_dir():
        return path
    if path.is_file():
        return path.parent
    from coordinator import store

    return store.STATE_DIR / "benchmarks" / target


def main() -> None:
    parser = argparse.ArgumentParser(description="Build figures and tables from a GradMesh sweep")
    parser.add_argument("suite", help="Sweep id, its directory, or a path to suite.json")
    parser.add_argument("--out", default=None, help="Where to write figures (defaults beside suite.json)")
    args = parser.parse_args()

    result = build(args.suite, args.out)
    print("Wrote %d files to %s" % (len(result["files"]), result["output_dir"]))
    for path in result["files"]:
        print("  " + path)


if __name__ == "__main__":
    main()
