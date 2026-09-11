# Testing and benchmarking

Everything needed to produce the numbers a reviewer will ask for, and an honest
account of what this harness does not measure.

Open **Testing parameters** in the dashboard sidebar.

---

## What a sweep is

A sweep is a factorial design. You choose the levels, it runs every cell the
required number of times, and it writes results after every trial so a day of
compute survives a crash.

| Factor | Levels you choose | Why it is a factor |
|---|---|---|
| Machine count | 1, 2, 3 … up to what is online | The scaling curve |
| Dataset size | 100, 1000, 10000 images | Does the mesh pay off more on bigger jobs |
| Partitioning | capability-proportional, equal shards | The ablation that validates the core claim |
| Repeats | 3 or more | Mean ± standard deviation, not one anecdote |

Trials run strictly one at a time. Two concurrent trials would share GPUs and
neither timing would mean anything.

### Controls that make the comparison valid

- **The same machines every time.** Going from four machines to two means
  choosing which two. The default takes the strongest by measured throughput, so
  the two-machine cell is always the same two machines and node count is the only
  thing that varies. A seeded random policy is available for sampling the space.
- **Subsets of one dataset, never different datasets.** A 1000-image cell is a
  reproducible subset of the parent, stored as a list of filenames rather than a
  copy. The **validation split never changes**, so a change in mAP is a change in
  what was learned rather than in what was measured.
- **A different training seed per repeat.** Ultralytics defaults to seed 0 with
  deterministic mode on. Without varying it, three repeats return bit-identical
  accuracy and the standard deviation is always zero. This was caught during
  testing: three repeats gave 0.0570, 0.0570, 0.0570 before the fix and 0.0570,
  0.0508, 0.0380 after.
- **Evaluation is never counted as training time.** It is timed separately and
  reported separately, so a configuration that evaluated more often does not look
  slower for reasons unrelated to distribution.
- **Evaluation runs on the coordinator.** Every configuration is scored by the
  same code on the same machine against the same images. Farming it out to
  whichever worker was idle would make accuracy depend on which GPU ran it.

---

## The two speedups, and which one to quote

This distinction matters more than any other number in the output.

**`speedup`** is wall clock on one machine divided by wall clock here, at the
same dataset size. This is what the paper means and what a reader assumes. It is
exactly 1.0 at one machine by construction. **Quote this one.**

**`parallel_speedup`** is the sum of shard times over round wall clock. It says
how well work was overlapped inside a round, and it sits slightly below 1 on a
single machine because orchestration is not free. It is a diagnostic, useful for
explaining *why* a speedup fell short. Do not quote it as speedup.

Efficiency is `speedup / machines`, from the first definition.

---

## Where to get datasets

You need **one dataset large enough to subsample**, not three different ones. The
harness derives 100, 1000 and 10000-image subsets from a single parent, which is
what keeps the comparison valid.

**Roboflow Universe** (`universe.roboflow.com`) is the fastest route. Filter for
object detection, pick a dataset with at least 10000 images, and export in
**YOLOv8** format. It hands you a zip with `images/train`, `labels/train`,
`images/val`, `labels/val` and a `data.yaml`, which is exactly what the upload
expects. No conversion.

Other options, roughly in order of how much work they are:

| Dataset | Size | Notes |
|---|---|---|
| Roboflow Universe | anything | Already in YOLO format. Start here. |
| VisDrone2019-DET | ~10k images | Ultralytics ships a converter. Good size for this sweep. |
| Global Wheat 2020 | ~4k images | Single class, quick to train, low label noise. |
| COCO128 / COCO8 | 128 / 8 | Ultralytics samples. Fine for a smoke test, far too small for results. |
| DOTA v1 | ~2800 large tiles | The usual choice for oriented boxes. Tiles are huge, so expect long epochs. |
| Full COCO | 118k | Overkill unless you have days. |

If your paper is specifically about **oriented bounding boxes**, use DOTA and a
`-obb` base model. The Intel XPU path already has an OBB trainer. Note that the
strawberry set carried over from v3 uses standard axis-aligned boxes, so it needs
a detection model rather than an OBB one.

Practical target: **10000 or more training images and at least 500 validation
images**. Below roughly 1000 training images the run-to-run noise is larger than
the effect being measured, which is visible in the smoke tests in this repo.

---

## Running the sweep

1. Get the machines on the mesh. **Discover devices** shows what is on the
   network; each contributor pastes the one-line join command. There is no cap on
   machine count, so six or seven is fine.
2. Upload the parent dataset on **Datasets**.
3. Open **Testing parameters**, choose the levels, and read the trial count and
   estimated duration before starting. Four checkboxes separate a twenty-minute
   sweep from an overnight one.
4. Press **Run the sweep**. It survives a coordinator restart: reopen it and
   press **Resume** to continue from the first trial without a result.

### Getting the figures

```bash
npm run report              # most recent sweep
npm run report <suite-id>   # a specific one
```

Six figures, each as PNG and PDF, plus `report.md` with the setup and results
tables in markdown. Written beside `suite.json`.

| Figure | Shows |
|---|---|
| `fig1_scaling` | Speedup and efficiency against machine count, with the linear-speedup ceiling drawn |
| `fig2_partitioning` | The ablation: shard-time imbalance and wall clock, proportional versus equal |
| `fig3_accuracy` | Final mAP by machine count, and the difference from the one-machine baseline |
| `fig4_time_to_accuracy` | mAP against cumulative training time, one curve per machine count |
| `fig5_communication` | Network share of worker time, and total bytes moved |
| `fig6_dataset_scaling` | Wall clock against dataset size on a log axis |

### Raw output

Everything lands in `.gradmesh/benchmarks/<suite-id>/`:

- **`suite.json`** — the complete record, and the source of truth. Every trial,
  every round, every accuracy point, plus the full environment block.
- **`results.csv`** — one row per trial, for a spreadsheet.
- **`summary.csv`** — one row per cell with mean and standard deviation.

The dashboard downloads all three. If a figure and the JSON disagree, the JSON is
right and the plotting code has a bug.

---

## Network conditions

The harness **records** network conditions, it does not **impose** them.

Every trial stores the observed per-machine control-plane latency, the measured
throughput, and the bytes transferred. The sweep carries a free-text
`network_label`, so results gathered on lab Ethernet, on college Wi-Fi and on a
phone hotspot stay distinguishable in one dataset.

To characterise a spectrum, run the same sweep once per network and compare by
label. That is a real experiment and it needs no special privileges.

**Deliberately not implemented: traffic shaping.** Imposing 50 ms of latency or
capping bandwidth needs `tc netem` on Linux or Clumsy on Windows, both requiring
administrator rights, with no portable interface. Shipping a half-working version
that silently does nothing on one platform would put fabricated conditions in a
results table. If you need a controlled latency sweep, run `tc netem` on the
worker machines yourself and label each sweep accordingly. The recorded latency
in the output will show whether the shaping actually took effect.

---

## What this harness does not measure

Being explicit here is worth more than a shaky claim. Each of these is a real gap
against a thorough review checklist.

**Time to a fixed accuracy target.** Trials run a fixed number of rounds, not
until a target mAP. The data needed is captured: `accuracy_history` records mAP
against cumulative training seconds per round, and `benchmark.time_to_accuracy()`
computes the crossing. What is not automated is *stopping* at a target. State the
fixed-round design explicitly in the paper rather than implying otherwise.

**Fault-tolerance timing.** Detection and recovery work and are visible in the
event stream, but there is no scripted fault-injection harness that kills a
worker at 25%, 50% and 75% through a shard and reports detect and recover
intervals separately. Doing it by hand is possible; the events carry timestamps.

**Agent and coordinator resource overhead.** No CPU, memory or idle-bandwidth
measurement for the agent, and no coordinator scaling curve against worker count.
The claim that the agent is lightweight is currently unquantified.

**Container isolation overhead.** There is no container layer, so there is
nothing to measure. Workers run in a virtualenv under the contributor's home
folder.

**Non-NVIDIA comparison.** The capability probe is already vendor-neutral: it
measures achieved FP32 matmul throughput and memory bandwidth rather than reading
CUDA compute capability, so an Intel XPU device is scored on the same scale as an
NVIDIA one, and the scheduler does not care which vendor a machine is. Apple
Silicon via MPS is **not** wired up: `accelerator.py` handles CUDA, XPU and CPU
only. Adding MPS is a small change to that one file, but it is untested, so treat
cross-vendor support as covering NVIDIA and Intel today and scope Apple as future
work rather than claiming it.

**Strategy selection between communication topologies.** There is one
communication pattern, a coordinator-mediated parameter exchange. There is no
ring all-reduce and therefore no strategy-selection logic to validate. Do not
claim adaptive topology selection.

---

## Disclosure block

`suite.json` carries an `environment` object recorded at sweep start, and
`report.md` renders it as a table. It contains every machine's GPU name, backend,
memory and measured GFLOP/s, the coordinator's platform, PyTorch, CUDA and
Ultralytics versions, the dataset name with train and validation counts and class
names, the scheduler policy in force, and the network label.

That covers the disclosure a reviewer checks before they check your numbers, with
one exception worth adding by hand: **GPU driver versions**, which are not
collected. Record them yourself with `nvidia-smi` on each machine.
