# Testing output, leg 1

First full run of the Testing parameters sweep. Two machines, two sweeps, one
day. This file says what the numbers mean, what is safe to put in the paper,
what is not, and what to change before leg 2.

Read it alongside `TESTING.md`, which describes how to run a sweep. This file is
only about what came out of the first one.

| | |
|---|---|
| Date | 11 September 2026 |
| Sweep A | `8c5f4e773071`, COCO8, 02:30–02:46 UTC, 16 min, 18/18 trials |
| Sweep B | `387af7e617b9`, African Wildlife, 02:50–03:55 UTC, 65 min, 18/18 trials |
| Machines | DESKTOP-RV928A7 and COE-2, both RTX 5070, 12 GB, torch 2.11.0+cu128 |
| Design | 2 machine counts × 2 dataset sizes × 2 strategies × 3 repeats, 5 rounds each |

**Sweep A should be discarded.** Everything below refers to sweep B unless it
says otherwise. Section 7 explains why A is void.

---

## Verdict up front

The harness works. Eighteen trials ran unattended for an hour with no crash, no
manual intervention, and produced JSON, CSV and six figures on their own. That
was the main thing leg 1 had to prove, and it proved it.

The science is a different matter. Of the four things the sweep was designed to
measure, one came out clean, one came out real but not yet defensible, and two
produced nothing usable.

| Axis | Outcome | Usable in the paper? |
|---|---|---|
| Machine count × dataset size | Clean, and matches a fitted model | **Yes**, this is the leg-1 result |
| Hardware heterogeneity | Clean, and stronger than expected | **Yes**, it motivates the scheduler |
| Partitioning ablation | Real effect, confounded measurement | Not yet, see section 3 |
| Accuracy | No signal, and the cause is a defect | **No**, see section 4 |

Two supporting measurements are also broken: the single-machine baseline is too
noisy to divide by (section 5), and the network latency probe returns zero
(section 6). Both need fixing before the bigger runs, because leg 2 multiplies
the cost of every one of these problems by the number of machines you add.

---

## 1. The result that holds: fixed cost per round sets the scaling floor

This is the finding. Adding a second machine helped enormously at 1000 images
and barely at all at 100.

| Machines | Images | Train time per trial | Speedup | Efficiency |
|---|---|---|---|---|
| 1 | 100 | 114.7 ± 4.3 s | 1.00 | 1.00 |
| 2 | 100 | 107.3 ± 1.6 s | 1.07 | 0.54 |
| 1 | 1000 | 302.9 ± 63.8 s | 1.00 | 1.00 |
| 2 | 1000 | 184.1 ± 0.5 s | 1.65 | 0.82 |

The reason is a large fixed cost paid once per round per machine, independent of
how many images that machine was given. Fitting a straight line through the two
single-machine cells gives a per-round cost of:

```
T(n) = 18.77 s  +  0.0418 s/image × n
```

Nineteen seconds of setup, then about 24 images per second of actual work. That
fixed 19 seconds is model download, dataset unpack, CUDA context creation, cuDNN
autotuning and Ultralytics trainer construction, and none of it shrinks when you
split the data.

So at 100 images per round, **82% of the round is fixed cost**. Splitting the
remaining 18% across two machines cannot produce much. At 1000 images the fixed
share drops to 31%, and splitting the other 69% is worth doing.

**The model was fitted only on single-machine data, and it predicts the
two-machine cells it never saw, within 8%:**

| Configuration | Observed per round | Predicted | Error |
|---|---|---|---|
| 2 machines, 100 img, equal | 21.32 s | 20.86 s | −2.2% |
| 2 machines, 100 img, proportional | 21.45 s | 20.86 s | −2.8% |
| 2 machines, 1000 img, equal | 36.82 s | 39.67 s | +7.7% |
| 2 machines, 1000 img, proportional | 38.30 s | 39.67 s | +3.6% |

That cross-validation is what makes this publishable rather than anecdotal. Two
points fitted, four points predicted, no free parameters left over.

### What it implies for leg 2

Extrapolating the same model to more machines gives the speedup ceiling for each
combination. This is an optimistic bound: it assumes perfect overlap and ignores
the fact that communication grows with machine count, so treat it as the best
case rather than a forecast.

| Images per round | 2 machines | 3 | 4 | 6 | 7 |
|---|---|---|---|---|---|
| 100 | 1.10x | 1.14x | 1.16x | 1.18x | 1.19x |
| 1,000 | 1.53x | 1.85x | 2.07x | 2.35x | 2.45x |
| 4,000 | 1.82x | 2.50x | 3.07x | 3.99x | 4.36x |
| 10,000 | 1.92x | 2.76x | 3.54x | 4.94x | 5.57x |

And, inverted, the dataset size each machine count needs before it holds 80%
efficiency:

| Machines | Images per round needed for 80% efficiency |
|---|---|
| 2 | ~1,350 |
| 3 | ~3,150 |
| 4 | ~4,900 |
| 6 | ~8,500 |
| 7 | ~10,300 |

**This is the single most useful number to come out of leg 1.** Running six or
seven college machines on 1000 images would return about 2.4x and look like a
failure of the system, when it is really a failure of the experiment design. The
ten/hundred/thousand ladder is too short at the top. For a seven-machine leg the
largest rung needs to be 10,000 images, which is exactly the rung already
planned, so keep it and drop the 10 and 100 rungs to buy the time.

Keep the small rungs in the sweep anyway, but frame them as the measurement of
the overhead floor rather than as a scaling failure. "Distribution does not pay
below ~600 images per round, and here is the fixed cost that explains why" is a
better paper sentence than a speedup table with a flat row in it.

---

## 2. Identical GPUs are not identical, which is the whole argument for the scheduler

Both machines have an RTX 5070 with 12,226 MB and the same driver-reported
capability. The capability probe scored them within 4% of each other. Their
actual training throughput differed by 103%.

| | DESKTOP-RV928A7 | COE-2 | Ratio |
|---|---|---|---|
| Probe, synthetic FP32 matmul | 24,185 GFLOP/s | 23,207 GFLOP/s | 1.04x |
| Measured YOLO throughput, 1000-image trials | 29.9 img/s | 14.7 img/s | **2.03x** |

A synthetic matmul benchmark predicted these machines were interchangeable.
Training them proved they were not, by a factor of two. That is a direct,
measured justification for scheduling on observed throughput with an EWMA rather
than on advertised or probed specification, and it is a stronger result than the
heterogeneous-hardware case would have been, because here there is no hardware
difference to point at at all.

The likely cause is that COE-2 also hosts the coordinator: it serves shards, runs
FedAvg aggregation over roughly 25 MB per round, and runs the evaluation pass.
Worth confirming in leg 2 by running one leg with the coordinator on a machine
that is not also a worker. Either way it is a real effect that a specification
sheet cannot see, which is the point.

---

## 3. The partitioning ablation: a real effect, measured in a way we cannot defend

The imbalance result looks good:

| Machines | Images | Strategy | Shard-time imbalance | Straggler gap |
|---|---|---|---|---|
| 2 | 1000 | equal | 0.319 ± 0.011 | 17.2 s |
| 2 | 1000 | proportional | 0.107 ± 0.080 | 6.1 s |
| 2 | 100 | equal | 0.067 ± 0.029 | 2.5 s |
| 2 | 100 | proportional | 0.126 ± 0.019 | 4.6 s |

At 1000 images, proportional partitioning cut imbalance by a factor of three and
closed 11 seconds of straggler gap. That is the behaviour the design predicts.

Three things stop this from being a paper claim yet.

**It reverses at 100 images.** Proportional was *worse* balanced than equal
(0.126 vs 0.067). The splits it chose swung from 51/49 to 79/21 and back to
75/25 across five rounds, because the throughput estimate is noisy when each
shard is only 50 images and the fixed 19 seconds swamps the signal. The
scheduler is over-correcting on noise. Figure 2 averages both dataset sizes and
so hides this reversal; if that figure goes in the paper, split it by size.

**It did not translate into wall clock.** Proportional took 191.5 s against
equal's 184.1 s. Better balance, worse time. On its own that would sink the
argument.

**But the comparison is confounded, and that explains the contradiction.** All
three proportional trials ran before all three equal trials. In between,
DESKTOP-RV928A7's throughput changed:

| Arm | DESKTOP throughput per trial | Mean | Spread |
|---|---|---|---|
| proportional | 10.43, 17.46, 31.65 img/s | 19.9 | 3.0x |
| equal | 29.55, 29.68, 30.44 img/s | 29.9 | 1.0x |

COE-2 was steady at about 14 img/s throughout. DESKTOP was not: it ran **51%
faster during the equal arm than during the proportional arm**, and swung by 3x
within the proportional arm alone. Because the arms ran sequentially rather than
interleaved, that drift maps directly onto the strategy variable. The equal arm
simply got a better machine.

Read correctly, this makes proportional look better, not worse: it achieved
one-third the imbalance while running on a slower and far more erratic node, and
its one trial where DESKTOP was fast (31.65 img/s) finished in 156.8 s, well
under anything the equal arm managed. But "read correctly" is not a measurement.

The fix is one line of experiment design, in section 8.

---

## 4. Accuracy produced no signal, and the cause is a defect rather than a result

Final mAP@50 across the whole sweep sits between 0.007 and 0.037. After five
rounds of YOLOv8n starting from COCO-pretrained weights on a four-class wildlife
set of 1052 images, that is roughly an order of magnitude below where it should
be.

Worse, **accuracy peaks after round 1 and falls every round after it.**

| Round | 1 machine, 1000 images | 2 machines, 1000 images |
|---|---|---|
| 1 | 0.0246 | 0.0274 |
| 2 | 0.0145 | 0.0082 |
| 3 | 0.0145 | 0.0042 |
| 4 | 0.0115 | 0.0080 |
| 5 | 0.0117 | 0.0050 |

That is figure 4, the time-to-accuracy plot, and it currently shows the model
getting worse the longer it trains, on both machine counts. It is not a scaling
curve, it is a bug report.

### The likely cause

Each round constructs a fresh Ultralytics trainer and calls `model.train()` with
`epochs=1`, inheriting every other default. Two of those defaults matter:

```
warmup_epochs: 3.0     # LR and momentum warmup lasts three epochs
warmup_bias_lr: 0.1    # bias learning rate during warmup
```

With one epoch per round against a three-epoch warmup, **every round runs
entirely inside warmup and never exits it.** The optimizer is rebuilt from
scratch each round, so momentum restarts at 0.8 every time, and the bias group
runs at a learning rate of 0.1 throughout, which is very large for a fine-tuned
detection head. The model is effectively given five independent warmup shocks and
never reaches the stable phase where it would actually converge.

Two pieces of evidence support this over the alternatives. First, the decline
happens on **single-machine** trials too, where no averaging occurs, so FedAvg is
not the culprit. Second, sweep A reached mAP 0.87 on COCO8 — but COCO8's
validation set is its training set, four images, so memorisation succeeds even at
a bad learning rate. The training and evaluation path works; the schedule does
not.

**I have not changed this.** The training call is explicitly fenced off, and the
code says so. The check is a single run with `warmup_epochs=0.0`, or with more
epochs per round, on one machine at 1000 images for 5 rounds. If mAP climbs
monotonically, that confirms it.

### What this means for the accuracy axis

Nothing on this axis is currently interpretable. The measured effect of
distributing is −0.006 mAP with a standard deviation of 0.014, so it is not
distinguishable from zero. With three repeats at that variance, the smallest
difference the design could detect is about **0.033 mAP** — nearly twice the
entire signal being measured. Figure 3 should not go in the paper.

Fix accuracy before leg 2. Running six machines against the same defect just
reproduces the same null result at six times the cost.

---

## 5. The single-machine baseline is too noisy to divide by

Every speedup number in the sweep is a ratio against the one-machine cell, and
that cell has a coefficient of variation of 21%:

```
1 machine, 1000 images:  231.79 s,  321.82 s,  355.08 s   ->  302.9 ± 63.8 s
```

The headline 1.65x therefore depends heavily on which repeat you happen to divide
by:

| Baseline used | Value | Resulting speedup |
|---|---|---|
| Fastest repeat | 231.8 s | 1.26x |
| Mean (what the report uses) | 302.9 s | 1.65x |
| Median | 321.8 s | 1.75x |
| Slowest repeat | 355.1 s | 1.93x |

The spread is not measurement error. It is the same DESKTOP instability from
section 3: the fastest baseline repeat ran at 28.7 img/s and the slowest at 15.3.
And the two-machine equal arm ran while that machine was in its fast state at
~29.9 img/s, closest to the *fastest* baseline repeat.

So the like-for-like comparison is nearer **1.26x than 1.65x**. Report the
conservative figure, or better, rerun with the arms interleaved so the question
does not arise. A reviewer will find this in thirty seconds if it is not
addressed, because the standard deviation is printed right next to the mean.

---

## 6. The network axis is currently inert

Every trial recorded `mean_latency_ms` of 0.00 or 0.01, for both nodes, in all 36
trials across both sweeps. Ten microseconds is not a wireless link, and the
sweeps were labelled `lab-wifi`.

The latency probe is not measuring the path it claims to. Since "change the
network between legs" is the next axis on the list, this needs fixing first, or
leg 2 produces a network comparison in which the network column reads zero
everywhere.

Communication cost itself was measured and looks sane: 5.7% to 7.1% of worker
time, roughly flat as machines are added, with total bytes rising from 221 MB
(1 machine, 100 images) to 713 MB (2 machines, 1000 images). On this link
communication is not the bottleneck. That is a finding with a short shelf life —
it is exactly what changes when the network gets worse, which is the point of the
next leg.

---

## 7. Discard sweep A entirely

Sweep `8c5f4e773071` used COCO8, which has **4 training and 4 validation images**.
Both the "100 images" and "1000 images" conditions clamped to the same 4 images,
so the dataset-size axis does not exist in that sweep. Every row trained on 4
images.

It is worse than that. In every round of every two-machine trial, the JSON
records `"workers": 1`. The second machine's share, 2 or 3 samples, fell below the
8-sample floor and was dropped, which the dashboard reported as:

```
c8a228ae — shard of 3 samples is below the 8 sample floor
```

So the two-machine rows are single-machine runs with a spectator attached, and
COE-2's recorded throughput is 0.0 in all of them. The scaling comparison,
the partitioning ablation and the dataset-size axis are all void.

The mAP of 0.87 is memorisation of four images that appear in both the training
and validation split, not accuracy. And figure 5 from that sweep has a y-axis
reading `1e-7+1.17144077e2`, which is matplotlib's way of saying every value is
identical to within seven bytes.

Nothing in sweep A is salvageable. It is a useful negative control for the
harness — the pipeline ran 18 trials correctly on a degenerate input — and
nothing more. Do not cite it.

**Guard to add:** refuse to start a sweep when the requested dataset size exceeds
the parent dataset's training count, or at minimum record the clamped size rather
than the requested one, so a row that says 1000 never means 4.

---

## Flukes and one-off anomalies

Things that are individually explainable but should be known before anyone reads
the tables.

1. **Trial 13, round 4.** DESKTOP finished 431 images in 16.9 s while COE-2 took
   38.4 s for 569. Imbalance 0.389, straggler gap 21.5 s, against ~0.02 in the
   neighbouring rounds. This is DESKTOP flipping into its fast state mid-trial.
   It is the single worst round in the sweep and it is a machine-state artefact.

2. **Trial 14 is the fast outlier** at 156.8 s against 211.6 and 206.1 for its
   sibling repeats. It is what pulls the proportional mean down to 191.5 s and
   inflates its standard deviation to 30.2. Same cause.

3. **Trial 4 reports `dropped_shards: 1` in rounds 1 and 2 with only one worker
   present.** A dropped shard in a single-worker round should not be possible.
   Worth a look at the drop accounting. That same trial also jumped from 31.6 s in
   round 0 to 67–73 s for the rest.

4. **Cold-start predictions are meaningless for the first two rounds.** Round 0
   of trial 0 predicted 269 s for a shard that took 23.5 s; round 0 of trial 6
   predicted 3.3 s for one that took 19.4 s. The EWMA needs about two rounds to
   become useful, so with 5 rounds per trial **40% of every trial runs on a bad
   estimate**, and that tax falls entirely on the proportional arm. Either raise
   rounds per trial to 10 or more, or carry throughput estimates across trials
   within a sweep.

5. **Repeats are genuinely independent now.** Per-repeat seeds 1000/1017/1034
   produced distinct trajectories throughout. The bit-identical-repeats problem
   from earlier is fixed and stayed fixed.

6. **The GPU requirements failure predates these sweeps.** The dashboard capture
   showing DESKTOP-RV928A7 admitted with no measured throughput and 0/12 shards
   failed is the Blackwell wheel mismatch, from before both machines moved to a
   CUDA 12.8 build. Both nodes report torch 2.11.0+cu128 in the sweep JSON, so
   these results are unaffected by it.

---

## Which figures to keep

| Figure | Verdict |
|---|---|
| fig1, scaling | **Keep.** The central result. Add the fitted model as an overlay. |
| fig2, partitioning | **Keep the imbalance panel**, split by dataset size. Drop the wall-clock panel until the arms are interleaved. |
| fig3, accuracy | **Drop.** No signal, and the y-axis spans 0.000–0.030. |
| fig4, time to accuracy | **Drop.** Currently shows accuracy falling with training time. Re-make after section 4 is fixed. |
| fig5, communication | **Keep the African version.** Discard the COCO8 version with the degenerate axis. |
| fig6, dataset scaling | **Keep.** Clean, and it is the visual companion to the overhead-floor argument. |

---

## What to change before leg 2

Ordered by how much they cost you if skipped.

1. **Fix the training schedule.** `warmup_epochs=3.0` against `epochs=1` per
   round. Verify with one single-machine run before committing a whole leg to it.
   Without this the accuracy axis stays empty no matter how many machines join.
2. **Interleave the strategy arms.** Run proportional and equal alternately, or
   randomise trial order, instead of all of one then all of the other. This is
   the difference between a confounded ablation and a clean one, and it costs
   nothing.
3. **Fix the latency probe** before the network leg, or the network column reads
   zero.
4. **Stabilise or characterise DESKTOP-RV928A7.** Its 3x throughput swing is the
   dominant noise source in the whole sweep. The browser, dashboard and other
   applications run on that machine while it trains. Either close them for
   benchmark legs, or keep the operator's machine out of the mesh and run it as
   coordinator only.
5. **Raise repeats to 5 on the baseline cells**, or report the median. Every
   speedup divides by the baseline, so its variance contaminates every number in
   the table.
6. **Size the dataset ladder to the machine count.** For 6–7 machines the top
   rung needs to be 10,000 images. Drop the 10 and 100 rungs to pay for it, or
   keep one small rung purely to measure the overhead floor.
7. **Raise rounds per trial to 10** so the scheduler's two-round warmup is a 20%
   tax instead of 40%.
8. **Guard against dataset clamping** so a sweep can never again report 1000
   images while training on 4.

---

## Environment, for the paper's reproducibility section

| Item | Value |
|---|---|
| Coordinator | Windows 10, Python 3.11.9, 32 CPUs, RTX 5070 |
| Workers | DESKTOP-RV928A7 (Windows 11), COE-2 (Windows 10), both RTX 5070 12 GB |
| torch | 2.11.0+cu128 on both nodes |
| ultralytics | 8.4.46 |
| Model | yolov8n.pt, 640 px, batch ceiling 8 |
| Dataset | African Wildlife, 1052 train / 225 val, 4 classes |

One discrepancy to resolve: `engine/requirements-train-cu128.txt` pins
`torch==2.7.1+cu128`, but both machines ran **2.11.0+cu128**. The paper must
record what actually ran, and the pin should be updated to match so the result is
reproducible from the repository. The ultralytics pin, 8.4.46, does match.

---

## Raw files

Exports from the Testing parameters page, sweep `387af7e617b9`:

```
suite.json      full trial, round and shard detail, plus policy and environment
results.csv     one row per trial
summary.csv     one row per cell, mean and standard deviation over repeats
report.md       generated tables
fig1..fig6.pdf  generated figures
```

Sweep artefacts live under `.gradmesh/`, which is not committed. Keep the
downloaded copies of these two sweeps somewhere durable before leg 2 overwrites
the working directory.
