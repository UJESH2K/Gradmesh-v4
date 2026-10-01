"use client";

import Link from "next/link";
import { useState } from "react";

/**
 * The runbook, on the page where the work happens.
 *
 * Everything needed to go from an empty mesh to exported figures lives here, so
 * "Testing parameters" in the sidebar is the only thing anybody has to find.
 * Collapsed once a sweep exists, because by then it is reference rather than
 * instruction.
 */
export default function SweepGuide({ hasSweeps }: { hasSweeps: boolean }) {
  const [open, setOpen] = useState(!hasSweeps);
  const [tab, setTab] = useState<"run" | "parameters" | "network" | "output">("run");

  return (
    <section className="guide">
      <button type="button" className="guide-head" onClick={() => setOpen((value) => !value)}>
        <span>
          <strong>How to run an experiment</strong>
          <span className="small faint" style={{ marginLeft: 10 }}>
            machines, datasets, the image ladder, changing networks, exporting results
          </span>
        </span>
        <span className="guide-toggle">{open ? "Hide" : "Show"}</span>
      </button>

      {open ? (
        <div className="guide-body">
          <div className="guide-tabs">
            {(
              [
                ["run", "1. Run one"],
                ["parameters", "2. Every parameter"],
                ["network", "3. Change the network"],
                ["output", "4. Get the results"],
              ] as const
            ).map(([key, label]) => (
              <button
                key={key}
                type="button"
                className={`guide-tab${tab === key ? " is-on" : ""}`}
                onClick={() => setTab(key)}
              >
                {label}
              </button>
            ))}
          </div>

          {tab === "run" ? <RunTab /> : null}
          {tab === "parameters" ? <ParametersTab /> : null}
          {tab === "network" ? <NetworkTab /> : null}
          {tab === "output" ? <OutputTab /> : null}
        </div>
      ) : null}
    </section>
  );
}

function RunTab() {
  return (
    <div className="stack">
      <ol className="guide-steps">
        <li>
          <strong>Get the machines on.</strong> Open{" "}
          <Link href="/dashboard/discover">Discover devices</Link>. Every contributor opens{" "}
          <code className="code-inline">http://gradmesh.local:3000</code> on their machine and pastes
          the one line from the join page. Watch <strong>Machines online</strong> above climb. There
          is no limit, so six or seven is fine.
        </li>
        <li>
          <strong>Get a dataset.</strong> Either import one in <strong>Standard datasets</strong>{" "}
          below, or upload your own YOLO export on <Link href="/dashboard/datasets">Datasets</Link>.
          Pick it in <strong>Parent dataset</strong>.
          <div className="notice notice-warn" style={{ marginTop: 8 }}>
            Use one dataset big enough to subsample, not several. Every size in the ladder is a
            subset of this one parent, and the validation images never change, which is what makes
            the sizes comparable.
          </div>
        </li>
        <li>
          <strong>Set the design.</strong> The defaults are a reasonable first sweep. The two that
          decide how long it takes are <strong>Dataset sizes</strong> and{" "}
          <strong>Repeats per cell</strong>.
        </li>
        <li>
          <strong>Read the estimate before you commit.</strong> <strong>Trials in this design</strong>{" "}
          and <strong>Estimated duration</strong> update as you type. Four checkboxes separate a
          twenty-minute sweep from an overnight one.
        </li>
        <li>
          <strong>Press Start testing.</strong> It re-counts the machines first, then runs every
          trial one at a time. Leave it alone; it writes results to disk after each trial and
          survives a crash or a reboot.
        </li>
      </ol>

      <div className="notice">
        <strong>A trial needing more machines than are online is recorded as skipped</strong>, never
        quietly run smaller. A three-machine cell run on two would silently break the comparison.
      </div>
    </div>
  );
}

function ParametersTab() {
  return (
    <div className="stack">
      <p className="small muted">
        Every control on this page, what it changes, and what to put in it.
      </p>

      <div className="table-scroll">
        <table className="table">
          <thead>
            <tr>
              <th>Parameter</th>
              <th>What it does</th>
              <th>Suggested</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Sweep name</strong></td>
              <td>Label on the results and the figures. Campaign legs inherit it.</td>
              <td><code className="code-inline">scaling-sweep</code></td>
            </tr>
            <tr>
              <td><strong>Parent dataset</strong></td>
              <td>
                The one dataset every trial draws from. Sizes below are reproducible subsets of it.
              </td>
              <td>10,000+ training images</td>
            </tr>
            <tr>
              <td><strong>Dataset sizes</strong></td>
              <td>
                Training images per trial. Each becomes its own row in the results, so this is how
                you answer &ldquo;does the mesh pay off more on bigger jobs&rdquo;.
              </td>
              <td><code className="code-inline">100, 1000, 10000</code></td>
            </tr>
            <tr>
              <td><strong>Machine counts</strong></td>
              <td>
                Which points on the scaling curve to measure. All on means 1, 2, 3 … up to what is
                online.
              </td>
              <td>every count</td>
            </tr>
            <tr>
              <td><strong>Partitioning arms</strong></td>
              <td>
                <strong>Capability-proportional</strong> sizes each shard to the machine.{" "}
                <strong>Equal shards</strong> is the naive control. Running both is what turns the
                core claim into a measurement.
              </td>
              <td>both</td>
            </tr>
            <tr>
              <td><strong>Repeats per cell</strong></td>
              <td>
                How many times each combination runs. Gives the ± in mean ± standard deviation. Each
                repeat uses a different training seed, so they genuinely differ.
              </td>
              <td>3 minimum</td>
            </tr>
            <tr>
              <td><strong>Rounds per trial</strong></td>
              <td>
                One local epoch per round, then weights are averaged. More rounds means higher
                accuracy and a longer sweep.
              </td>
              <td>5 to 10</td>
            </tr>
            <tr>
              <td><strong>Image size</strong></td>
              <td>
                Pixels per side. Dominates both time and memory: doubling it roughly quadruples
                activation memory.
              </td>
              <td>640</td>
            </tr>
            <tr>
              <td><strong>Batch ceiling</strong></td>
              <td>
                An upper bound, not a fixed value. Each machine is given whatever its memory can
                actually hold, so a 4 GB laptop does not crash on a number chosen for a 24 GB card.
              </td>
              <td>8 or 16</td>
            </tr>
            <tr>
              <td><strong>Machine selection</strong></td>
              <td>
                Which machines a smaller cell uses. <strong>Strongest first</strong> keeps the
                2-machine cell the same two machines every time, so node count is the only variable.
                Random samples the hardware space instead, seeded so it reproduces.
              </td>
              <td>strongest first</td>
            </tr>
            <tr>
              <td><strong>Network label</strong></td>
              <td>
                Free text tagging the network these trials ran on. It is what separates lab Ethernet
                from a phone hotspot in one dataset.
              </td>
              <td><code className="code-inline">college-wifi</code></td>
            </tr>
            <tr>
              <td><strong>Score after every round</strong></td>
              <td>
                Measures mAP on the held-out split. Required for accuracy and time-to-accuracy
                columns. Adds real time per round, timed separately so it never inflates training
                time.
              </td>
              <td>on</td>
            </tr>
            <tr>
              <td><strong>Base model</strong></td>
              <td>Starting checkpoint. Use a <code className="code-inline">-obb</code> model only with an oriented-box dataset.</td>
              <td><code className="code-inline">yolov8n.pt</code></td>
            </tr>
            <tr>
              <td><strong>Trial timeout</strong></td>
              <td>Abandons a trial that has clearly hung, so one bad cell does not cost the night.</td>
              <td>3600 s</td>
            </tr>
            <tr>
              <td><strong>Settle time</strong></td>
              <td>
                Pause between trials so GPU memory frees and clocks drop before the next
                measurement.
              </td>
              <td>6 s</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="notice notice-accent">
        <strong>The image ladder.</strong> Powers of ten are the convention because the x-axis is
        logarithmic: <code className="code-inline">100, 1000, 10000</code>. Add{" "}
        <code className="code-inline">10</code> only as a pipeline check, never as a result, because
        below roughly 1000 images the run-to-run noise is larger than the effect being measured. A
        size larger than the parent dataset is clamped to the whole of it, so two sizes above your
        dataset produce two identical rows.
      </div>

      <div className="notice">
        <strong>How long a sweep takes.</strong> Trials ={" "}
        <em>machine counts × dataset sizes × arms × repeats</em>, minus the equal-shard cells at one
        machine, where the two arms are identical. Four machine counts, three sizes, two arms and
        three repeats is 63 trials. At five rounds each that is most of a day.
      </div>
    </div>
  );
}

function NetworkTab() {
  return (
    <div className="stack">
      <p className="small muted">
        To measure whether the network matters, run the <em>same</em> design on each one. That is a
        campaign, and its legs differ only in the network label.
      </p>

      <ol className="guide-steps">
        <li>
          <strong>Run leg 1 normally.</strong> Set <strong>Network label</strong> to the network you
          are on, say <code className="code-inline">college-wifi</code>, and press Start testing.
        </li>
        <li>
          <strong>Wait for the prompt.</strong> When the leg finishes, a green{" "}
          <strong>Change the network now</strong> panel appears at the top of this page. It reports
          what finished and how many machines are back online.
        </li>
        <li>
          <strong>Switch every machine.</strong> Host and workers both. Workers reconnect on their
          own and <code className="code-inline">gradmesh.local</code> follows the host, so usually
          nothing needs retyping. Wait for the machine count in the prompt to return to its old
          value.
        </li>
        <li>
          <strong>Name it and start leg 2.</strong> Type the new label, say{" "}
          <code className="code-inline">phone-hotspot</code>, and press{" "}
          <strong>Start leg 2</strong>. The whole design is copied: same machines, dataset, subsets,
          seeds and matrix. Only the network changes.
        </li>
        <li>
          <strong>Repeat for as many networks as you want.</strong> Each finished leg offers the
          next one.
        </li>
      </ol>

      <div className="notice notice-accent">
        A cross-network comparison table appears once a campaign has two legs, with one row per
        network: observed latency, training time, speedup, mAP, communication share, and the change
        against leg 1.
      </div>

      <div className="notice notice-warn">
        <strong>The prompt refuses a repeated label.</strong> Two legs with the same name cannot be
        told apart in the results. If a machine does not come back, its cells are recorded as
        skipped rather than run smaller.
      </div>

      <p className="small faint">
        This records the network rather than imposing it. To force a specific latency or bandwidth
        you need <code className="code-inline">tc netem</code> on Linux or Clumsy on Windows, both
        requiring administrator rights. Apply the shaping yourself and label the leg accordingly; the
        measured latency in the output shows whether it took effect.
      </p>
    </div>
  );
}

function OutputTab() {
  return (
    <div className="stack">
      <p className="small muted">
        Everything lands in <code className="code-inline">.gradmesh/benchmarks/&lt;sweep-id&gt;/</code>{" "}
        and is downloadable from the sweep panel below.
      </p>

      <div className="table-scroll">
        <table className="table">
          <thead>
            <tr>
              <th>File</th>
              <th>Contains</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><code className="code-inline">suite.json</code></td>
              <td>
                The complete record and the source of truth. Every trial, round, accuracy point, plus
                the hardware and software disclosure block.
              </td>
            </tr>
            <tr>
              <td><code className="code-inline">results.csv</code></td>
              <td>One row per trial, for a spreadsheet.</td>
            </tr>
            <tr>
              <td><code className="code-inline">summary.csv</code></td>
              <td>One row per cell with mean and standard deviation.</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="stack-sm">
        <strong className="small">For the figures, run this on the host:</strong>
        <pre className="code">npm run report</pre>
        <p className="small faint">
          With no argument it builds the most recent sweep. Pass a sweep id for a specific one. It
          writes six figures as PNG and PDF plus a markdown table beside the JSON.
        </p>
      </div>

      <div className="notice">
        <strong>Two speedups, and only one belongs in a paper.</strong>{" "}
        <code className="code-inline">speedup</code> is wall clock on one machine over wall clock
        here, which is exactly 1.0 at one machine. Quote that one.{" "}
        <code className="code-inline">parallel_speedup</code> measures shard overlap inside a round
        and sits below 1 on a single machine because orchestration is not free. It explains a
        disappointing result; it is not the result.
      </div>

      <p className="small faint">
        If a figure and the JSON ever disagree, the JSON is right and the plotting code has a bug.
      </p>
    </div>
  );
}
