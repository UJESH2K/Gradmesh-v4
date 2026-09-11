"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { useMesh } from "@/components/dashboard/MeshProvider";
import { CampaignComparison, CampaignList, NetworkChangePrompt } from "./CampaignPanel";
import DatasetImporter from "./DatasetImporter";
import { Empty, Panel, StatTile } from "@/components/dashboard/ui";
import { bytes, clock, compact, percent, seconds } from "@/lib/format";
import type {
  BenchmarkIndex,
  Campaign,
  Dataset,
  PartitionStrategy,
  Suite,
  SuiteConfig,
  SuitePreview,
} from "@/lib/types";

/**
 * The experiment lab.
 *
 * A sweep is a factorial design over machine count, dataset size and
 * partitioning strategy. This screen is where it is specified, watched, and
 * exported. It deliberately shows the trial count and the time estimate before
 * anything starts, because the difference between a twenty-minute sweep and an
 * overnight one is four checkboxes and nobody should discover that afterwards.
 */

const DEFAULT_SIZES = [100, 1000];

function defaultConfig(): SuiteConfig {
  return {
    name: "scaling-sweep",
    base_model: "yolov8n.pt",
    parent_dataset_id: null,
    dataset_sizes: DEFAULT_SIZES,
    node_counts: [],
    strategies: ["proportional", "equal"],
    repeats: 3,
    rounds: 5,
    imgsz: 640,
    batch_size: 8,
    node_selection: "strongest",
    evaluate: true,
    network_label: "lab-wifi",
    notes: "",
    trial_timeout_seconds: 3600,
    settle_seconds: 6,
    campaign_id: null,
    leg: 1,
  };
}

export default function TestingLab({ canManage }: { canManage: boolean }) {
  const { request, events } = useMesh();

  const [index, setIndex] = useState<BenchmarkIndex | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [config, setConfig] = useState<SuiteConfig>(defaultConfig);
  const [preview, setPreview] = useState<SuitePreview | null>(null);
  const [suite, setSuite] = useState<Suite | null>(null);
  const [openSuiteId, setOpenSuiteId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [handoff, setHandoff] = useState<Suite | null>(null);

  const patch = (values: Partial<SuiteConfig>) => setConfig((current) => ({ ...current, ...values }));

  const loadIndex = useCallback(async () => {
    try {
      const payload = await request<BenchmarkIndex>("/api/mesh/benchmarks");
      setIndex(payload);
      if (payload.active_suite_id) setOpenSuiteId(payload.active_suite_id);
      else if (!openSuiteId && payload.suites[0]) setOpenSuiteId(payload.suites[0].id);
    } catch (cause) {
      setError((cause as Error).message);
    }
  }, [request, openSuiteId]);

  const loadDatasets = useCallback(
    async (selectNewest = false) => {
      try {
        const payload = await request<{ datasets: Dataset[] }>("/api/mesh/datasets");
        const parents = payload.datasets.filter((item) => !("is_subset" in item && item.is_subset));
        setDatasets(parents);
        setConfig((current) => {
          if (!selectNewest && current.parent_dataset_id) return current;
          const preferred = parents.find((d) => d.is_default) ?? parents[0];
          return preferred ? { ...current, parent_dataset_id: preferred.id } : current;
        });
      } catch {
        // The dataset list is not critical to rendering the page.
      }
    },
    [request]
  );

  useEffect(() => {
    void loadIndex();
    void loadDatasets();
    // Intentionally once on mount; the stream drives refreshes after that.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // A sweep publishes an event per trial, so the screen follows it without polling hard.
  useEffect(() => {
    const latest = events.at(-1);
    if (!latest || !latest.kind.startsWith("suite.")) return;

    void loadIndex();
    if (openSuiteId) void openSuite(openSuiteId);

    // A finished leg is the moment somebody has to go and change the network,
    // so it opens the handoff prompt rather than waiting to be noticed.
    if (latest.kind === "suite.finished" && latest.data.awaiting_network_change) {
      const finishedId = String(latest.data.suite_id);
      setOpenSuiteId(finishedId);
      request<Suite>(`/api/mesh/benchmarks/${finishedId}`)
        .then(setHandoff)
        .catch(() => {});
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [events]);

  const openSuite = useCallback(
    async (id: string) => {
      try {
        setSuite(await request<Suite>(`/api/mesh/benchmarks/${id}`));
        setOpenSuiteId(id);
      } catch (cause) {
        setError((cause as Error).message);
      }
    },
    [request]
  );

  useEffect(() => {
    if (openSuiteId) void openSuite(openSuiteId);
  }, [openSuiteId, openSuite]);

  // Keep a running sweep's view fresh even between trial events.
  useEffect(() => {
    if (!suite?.is_active) return;
    const timer = setInterval(() => {
      if (openSuiteId) void openSuite(openSuiteId);
    }, 8000);
    return () => clearInterval(timer);
  }, [suite?.is_active, openSuiteId, openSuite]);

  // Re-price the sweep whenever the design changes.
  useEffect(() => {
    let cancelled = false;
    const timer = setTimeout(() => {
      request<SuitePreview>("/api/mesh/benchmarks/preview", {
        method: "POST",
        body: JSON.stringify({ config }),
      })
        .then((payload) => !cancelled && setPreview(payload))
        .catch(() => {});
    }, 250);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [config, request]);

  async function abort() {
    if (!openSuiteId) return;
    setBusy(true);
    try {
      await request(`/api/mesh/benchmarks/${openSuiteId}/abort`, { method: "POST" });
    } catch (cause) {
      setError((cause as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function startNextLeg(networkLabel: string) {
    if (!handoff) return;
    setBusy(true);
    setError(null);
    try {
      const created = await request<Suite>(`/api/mesh/benchmarks/${handoff.id}/next-leg`, {
        method: "POST",
        body: JSON.stringify({ network_label: networkLabel }),
      });
      setHandoff(null);
      setOpenSuiteId(created.id);
      await loadIndex();
    } catch (cause) {
      setError((cause as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function resume() {
    if (!openSuiteId) return;
    setBusy(true);
    try {
      await request(`/api/mesh/benchmarks/${openSuiteId}/resume`, { method: "POST" });
      await loadIndex();
    } catch (cause) {
      setError((cause as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const available = index?.available_nodes ?? 0;
  const running = Boolean(index?.active_suite_id);
  const campaigns: Campaign[] = index?.campaigns ?? [];

  /**
   * Re-count the machines, then start.
   *
   * The count is re-read immediately before starting rather than trusted from
   * the last poll, because the gap between opening this page and pressing the
   * button is exactly when somebody plugs in one more laptop.
   */
  async function startTesting() {
    setBusy(true);
    setError(null);
    try {
      const fresh = await request<BenchmarkIndex>("/api/mesh/benchmarks");
      setIndex(fresh);
      if (fresh.available_nodes === 0) {
        setError("No machine is online. Bring workers in from the Discover page first.");
        return;
      }
      const created = await request<Suite>("/api/mesh/benchmarks", {
        method: "POST",
        body: JSON.stringify({ config }),
      });
      setHandoff(null);
      setOpenSuiteId(created.id);
      await loadIndex();
    } catch (cause) {
      setError((cause as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const parent = datasets.find((item) => item.id === config.parent_dataset_id) ?? null;
  const oversized = useMemo(
    () => (parent ? config.dataset_sizes.filter((size) => size > parent.train_count) : []),
    [config.dataset_sizes, parent]
  );

  return (
    <>
      <div className="row-between">
        <div>
          <h1 className="page-title">Testing parameters</h1>
          <p className="small faint" style={{ marginTop: 4 }}>
            Sweep machine count, dataset size and partitioning strategy, repeat each cell, and
            export the numbers as JSON, CSV and figures.
          </p>
        </div>
        <div className="row" style={{ gap: 8 }}>
          {running ? (
            <button className="btn btn-sm btn-danger" onClick={abort} disabled={busy || !canManage}>
              Stop testing
            </button>
          ) : (
            <button
              className="btn btn-primary btn-lg"
              onClick={startTesting}
              disabled={busy || !canManage}
              title={
                available === 0
                  ? "No machine is online yet"
                  : `Detects machines, then runs ${preview?.trials ?? "the"} trials`
              }
            >
              {busy
                ? "Detecting machines…"
                : `Start testing${preview ? ` · ${preview.trials} trials` : ""}`}
            </button>
          )}
        </div>
      </div>

      {handoff ? (
        <NetworkChangePrompt
          suite={handoff}
          onStartNextLeg={startNextLeg}
          busy={busy}
          nodesOnline={available}
        />
      ) : null}

      {error ? <div className="notice notice-danger">{error}</div> : null}
      {!canManage ? (
        <div className="notice notice-warn">
          You are signed in as a member, so you can read results but not start a sweep.
        </div>
      ) : null}
      {index && !index.torch_ready ? (
        <div className="notice notice-warn">
          The training runtime is still installing on this machine. Sweeps cannot start until it
          finishes.
        </div>
      ) : null}

      <div className="grid grid-4">
        <StatTile label="Machines online" value={available} foot="eligible to train" accent={available > 1} />
        <StatTile label="Trials in this design" value={preview?.trials ?? "—"} foot="cells times repeats" />
        <StatTile
          label="Estimated duration"
          value={preview ? seconds(preview.estimated_seconds) : "—"}
          foot="rough, and pessimistic"
        />
        <StatTile
          label="Dataset"
          value={parent ? compact(parent.train_count) : "—"}
          foot={parent ? `${parent.name}, ${parent.val_count} val` : "none selected"}
        />
      </div>

      <div className="lab-grid">
        <Panel title="The design">
          <div className="stack">
            <div className="field">
              <label className="label" htmlFor="sweep-name">
                Sweep name
              </label>
              <input
                id="sweep-name"
                className="input"
                value={config.name}
                onChange={(event) => patch({ name: event.target.value })}
              />
            </div>

            <div className="field">
              <label className="label" htmlFor="sweep-dataset">
                Parent dataset
              </label>
              <select
                id="sweep-dataset"
                className="select"
                value={config.parent_dataset_id ?? ""}
                onChange={(event) => patch({ parent_dataset_id: event.target.value || null })}
              >
                {datasets.length === 0 ? <option value="">Upload a dataset first</option> : null}
                {datasets.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name} · {item.train_count} train / {item.val_count} val
                  </option>
                ))}
              </select>
              <p className="hint">
                Every dataset size below is a reproducible subset of this one. The validation split
                never changes, so a change in accuracy is a change in what was learned rather than
                in what was measured.
              </p>
            </div>

            <div className="field">
              <span className="label">Dataset sizes</span>
              <input
                className="input"
                value={config.dataset_sizes.join(", ")}
                onChange={(event) =>
                  patch({
                    dataset_sizes: event.target.value
                      .split(/[,\s]+/)
                      .map((value) => Number.parseInt(value, 10))
                      .filter((value) => Number.isFinite(value) && value > 0),
                  })
                }
              />
              <p className="hint">
                Training images per trial, comma separated. 100, 1000, 10000 is the usual ladder.
                {oversized.length > 0 ? (
                  <>
                    {" "}
                    <strong>
                      {oversized.join(", ")} exceeds this dataset&apos;s {parent?.train_count} training
                      images, so {oversized.length === 1 ? "that size uses" : "those sizes use"} the
                      whole set.
                    </strong>
                  </>
                ) : null}
              </p>
            </div>

            <div className="field">
              <span className="label">Machine counts</span>
              <div className="chip-row">
                {Array.from({ length: Math.max(available, 1) }).map((_, position) => {
                  const count = position + 1;
                  const on =
                    config.node_counts.length === 0 || config.node_counts.includes(count);
                  return (
                    <button
                      key={count}
                      type="button"
                      className={`chip${on ? " is-on" : ""}`}
                      onClick={() => {
                        const current =
                          config.node_counts.length === 0
                            ? Array.from({ length: available }, (_, i) => i + 1)
                            : config.node_counts;
                        const next = current.includes(count)
                          ? current.filter((value) => value !== count)
                          : [...current, count].sort((a, b) => a - b);
                        patch({ node_counts: next });
                      }}
                    >
                      {count}
                    </button>
                  );
                })}
              </div>
              <p className="hint">
                Which subset of machines a trial uses is chosen by measured capability, so the
                2-machine cell is always the same two machines and node count is the only thing
                that varies.
              </p>
            </div>

            <div className="field">
              <span className="label">Partitioning arms</span>
              <div className="chip-row">
                {(["proportional", "equal"] as PartitionStrategy[]).map((arm) => (
                  <button
                    key={arm}
                    type="button"
                    className={`chip${config.strategies.includes(arm) ? " is-on" : ""}`}
                    onClick={() =>
                      patch({
                        strategies: config.strategies.includes(arm)
                          ? (config.strategies.filter((value) => value !== arm) as PartitionStrategy[])
                          : ([...config.strategies, arm] as PartitionStrategy[]),
                      })
                    }
                  >
                    {arm === "proportional" ? "Capability-proportional" : "Equal shards (control)"}
                  </button>
                ))}
              </div>
              <p className="hint">
                Running both arms is what turns &ldquo;proportional partitioning helps&rdquo; from a
                claim into a measurement. Equal shards are skipped at one machine, where the two are
                identical.
              </p>
            </div>

            <div className="grid grid-3">
              <NumberField
                label="Repeats per cell"
                value={config.repeats}
                min={1}
                max={10}
                onChange={(value) => patch({ repeats: value })}
                hint="3 is the minimum worth reporting"
              />
              <NumberField
                label="Rounds per trial"
                value={config.rounds}
                min={1}
                max={100}
                onChange={(value) => patch({ rounds: value })}
                hint="one local epoch each"
              />
              <NumberField
                label="Image size"
                value={config.imgsz}
                min={64}
                max={1280}
                step={32}
                onChange={(value) => patch({ imgsz: value })}
                hint="640 is the YOLO default"
              />
              <NumberField
                label="Batch ceiling"
                value={config.batch_size}
                min={1}
                max={64}
                onChange={(value) => patch({ batch_size: value })}
                hint="capped per machine by memory"
              />
              <div className="field">
                <label className="label" htmlFor="sweep-selection">
                  Machine selection
                </label>
                <select
                  id="sweep-selection"
                  className="select"
                  value={config.node_selection}
                  onChange={(event) =>
                    patch({ node_selection: event.target.value as SuiteConfig["node_selection"] })
                  }
                >
                  <option value="strongest">Strongest first</option>
                  <option value="random">Random, seeded</option>
                </select>
                <p className="hint">Random samples the hardware space instead</p>
              </div>
              <div className="field">
                <label className="label" htmlFor="sweep-network">
                  Network label
                </label>
                <input
                  id="sweep-network"
                  className="input"
                  value={config.network_label}
                  onChange={(event) => patch({ network_label: event.target.value })}
                />
                <p className="hint">Tags the run, e.g. lab-ethernet or phone-hotspot</p>
              </div>
            </div>

            <label className="row" style={{ gap: 9, cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={config.evaluate}
                onChange={(event) => patch({ evaluate: event.target.checked })}
              />
              <span className="small">
                Score the model after every round. Needed for accuracy and time-to-accuracy, and it
                adds real time per round.
              </span>
            </label>
          </div>
        </Panel>

        <div className="stack">
          <Panel title={`Machines in this sweep (${available})`} flush>
            {available === 0 ? (
              <Empty>No machine is online. Bring workers in from the Discover page first.</Empty>
            ) : (
              <div className="table-scroll">
                <table className="table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Machine</th>
                      <th>Backend</th>
                      <th className="num">Memory</th>
                      <th className="num">GFLOP/s</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(index?.nodes ?? []).map((node, position) => (
                      <tr key={node.node_id}>
                        <td className="mono faint">{position + 1}</td>
                        <td>
                          <div className="truncate">{node.name || node.node_id}</div>
                          <div className="small faint truncate">{node.gpu}</div>
                        </td>
                        <td className="small">{node.backend}</td>
                        <td className="num">{node.memory_mb ? `${Math.round(node.memory_mb / 1024)} GB` : "—"}</td>
                        <td className="num">{node.gflops ? Math.round(node.gflops) : "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <div className="panel-body" style={{ borderTop: "1px solid var(--line)" }}>
              <p className="small faint">
                Ordered strongest first by measured throughput. A trial that needs more machines than
                are online is recorded as skipped rather than silently run smaller.
              </p>
            </div>
          </Panel>

          <DatasetImporter canManage={canManage} onImported={() => void loadDatasets(true)} />

          <CampaignList
            campaigns={campaigns}
            openSuiteId={openSuiteId}
            onOpen={(id) => setOpenSuiteId(id)}
          />

          <Panel title="Past sweeps" flush>
            {(index?.suites ?? []).length === 0 ? (
              <Empty>Nothing has been run yet.</Empty>
            ) : (
              <div className="suite-list">
                {(index?.suites ?? []).map((entry) => (
                  <button
                    key={entry.id}
                    type="button"
                    className={`suite-row${openSuiteId === entry.id ? " is-open" : ""}`}
                    onClick={() => setOpenSuiteId(entry.id)}
                  >
                    <span className="grow" style={{ minWidth: 0 }}>
                      <span className="row" style={{ gap: 8 }}>
                        <strong className="truncate">{entry.name}</strong>
                        <span
                          className={`badge ${
                            entry.status === "running"
                              ? "badge-live"
                              : entry.status === "done"
                                ? "badge-accent"
                                : "badge-warn"
                          }`}
                        >
                          {entry.status}
                        </span>
                      </span>
                      <span className="small faint">
                        {entry.completed_trials}/{entry.total_trials} trials · {entry.network_label} ·{" "}
                        {clock(entry.created_at)}
                      </span>
                    </span>
                  </button>
                ))}
              </div>
            )}
          </Panel>
        </div>
      </div>

      {suite?.config.campaign_id
        ? campaigns
            .filter((campaign) => campaign.campaign_id === suite.config.campaign_id)
            .map((campaign) => (
              <CampaignComparison key={campaign.campaign_id} campaign={campaign} />
            ))
        : null}

      {suite ? <SuiteView suite={suite} onResume={resume} canManage={canManage} busy={busy} /> : null}
    </>
  );
}

function NumberField({
  label,
  value,
  min,
  max,
  step,
  hint,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  hint?: string;
  onChange: (value: number) => void;
}) {
  const id = `field-${label.replace(/\s+/g, "-").toLowerCase()}`;
  return (
    <div className="field">
      <label className="label" htmlFor={id}>
        {label}
      </label>
      <input
        id={id}
        className="input"
        type="number"
        value={value}
        min={min}
        max={max}
        step={step ?? 1}
        onChange={(event) => {
          const parsed = Number.parseInt(event.target.value, 10);
          if (Number.isFinite(parsed)) onChange(Math.min(max, Math.max(min, parsed)));
        }}
      />
      {hint ? <p className="hint">{hint}</p> : null}
    </div>
  );
}

function SuiteView({
  suite,
  onResume,
  canManage,
  busy,
}: {
  suite: Suite;
  onResume: () => void;
  canManage: boolean;
  busy: boolean;
}) {
  const total = suite.trials.length;
  const finished = suite.results.length;
  const completed = suite.results.filter((r) => r.status === "done").length;
  const progress = total > 0 ? finished / total : 0;
  const current = suite.current_trial;

  return (
    <>
      <Panel
        title={`${suite.config.name} · ${suite.id}`}
        action={
          <div className="row" style={{ gap: 8 }}>
            {suite.status === "interrupted" && canManage ? (
              <button className="btn btn-sm" onClick={onResume} disabled={busy}>
                Resume
              </button>
            ) : null}
            <a className="btn btn-sm" href={`/api/benchmark/${suite.id}/suite.json`}>
              JSON
            </a>
            <a className="btn btn-sm" href={`/api/benchmark/${suite.id}/results.csv`}>
              CSV
            </a>
            <a className="btn btn-sm" href={`/api/benchmark/${suite.id}/summary.csv`}>
              Summary
            </a>
          </div>
        }
      >
        <div className="stack">
          <div className="train-bar">
            <div className="train-bar-fill" style={{ width: `${progress * 100}%` }} />
          </div>
          <div className="row-between small">
            <span className="muted">
              {finished} of {total} trials finished, {completed} usable
            </span>
            <span className="faint">
              {suite.is_active && current
                ? `Running: ${current.label}`
                : suite.status === "done"
                  ? `Finished ${clock(suite.finished_at ?? 0)}`
                  : suite.status}
            </span>
          </div>

          {suite.status === "interrupted" ? (
            <div className="notice notice-warn">
              This sweep was interrupted, most likely because the coordinator restarted. Resuming
              continues from the first trial without a result. Finished trials are kept.
            </div>
          ) : null}

          <p className="small faint">
            Figures: run <code className="code-inline">npm run report {suite.id}</code> on this
            machine. It writes six matplotlib figures as PNG and PDF plus a markdown table beside
            the JSON.
          </p>
        </div>
      </Panel>

      {suite.cells.length > 0 ? (
        <Panel title="Results by cell, mean ± standard deviation" flush>
          <div className="table-scroll">
            <table className="table">
              <thead>
                <tr>
                  <th className="num">Machines</th>
                  <th className="num">Images</th>
                  <th>Strategy</th>
                  <th className="num">Runs</th>
                  <th className="num">Train time</th>
                  <th className="num" title="Wall clock on one machine divided by wall clock here, at the same dataset size">
                    Speedup vs 1
                  </th>
                  <th className="num">Efficiency</th>
                  <th className="num">mAP@50</th>
                  <th className="num">Δ mAP</th>
                  <th className="num">Imbalance</th>
                </tr>
              </thead>
              <tbody>
                {suite.cells.map((cell) => (
                  <tr key={`${cell.node_count}-${cell.sample_count}-${cell.strategy}`}>
                    <td className="num">{cell.node_count}</td>
                    <td className="num">{cell.sample_count}</td>
                    <td>
                      <span className={`badge ${cell.strategy === "equal" ? "badge-warn" : "badge-accent"}`}>
                        {cell.strategy}
                      </span>
                    </td>
                    <td className="num">{cell.runs}</td>
                    <td className="num">
                      <Stat mean={cell.train_seconds_mean} std={cell.train_seconds_std} digits={1} />
                    </td>
                    <td className="num">
                      <Stat mean={cell.speedup_mean} std={cell.speedup_std} digits={2} suffix="x" />
                    </td>
                    <td className="num">
                      <Stat mean={cell.efficiency_mean} std={cell.efficiency_std} digits={2} />
                    </td>
                    <td className="num">
                      <Stat mean={cell.map50_mean} std={cell.map50_std} digits={3} />
                    </td>
                    <td className="num">
                      {cell.delta_map50 == null ? (
                        "—"
                      ) : (
                        <span className={cell.delta_map50 < -0.02 ? "" : "accent"}>
                          {cell.delta_map50 > 0 ? "+" : ""}
                          {cell.delta_map50.toFixed(3)}
                        </span>
                      )}
                    </td>
                    <td className="num">
                      {cell.mean_imbalance_mean == null ? "—" : cell.mean_imbalance_mean.toFixed(3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      ) : null}

      <Panel title={`Every trial (${suite.results.length})`} flush>
        {suite.results.length === 0 ? (
          <Empty>No trial has finished yet.</Empty>
        ) : (
          <div className="table-scroll">
            <table className="table">
              <thead>
                <tr>
                  <th>Trial</th>
                  <th>Status</th>
                  <th className="num">Train time</th>
                  <th className="num" title="Wall clock on one machine divided by wall clock here, at the same dataset size">
                    Speedup vs 1
                  </th>
                  <th className="num">mAP@50</th>
                  <th className="num">Imbalance</th>
                  <th className="num">Network share</th>
                  <th className="num">Transferred</th>
                </tr>
              </thead>
              <tbody>
                {[...suite.results].reverse().map((result) => (
                  <tr key={result.trial_id}>
                    <td className="small">{result.label}</td>
                    <td>
                      <span
                        className={`badge ${
                          result.status === "done"
                            ? "badge-accent"
                            : result.status === "skipped"
                              ? "badge-warn"
                              : "badge-danger"
                        }`}
                        title={result.error || undefined}
                      >
                        {result.status}
                      </span>
                    </td>
                    <td className="num">{result.train_seconds ? seconds(result.train_seconds) : "—"}</td>
                    <td className="num">{result.speedup ? `${result.speedup.toFixed(2)}x` : "—"}</td>
                    <td className="num">{result.map50 != null ? result.map50.toFixed(3) : "—"}</td>
                    <td className="num">
                      {result.mean_imbalance != null ? result.mean_imbalance.toFixed(3) : "—"}
                    </td>
                    <td className="num">
                      {result.comm_fraction != null ? percent(result.comm_fraction) : "—"}
                    </td>
                    <td className="num">{result.comm_bytes ? bytes(result.comm_bytes) : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </>
  );
}

function Stat({
  mean,
  std,
  digits,
  suffix = "",
}: {
  mean: number | null;
  std: number | null;
  digits: number;
  suffix?: string;
}) {
  if (mean == null) return <>—</>;
  return (
    <>
      {mean.toFixed(digits)}
      {suffix}
      {std ? <span className="faint small"> ± {std.toFixed(digits)}</span> : null}
    </>
  );
}
