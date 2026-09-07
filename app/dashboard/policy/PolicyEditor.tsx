"use client";

import { useEffect, useState } from "react";

import { useMesh } from "@/components/dashboard/MeshProvider";
import { Empty, Meter, Panel } from "@/components/dashboard/ui";

type Group = {
  title: string;
  blurb: string;
  fields: { key: string; label: string; help: string; min: number; max: number; step: number }[];
};

/**
 * The scheduler's policy, exposed rather than buried.
 *
 * Every number here changes a decision the coordinator makes every round, and
 * the preview underneath re-plans live as you move them, so the effect of a
 * change is visible before it is saved.
 */
const GROUPS: Group[] = [
  {
    title: "Fitness weights",
    blurb:
      "The scheduling score is a weighted sum of measured compute, free memory, heartbeat freshness and learned reliability, minus control-plane latency.",
    fields: [
      { key: "w_compute", label: "Compute", help: "Measured throughput relative to the strongest device", min: 0, max: 1, step: 0.05 },
      { key: "w_memory", label: "Memory", help: "Free device memory relative to the largest device", min: 0, max: 1, step: 0.05 },
      { key: "w_health", label: "Health", help: "How recently the machine sent a heartbeat", min: 0, max: 1, step: 0.05 },
      { key: "w_reliability", label: "Reliability", help: "Learned from finished versus failed rounds", min: 0, max: 1, step: 0.05 },
      { key: "w_latency", label: "Latency penalty", help: "Subtracted for slow control-plane round trips", min: 0, max: 1, step: 0.05 },
    ],
  },
  {
    title: "Admission",
    blurb: "What a machine has to clear before it may hold a shard at all.",
    fields: [
      { key: "min_gflops", label: "Minimum measured GFLOP/s", help: "Below this a device is capped to a micro shard", min: 0, max: 500, step: 5 },
      { key: "min_memory_mb", label: "Minimum device memory (MB)", help: "Below this YOLO will not fit at all", min: 512, max: 16384, step: 256 },
      { key: "probation_shard_cap", label: "Probation shard cap", help: "Largest fraction of the dataset a probation machine may take", min: 0.02, max: 0.5, step: 0.01 },
    ],
  },
  {
    title: "Sharding",
    blurb: "How aggressively work is skewed toward the strongest machines.",
    fields: [
      { key: "max_shard_skew", label: "Maximum fastest to slowest ratio", help: "Caps the skew so weak machines still contribute meaningful gradients", min: 1, max: 40, step: 0.5 },
      { key: "min_shard_samples", label: "Minimum shard size", help: "Smaller shards cost more to ship than to run", min: 1, max: 200, step: 1 },
    ],
  },
  {
    title: "Stragglers",
    blurb: "When a slow machine gets helped, and when the round moves on without it.",
    fields: [
      { key: "soft_deadline_factor", label: "Speculation threshold", help: "Multiple of the prediction that triggers a duplicate on an idle machine", min: 1, max: 6, step: 0.1 },
      { key: "hard_deadline_factor", label: "Drop threshold", help: "Multiple of the prediction after which the shard is abandoned", min: 1.5, max: 12, step: 0.25 },
      { key: "min_deadline_seconds", label: "Grace period (s)", help: "Never chase a machine that has only just started", min: 10, max: 600, step: 5 },
      { key: "cold_start_grace_seconds", label: "First-round grace (s)", help: "A machine's first round also pays checkpoint transfer and CUDA warmup, which the throughput model cannot see", min: 30, max: 900, step: 15 },
      { key: "max_dropped_fraction", label: "Maximum data loss per round", help: "Above this the round is aborted rather than aggregated", min: 0.05, max: 0.8, step: 0.01 },
    ],
  },
  {
    title: "Learning",
    blurb: "How fast the coordinator updates its model of each machine.",
    fields: [
      { key: "throughput_ewma_alpha", label: "Throughput smoothing", help: "Higher reacts faster to change, lower is steadier", min: 0.05, max: 1, step: 0.05 },
      { key: "reliability_reward", label: "Reliability gain per success", help: "Additive increase after a clean round", min: 0.01, max: 0.3, step: 0.01 },
      { key: "reliability_penalty", label: "Reliability loss per failure", help: "Multiplicative decrease after a failed round", min: 0.05, max: 0.9, step: 0.05 },
    ],
  },
];

export default function PolicyEditor({ canManage }: { canManage: boolean }) {
  const { mesh, request, refresh } = useMesh();
  const [values, setValues] = useState<Record<string, number> | null>(null);
  const [defaults, setDefaults] = useState<Record<string, number> | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    request<{ policy: Record<string, number>; defaults: Record<string, number> }>("/api/mesh/policy")
      .then((payload) => {
        setValues(payload.policy);
        setDefaults(payload.defaults);
      })
      .catch((cause) => setError((cause as Error).message));
  }, [request]);

  async function save() {
    if (!values) return;
    setSaving(true);
    setError(null);
    try {
      await request("/api/mesh/policy", { method: "PUT", body: JSON.stringify({ values }) });
      await refresh();
      setSaved(true);
      setTimeout(() => setSaved(false), 2200);
    } catch (cause) {
      setError((cause as Error).message);
    } finally {
      setSaving(false);
    }
  }

  function reset() {
    if (defaults) setValues({ ...defaults });
  }

  if (!values) {
    return (
      <div className="panel">
        <Empty>{error || "Loading policy…"}</Empty>
      </div>
    );
  }

  const dirty = defaults
    ? Object.keys(values).some((key) => Math.abs(values[key] - defaults[key]) > 1e-9)
    : false;
  const plan = mesh?.plan_preview;

  return (
    <>
      <div className="row-between">
        <div>
          <h1 className="page-title">Scheduler</h1>
          <p className="small faint" style={{ marginTop: 4 }}>
            These values drive admission, shard sizing, straggler handling and aggregation weight on
            every round.
          </p>
        </div>
        {canManage ? (
          <div className="row" style={{ gap: 8 }}>
            {dirty ? (
              <button className="btn btn-sm" type="button" onClick={reset}>
                Reset to defaults
              </button>
            ) : null}
            <button className="btn btn-primary btn-sm" type="button" onClick={save} disabled={saving}>
              {saving ? "Saving…" : saved ? "Saved" : "Save policy"}
            </button>
          </div>
        ) : (
          <span className="badge">Read only</span>
        )}
      </div>

      {error ? <div className="notice notice-danger">{error}</div> : null}

      {plan ? (
        <Panel title="Live effect on the next round">
          {plan.assignments.length === 0 ? (
            <Empty>No eligible machines, so there is nothing to plan yet.</Empty>
          ) : (
            <>
              <div className="row wrap" style={{ gap: 22, marginBottom: 18 }}>
                <span className="small faint">
                  Predicted makespan{" "}
                  <strong className="mono accent">{plan.predicted_makespan_seconds}s</strong>
                </span>
                <span className="small faint">
                  Serial estimate <strong className="mono">{plan.predicted_serial_seconds}s</strong>
                </span>
                <span className="small faint">
                  Speedup <strong className="mono accent">{plan.predicted_speedup.toFixed(2)}x</strong>
                </span>
              </div>
              <div className="stack-sm" style={{ gap: 12 }}>
                {plan.assignments.map((assignment) => {
                  const node = mesh?.nodes.find((item) => item.node_id === assignment.node_id);
                  return (
                    <div key={assignment.node_id} className="stack-sm" style={{ gap: 5 }}>
                      <div className="row-between small">
                        <span className="truncate">{node?.display_name || assignment.node_id}</span>
                        <span className="mono faint">
                          {assignment.samples} images · {assignment.predicted_seconds}s
                        </span>
                      </div>
                      <Meter
                        value={plan.total_samples ? assignment.samples / plan.total_samples : 0}
                        tone={assignment.tier === "probation" ? "warn" : "accent"}
                      />
                    </div>
                  );
                })}
              </div>
              {plan.rejected.length > 0 ? (
                <div className="stack-sm" style={{ marginTop: 18 }}>
                  <span className="eyebrow">Excluded by the current policy</span>
                  {plan.rejected.map((item) => (
                    <span key={item.node_id} className="small faint">
                      <span className="mono">{item.node_id.slice(0, 8)}</span> — {item.reason}
                    </span>
                  ))}
                </div>
              ) : null}
              <p className="small faint" style={{ marginTop: 16 }}>
                Save the policy to apply it. The preview above reflects the saved policy, not the
                sliders.
              </p>
            </>
          )}
        </Panel>
      ) : null}

      {GROUPS.map((group) => (
        <Panel key={group.title} title={group.title}>
          <p className="small faint" style={{ marginBottom: 20 }}>
            {group.blurb}
          </p>
          <div className="grid grid-2">
            {group.fields.map((field) => (
              <div className="field" key={field.key}>
                <div className="row-between">
                  <label className="label" htmlFor={field.key}>
                    {field.label}
                  </label>
                  <span className="small mono accent">{values[field.key]}</span>
                </div>
                <input
                  className="range"
                  id={field.key}
                  type="range"
                  min={field.min}
                  max={field.max}
                  step={field.step}
                  value={values[field.key] ?? field.min}
                  disabled={!canManage}
                  onChange={(event) =>
                    setValues({ ...values, [field.key]: Number(event.target.value) })
                  }
                />
                <span className="hint">{field.help}</span>
              </div>
            ))}
          </div>
        </Panel>
      ))}
    </>
  );
}
