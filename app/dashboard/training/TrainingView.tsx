"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { useMesh } from "@/components/dashboard/MeshProvider";
import TrainingRig, { type RigState } from "@/components/dashboard/TrainingRig";
import EventFeed from "@/components/dashboard/EventFeed";
import { Empty, Panel, StatTile } from "@/components/dashboard/ui";
import { compact, percent, seconds } from "@/lib/format";
import type { RunDetail } from "@/lib/types";

/**
 * The live training screen.
 *
 * Progress is reported at two granularities because a round is coarse. Round
 * progress alone sits at "1 of 4" for minutes at a time and looks stuck, so the
 * shards finished inside the current round fill in the gap between rounds.
 */
export default function TrainingView({ canManage }: { canManage: boolean }) {
  const { mesh, request, events, connected, error: meshError } = useMesh();
  const [run, setRun] = useState<RunDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);

  const activeId = mesh?.active_runs?.[0]?.id ?? null;

  const load = useCallback(
    async (id: string) => {
      try {
        setRun(await request<RunDetail>(`/api/mesh/runs/${id}`));
        setError(null);
      } catch (cause) {
        setError((cause as Error).message);
      }
    },
    [request]
  );

  useEffect(() => {
    if (activeId) {
      void load(activeId);
      return;
    }
    // Nothing is running, so fall back to the most recent finished run rather
    // than showing an empty screen.
    request<{ runs: { id: string }[] }>("/api/mesh/runs")
      .then((payload) => {
        const latest = payload.runs?.[0];
        if (latest) void load(latest.id);
        else setRun(null);
      })
      .catch(() => {});
  }, [activeId, load, request, events.length]);

  useEffect(() => {
    const timer = setInterval(() => setTick((value) => value + 1), 1000);
    return () => clearInterval(timer);
  }, []);

  if (error && !run) return <div className="notice notice-danger">{error}</div>;

  // A dead coordinator used to look identical to a run thinking hard: the page
  // simply kept its last state forever. Say so instead.
  const offline = !connected || Boolean(meshError);

  if (!run) {
    return (
      <>
        <h1 className="page-title">Training</h1>
        <div className="panel">
          <div className="grid" style={{ gridTemplateColumns: "minmax(0,1fr) minmax(0,1fr)" }}>
            <TrainingRig state="idle" activity={0} height={240} />
            <div className="stack" style={{ justifyContent: "center" }}>
              <h2>Nothing is training yet.</h2>
              <p className="muted small">
                Upload a dataset, get a machine or two onto the mesh, then start a run. This screen
                follows it from the first shard to the final weights.
              </p>
              <div className="row" style={{ gap: 8 }}>
                <Link className="btn btn-primary btn-sm" href="/dashboard/runs?new=1">
                  Start a run
                </Link>
                <Link className="btn btn-sm" href="/dashboard/discover">
                  Find machines
                </Link>
              </div>
            </div>
          </div>
        </div>
      </>
    );
  }

  const live = ["running", "planning", "waiting"].includes(run.status);
  const shards = run.live_shards || [];
  const finishedShards = shards.filter((shard) => shard.status === "done").length;

  // Round progress plus the fraction of the current round already finished.
  const roundFraction = shards.length > 0 ? finishedShards / shards.length : 0;
  const overall = run.rounds > 0 ? Math.min(1, (run.current_round + roundFraction) / run.rounds) : 0;

  const running = shards.filter((shard) => shard.status === "assigned");
  const rigState: RigState = !live
    ? run.status === "done"
      ? "done"
      : run.status === "failed"
        ? "failed"
        : "idle"
    : running.length > 0
      ? "training"
      : shards.length > 0 && finishedShards === shards.length
        ? "aggregating"
        : "idle";

  // Activity is how much of the mesh is busy right now, which is what should
  // drive how hard the rig looks like it is working.
  const activity = shards.length > 0 ? running.length / shards.length : 0;
  const lastRound = run.round_history?.at(-1);

  return (
    <>
      <div className="row-between">
        <div>
          <div className="row" style={{ gap: 10 }}>
            <h1 className="page-title">{run.name}</h1>
            <span className={`badge ${live ? "badge-live" : "badge-accent"}`}>
              {live ? <span className="dot dot-live" /> : null}
              {run.status}
            </span>
          </div>
          <p className="small faint" style={{ marginTop: 4 }}>
            {run.dataset_name} · {run.base_model} · {compact(run.total_samples)} images ·{" "}
            {run.imgsz}px
          </p>
        </div>
        <div className="row" style={{ gap: 8 }}>
          <Link className="btn btn-sm" href={`/dashboard/runs/${run.id}`}>
            Full detail
          </Link>
          {run.has_artifact ? (
            <a className="btn btn-primary btn-sm" href={`/api/artifact/${run.id}`}>
              Download weights
            </a>
          ) : null}
        </div>
      </div>

      {offline ? (
        <div className="notice notice-danger">
          <strong>The coordinator is not responding.</strong> Anything shown below is the last
          state received, not what is happening now. Check the terminal running{" "}
          <code className="code-inline">npm run dev</code>.
        </div>
      ) : null}
      
      <section className="train-hero">
        <div className="train-rig">
          <TrainingRig state={rigState} activity={activity} height={300} />
        </div>

        <div className="train-progress">
          <span className="eyebrow">Overall progress</span>
          <div className="train-percent mono">{percent(overall, 1)}</div>

          <div className="train-bar">
            <div className="train-bar-fill" style={{ width: `${overall * 100}%` }} />
            {Array.from({ length: Math.max(0, run.rounds - 1) }).map((_, index) => (
              <span
                key={index}
                className="train-bar-tick"
                style={{ left: `${((index + 1) / run.rounds) * 100}%` }}
              />
            ))}
          </div>

          <div className="row-between small faint" style={{ marginTop: 10 }}>
            <span>
              Round {Math.min(run.current_round + 1, run.rounds)} of {run.rounds}
            </span>
            <span data-tick={tick}>
              {live ? `${seconds(run.wall_clock_seconds)} elapsed` : "finished"}
            </span>
          </div>

          <div className="train-state">
            {rigState === "training"
              ? `${running.length} machine${running.length === 1 ? "" : "s"} training right now`
              : rigState === "aggregating"
                ? "Averaging the round's weights"
                : run.status === "waiting"
                  ? "Waiting for an eligible machine to join"
                  : run.status === "done"
                    ? "Finished"
                    : run.status === "failed"
                      ? run.error || "Failed"
                      : "Planning the next round"}
          </div>
        </div>
      </section>

      <div className="grid grid-4">
        <StatTile
          label="Rounds done"
          value={`${run.current_round} / ${run.rounds}`}
          foot="one local epoch each"
        />
        <StatTile
          label="Machines"
          value={run.peak_workers || shards.length || 0}
          foot="contributing to this run"
          accent={running.length > 0}
        />
        <StatTile
          label="Speedup"
          value={run.speedup ? `${run.speedup.toFixed(2)}x` : "—"}
          foot="versus running it serially"
          accent={run.speedup > 1}
        />
        <StatTile
          label="Efficiency"
          value={run.efficiency ? percent(run.efficiency) : "—"}
          foot="speedup per machine"
        />
      </div>

      {shards.length > 0 ? (
        <Panel title="This round, machine by machine">
          <div className="train-shards">
            {shards.map((shard) => {
              const elapsed = shard.elapsed_seconds ?? 0;
              const fraction =
                shard.status === "done"
                  ? 1
                  : Math.min(0.97, elapsed / Math.max(shard.predicted_seconds, 1));
              const tone =
                shard.status === "done"
                  ? "is-done"
                  : shard.status === "dropped" || shard.status === "failed"
                    ? "is-lost"
                    : shard.status === "queued"
                      ? "is-queued"
                      : elapsed > shard.soft_deadline_seconds
                        ? "is-late"
                        : "is-running";

              return (
                <div className="train-shard" key={shard.batch_id}>
                  <div className="row-between" style={{ gap: 10 }}>
                    <span className="truncate small">{shard.node_name || shard.node_id}</span>
                    <span className="small mono faint">
                      {shard.samples} imgs · {shard.status}
                    </span>
                  </div>
                  <div className="shard-track" style={{ marginTop: 7 }}>
                    <div className={`shard-fill ${tone}`} style={{ width: `${fraction * 100}%` }} />
                  </div>
                </div>
              );
            })}
          </div>
        </Panel>
      ) : null}

      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1fr)" }}>
        {lastRound ? (
          <Panel title={`Round ${lastRound.round + 1} result`}>
            <div className="node-facts">
              <div className="node-fact">
                <span>Wall clock</span>
                <span>{seconds(lastRound.wall_clock_seconds)}</span>
              </div>
              <div className="node-fact">
                <span>Slowest machine</span>
                <span>{seconds(lastRound.makespan_seconds)}</span>
              </div>
              <div className="node-fact">
                <span>Straggler gap</span>
                <span>{seconds(lastRound.straggler_gap_seconds)}</span>
              </div>
              <div className="node-fact">
                <span>Aggregation</span>
                <span>{seconds(lastRound.aggregation_seconds)}</span>
              </div>
              <div className="node-fact">
                <span>Images</span>
                <span>{lastRound.samples}</span>
              </div>
              <div className="node-fact">
                <span>Shards lost</span>
                <span>{lastRound.dropped_shards || "none"}</span>
              </div>
            </div>
            <p className="small faint" style={{ marginTop: 14 }}>
              The straggler gap is idle capacity: the difference between the first and last machine
              to finish. It should fall as the scheduler learns each machine&apos;s real speed.
            </p>
          </Panel>
        ) : (
          <Panel title="Round results">
            <Empty>The first round has not finished yet.</Empty>
          </Panel>
        )}

        <Panel title="Activity" flush>
          <EventFeed limit={30} />
        </Panel>
      </div>

      {!canManage ? (
        <p className="small faint">You are viewing as a member, so run controls are hidden.</p>
      ) : null}
    </>
  );
}
