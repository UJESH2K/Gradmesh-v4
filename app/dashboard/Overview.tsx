"use client";

import Link from "next/link";

import EventFeed from "@/components/dashboard/EventFeed";
import NodeCard from "@/components/dashboard/NodeCard";
import { useMesh } from "@/components/dashboard/MeshProvider";
import { Empty, Meter, Panel, StatTile, StatusBadge } from "@/components/dashboard/ui";
import { compact, gflops, memory, seconds } from "@/lib/format";

export default function Overview({ canManage }: { canManage: boolean }) {
  const { mesh, request, refresh } = useMesh();

  if (!mesh) {
    return (
      <div className="panel">
        <Empty>Connecting to the coordinator…</Empty>
      </div>
    );
  }

  const { metrics, plan_preview: plan, nodes, active_runs: activeRuns } = mesh;
  const online = nodes.filter((node) => node.active);
  const admitted = plan.assignments.length;

  async function evict(nodeId: string) {
    await request(`/api/mesh/nodes/${nodeId}`, { method: "DELETE" }).catch(() => {});
    await refresh();
  }

  return (
    <>
      <div className="grid grid-4">
        <StatTile
          label="Machines online"
          value={metrics.nodes_active}
          foot={`${metrics.nodes_admitted} eligible to train`}
          accent={metrics.nodes_active > 0}
        />
        <StatTile
          label="Mesh compute"
          value={gflops(metrics.total_gflops)}
          foot="Measured, not advertised"
        />
        <StatTile label="Device memory" value={memory(metrics.total_memory_mb)} foot="Across the mesh" />
        <StatTile
          label="Predicted speedup"
          value={plan.predicted_speedup ? `${plan.predicted_speedup.toFixed(2)}x` : "—"}
          foot={
            plan.total_samples
              ? `on ${compact(plan.total_samples)} images vs the best single GPU`
              : "upload a dataset to see this"
          }
          accent={plan.predicted_speedup > 1}
        />
      </div>

      {activeRuns.length > 0 ? (
        <Panel
          title="Running now"
          action={
            <Link className="btn btn-sm" href="/dashboard/runs">
              All runs
            </Link>
          }
          flush
        >
          <div className="table-scroll">
            <table className="table">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Status</th>
                  <th>Round</th>
                  <th className="num">Machines</th>
                  <th className="num">Elapsed</th>
                  <th className="num">Speedup</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {activeRuns.map((run) => (
                  <tr key={run.id}>
                    <td>
                      <Link href={`/dashboard/runs/${run.id}`} className="accent">
                        {run.name}
                      </Link>
                      <div className="small faint">{run.dataset_name}</div>
                    </td>
                    <td>
                      <StatusBadge status={run.status} />
                    </td>
                    <td style={{ minWidth: 140 }}>
                      <div className="small mono" style={{ marginBottom: 5 }}>
                        {run.current_round} / {run.rounds}
                      </div>
                      <Meter value={run.rounds ? run.current_round / run.rounds : 0} />
                    </td>
                    <td className="num mono">{run.peak_workers || "—"}</td>
                    <td className="num mono">{seconds(run.wall_clock_seconds)}</td>
                    <td className="num mono accent">{run.speedup ? `${run.speedup}x` : "—"}</td>
                    <td className="num">
                      <Link className="btn btn-sm" href={`/dashboard/runs/${run.id}`}>
                        Open
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      ) : null}

      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1.55fr) minmax(0, 1fr)" }}>
        <div className="stack">
          <Panel
            title={`Machines (${online.length} online)`}
            action={
              <Link className="btn btn-sm" href="/dashboard/nodes">
                Details
              </Link>
            }
          >
            {nodes.length === 0 ? (
              <Empty>
                No machines yet. Open{" "}
                <Link href="/dashboard/invite" className="accent">
                  Invite a GPU
                </Link>{" "}
                and send someone the one-line command, or contribute this machine from the top bar.
              </Empty>
            ) : (
              <div className="grid grid-2">
                {nodes.slice(0, 6).map((node) => (
                  <NodeCard key={node.node_id} node={node} canEvict={canManage} onEvict={evict} />
                ))}
              </div>
            )}
          </Panel>

          {admitted > 0 ? (
            <Panel title="How the next round would be split">
              <p className="small faint" style={{ marginBottom: 16 }}>
                Recomputed continuously from measured throughput. Every bar is sized so all machines
                finish at the same moment.
              </p>
              <div className="stack-sm" style={{ gap: 12 }}>
                {plan.assignments.map((assignment) => {
                  const node = nodes.find((item) => item.node_id === assignment.node_id);
                  const share = plan.total_samples ? assignment.samples / plan.total_samples : 0;
                  return (
                    <div key={assignment.node_id} className="stack-sm" style={{ gap: 5 }}>
                      <div className="row-between small">
                        <span className="truncate">{node?.display_name || assignment.node_id}</span>
                        <span className="mono faint">
                          {assignment.samples} images · {assignment.predicted_seconds}s
                        </span>
                      </div>
                      <Meter value={share} tone={assignment.tier === "probation" ? "warn" : "accent"} />
                    </div>
                  );
                })}
              </div>

              {plan.rejected.length > 0 ? (
                <div className="stack-sm" style={{ marginTop: 20 }}>
                  <span className="eyebrow">Not scheduled</span>
                  {plan.rejected.map((item) => (
                    <div key={item.node_id} className="small faint">
                      <span className="mono">{item.node_id.slice(0, 8)}</span> — {item.reason}
                    </div>
                  ))}
                </div>
              ) : null}
            </Panel>
          ) : null}
        </div>

        <Panel title="Activity" flush>
          <EventFeed />
        </Panel>
      </div>
    </>
  );
}
