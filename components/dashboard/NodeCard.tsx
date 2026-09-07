"use client";

import { ago, backendLabel, gflops, memory, seconds } from "@/lib/format";
import type { MeshNode } from "@/lib/types";
import { Meter, TierBadge } from "./ui";

export default function NodeCard({
  node,
  onEvict,
  canEvict = false,
}: {
  node: MeshNode;
  onEvict?: (nodeId: string) => void;
  canEvict?: boolean;
}) {
  const training = node.active_batches > 0;
  const epochProgress =
    node.training_total_epochs > 0 ? node.training_epoch / node.training_total_epochs : 0;

  return (
    <article
      className={`node-card${training ? " is-training" : ""}${node.active ? "" : " is-offline"}`}
    >
      <div className="node-head">
        <div className="stack-sm" style={{ gap: 4, minWidth: 0 }}>
          <div className="row" style={{ gap: 8 }}>
            <span
              className={`dot${training ? " dot-live" : ""}`}
              style={
                training
                  ? undefined
                  : { background: node.active ? "var(--cyan)" : "var(--text-faint)" }
              }
            />
            <span className="node-name truncate">{node.display_name || node.node_id}</span>
          </div>
          <span className="small faint truncate" title={node.gpu}>
            {node.gpu}
          </span>
        </div>
        <TierBadge tier={node.tier} title={node.admission_reason} />
      </div>

      <div className="node-facts">
        <div className="node-fact">
          <span>Backend</span>
          <span>{backendLabel(node.backend)}</span>
        </div>
        <div className="node-fact">
          <span>Memory</span>
          <span>{memory(node.gpu_memory_mb)}</span>
        </div>
        <div className="node-fact">
          <span>Measured</span>
          <span>{gflops(node.capability?.gflops)}</span>
        </div>
        <div className="node-fact">
          <span>Throughput</span>
          <span>{node.throughput_sps ? `${node.throughput_sps.toFixed(1)} img/s` : "unmeasured"}</span>
        </div>
        <div className="node-fact">
          <span>Rounds</span>
          <span>
            {node.completed_rounds}
            {node.failed_rounds ? ` / ${node.failed_rounds} failed` : ""}
          </span>
        </div>
        <div className="node-fact">
          <span>Last seen</span>
          <span>{node.active ? "now" : ago(node.last_seen)}</span>
        </div>
      </div>

      <div className="stack-sm" style={{ gap: 6 }}>
        <div className="row-between small faint" style={{ gap: 8 }}>
          <span>Fitness</span>
          <span className="mono">{node.fitness.toFixed(3)}</span>
        </div>
        <Meter value={node.fitness} tone={node.tier === "probation" ? "warn" : "accent"} />
      </div>

      {training && node.training_total_epochs > 0 ? (
        <div className="stack-sm" style={{ gap: 6 }}>
          <div className="row-between small faint">
            <span>Local epoch</span>
            <span className="mono">
              {node.training_epoch} / {node.training_total_epochs}
            </span>
          </div>
          <Meter value={epochProgress} tone="cyan" />
        </div>
      ) : null}

      {node.admission_reason && node.tier !== "full" ? (
        <p className="small faint">{node.admission_reason}</p>
      ) : null}

      <div className="row-between">
        <span className="small faint">
          {node.samples_trained ? `${node.samples_trained} images trained` : "no work yet"}
          {node.seconds_trained ? ` in ${seconds(node.seconds_trained)}` : ""}
        </span>
        {canEvict && onEvict ? (
          <button className="btn btn-ghost btn-sm" type="button" onClick={() => onEvict(node.node_id)}>
            Remove
          </button>
        ) : null}
      </div>
    </article>
  );
}
