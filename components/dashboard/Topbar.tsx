"use client";

import Link from "next/link";

import { gflops, memory } from "@/lib/format";
import LocalWorkerToggle from "./LocalWorkerToggle";
import { useMesh } from "./MeshProvider";

export default function Topbar({ role }: { role: string }) {
  const { mesh, error } = useMesh();
  const metrics = mesh?.metrics;

  return (
    <header className="dash-topbar">
      <div className="row wrap" style={{ gap: 18 }}>
        <span className="badge badge-live">
          <span className="dot dot-live" />
          {metrics ? `${metrics.nodes_active} online` : "connecting"}
        </span>
        {metrics ? (
          <>
            <span className="small faint">
              Mesh compute <strong className="mono accent">{gflops(metrics.total_gflops)}</strong>
            </span>
            <span className="small faint">
              Device memory <strong className="mono">{memory(metrics.total_memory_mb)}</strong>
            </span>
            {metrics.active_shards > 0 ? (
              <span className="small faint">
                Shards in flight <strong className="mono accent">{metrics.active_shards}</strong>
              </span>
            ) : null}
          </>
        ) : null}
        {error ? <span className="badge badge-danger">{error}</span> : null}
      </div>

      <div className="row" style={{ gap: 8 }}>
        {role === "owner" ? <LocalWorkerToggle /> : null}
        <Link className="btn btn-sm" href="/dashboard/training">
          Live training
        </Link>
        <Link className="btn btn-sm" href="/dashboard/discover">
          Discover devices
        </Link>
        <Link className="btn btn-sm" href="/dashboard/invite">
          Invite a GPU
        </Link>
        <Link className="btn btn-primary btn-sm" href="/dashboard/runs?new=1">
          New run
        </Link>
      </div>
    </header>
  );
}
