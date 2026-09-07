"use client";

import { useState } from "react";

import CopyLine from "@/components/CopyLine";
import { useMesh } from "@/components/dashboard/MeshProvider";
import { Panel } from "@/components/dashboard/ui";

export default function InviteActions({
  canManage,
  joinUrl,
}: {
  canManage: boolean;
  joinUrl: string;
}) {
  const { request, refresh } = useMesh();
  const [confirming, setConfirming] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function rotate() {
    setError(null);
    try {
      const payload = await request<{ removed_nodes: number }>("/api/mesh/token/rotate", {
        method: "POST",
      });
      setConfirming(false);
      setStatus(
        `New token issued. ${payload.removed_nodes} machine${payload.removed_nodes === 1 ? "" : "s"} were removed and will need the new command.`
      );
      await refresh();
      // The page renders the token server-side, so it has to reload to show it.
      setTimeout(() => window.location.reload(), 1400);
    } catch (cause) {
      setError((cause as Error).message);
    }
  }

  return (
    <>
      <Panel title="Share the page instead">
        <div className="stack">
          <CopyLine value={joinUrl} label="Send this link to anyone on the network" />
          <p className="small faint">
            The join page shows the same commands and a QR code, so someone can scan it from their
            phone and read it on the machine they are setting up.
          </p>
        </div>
      </Panel>

      {canManage ? (
        <Panel title="Access">
          <div className="stack">
            <p className="small muted">
              Anyone who can reach this host on the network can read the join page and get the mesh
              token. That is deliberate: on a LAN, being on the network is the trust boundary.
              Rotate the token to revoke it, which immediately removes every connected machine.
            </p>

            {status ? <div className="notice notice-accent">{status}</div> : null}
            {error ? <div className="notice notice-danger">{error}</div> : null}

            {confirming ? (
              <div className="row" style={{ gap: 8 }}>
                <button className="btn btn-danger btn-sm" type="button" onClick={rotate}>
                  Yes, rotate and disconnect everyone
                </button>
                <button
                  className="btn btn-ghost btn-sm"
                  type="button"
                  onClick={() => setConfirming(false)}
                >
                  Cancel
                </button>
              </div>
            ) : (
              <div>
                <button className="btn btn-sm" type="button" onClick={() => setConfirming(true)}>
                  Rotate the mesh token
                </button>
              </div>
            )}
          </div>
        </Panel>
      ) : null}
    </>
  );
}
