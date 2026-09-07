"use client";

import { useCallback, useEffect, useState } from "react";

type Status = { running: boolean; startedAt: number | null; log: string[]; trainingPlane: string };

/**
 * Lets the host contribute its own GPU from the dashboard.
 *
 * The dashboard runs on the host machine, so it can start the agent process
 * directly. That is the difference between needing a second laptop to see the
 * mesh work and seeing it work on the machine already in front of you.
 */
export default function LocalWorkerToggle() {
  const [status, setStatus] = useState<Status | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const response = await fetch("/api/local-worker");
      if (response.ok) setStatus(await response.json());
    } catch {
      // The dashboard is still useful without this control.
    }
  }, []);

  useEffect(() => {
    void load();
    const timer = setInterval(load, 5000);
    return () => clearInterval(timer);
  }, [load]);

  async function toggle() {
    setBusy(true);
    setError(null);
    try {
      const response = await fetch("/api/local-worker", {
        method: status?.running ? "DELETE" : "POST",
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload?.detail || "Could not change the local worker.");
      await load();
    } catch (cause) {
      setError((cause as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const installing = status?.trainingPlane === "installing";

  return (
    <div className="row" style={{ gap: 8 }}>
      {error ? <span className="small" style={{ color: "var(--danger)" }}>{error}</span> : null}
      <button
        type="button"
        className={`btn btn-sm${status?.running ? " btn-danger" : ""}`}
        onClick={toggle}
        disabled={busy || installing}
        title={
          installing
            ? "PyTorch is still installing on this machine"
            : status?.running
              ? "Stop contributing this machine's GPU"
              : "Contribute this machine's GPU to the mesh"
        }
      >
        {busy
          ? "Working…"
          : installing
            ? "Installing PyTorch…"
            : status?.running
              ? "Stop this machine"
              : "Use this machine"}
      </button>
    </div>
  );
}
