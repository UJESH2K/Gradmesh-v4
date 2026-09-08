"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { useMesh } from "./MeshProvider";
import { seconds } from "@/lib/format";

/**
 * The always-present handle on whatever is training.
 *
 * Pinned bottom right on every dashboard page. It answers the one question the
 * old build could not answer from anywhere except the run page: is anything
 * running, and how far along is it. Clicking it goes straight to the live view.
 *
 * It hides itself on the training page, where it would be pointing at the page
 * you are already on.
 */
export default function TrainingDock() {
  const { mesh } = useMesh();
  const pathname = usePathname();
  const [dismissed, setDismissed] = useState(false);

  const run = mesh?.active_runs?.[0] ?? null;

  // A new run should always bring the dock back, even if the last one was
  // dismissed, otherwise "dismiss" quietly means "never show me this again".
  useEffect(() => {
    setDismissed(false);
  }, [run?.id]);

  if (!run || dismissed) return null;
  if (pathname.startsWith("/dashboard/training")) return null;

  const progress = run.rounds > 0 ? run.current_round / run.rounds : 0;
  const waiting = run.status === "waiting";

  return (
    <div className={`dock${waiting ? " is-waiting" : ""}`}>
      <Link href="/dashboard/training" className="dock-body">
        <span className="dock-ring" style={{ ["--dock-progress" as string]: `${progress * 360}deg` }}>
          <span className="dock-ring-value">{Math.round(progress * 100)}</span>
        </span>

        <span className="dock-text">
          <span className="dock-title truncate">{run.name}</span>
          <span className="dock-sub truncate">
            {waiting
              ? "Waiting for a machine"
              : `Round ${Math.min(run.current_round + 1, run.rounds)} of ${run.rounds} · ${run.peak_workers || 0} machine${run.peak_workers === 1 ? "" : "s"}`}
          </span>
          <span className="dock-meter">
            <span className="dock-meter-fill" style={{ width: `${progress * 100}%` }} />
          </span>
        </span>
      </Link>

      <div className="dock-side">
        <span className="dock-elapsed mono">{seconds(run.wall_clock_seconds)}</span>
        <button
          type="button"
          className="dock-close"
          onClick={() => setDismissed(true)}
          aria-label="Hide the training indicator"
          title="Hide until the next run"
        >
          ×
        </button>
      </div>
    </div>
  );
}
