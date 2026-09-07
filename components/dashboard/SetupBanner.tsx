"use client";

import { useMesh } from "./MeshProvider";

/**
 * The dashboard comes up before PyTorch finishes installing, which is the whole
 * point of splitting the two planes. This says so plainly rather than letting
 * someone hit a confusing failure when they try to start a run.
 */
export default function SetupBanner() {
  const { setup, mesh } = useMesh();
  if (!setup) return null;

  if (setup.trainingPlane === "installing") {
    const latest = setup.messages?.at(-1)?.message;
    return (
      <div className="notice notice-warn row" style={{ gap: 12 }}>
        <span className="dot" style={{ background: "var(--warn)" }} />
        <span className="grow">
          Installing the training runtime on this machine. The dashboard works now; runs can start
          once it finishes.
          {latest ? <span className="faint"> {latest}</span> : null}
        </span>
      </div>
    );
  }

  if (setup.trainingPlane === "failed") {
    return (
      <div className="notice notice-danger">
        The training runtime failed to install on this machine, so the coordinator cannot aggregate
        weights. Run <code className="code-inline">npm run setup</code> in the project folder to see
        the error.
      </div>
    );
  }

  if (mesh && !mesh.torch_ready) {
    return (
      <div className="notice notice-warn">
        PyTorch is not importable on the coordinator, so aggregation would fail. Run{" "}
        <code className="code-inline">npm run doctor</code> on the host to diagnose it.
      </div>
    );
  }

  if (mesh && !mesh.default_dataset) {
    return (
      <div className="notice notice-accent">
        No dataset yet. Upload a YOLO export on the Datasets page and it becomes the starting
        dataset for every machine that joins.
      </div>
    );
  }

  return null;
}
