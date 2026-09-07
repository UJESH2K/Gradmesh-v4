/**
 * `npm run coordinator` - run only the Python control plane.
 *
 * Useful when the dashboard is served from somewhere else, or when debugging
 * the scheduler against the FastAPI docs at /docs.
 */

import path from "node:path";

import { setupControlPlane, downloadModels } from "./bootstrap.mjs";
import { COORDINATOR_PORT, ENGINE_DIR, REPO_ROOT, log, paint, run, venvPython } from "./lib/env.mjs";

async function main() {
  await setupControlPlane();
  await downloadModels();
  log("coordinator", `listening on 0.0.0.0:${COORDINATOR_PORT}`);
  await run(
    venvPython(),
    ["-m", "uvicorn", "coordinator.app:app", "--host", "0.0.0.0", "--port", String(COORDINATOR_PORT)],
    {
      cwd: ENGINE_DIR,
      env: { PYTHONUNBUFFERED: "1", GRADMESH_STATE_DIR: path.join(REPO_ROOT, ".gradmesh") },
      allowFailure: true,
    }
  );
}

main().catch((error) => {
  console.error(paint("red", error.message));
  process.exit(1);
});
