/**
 * `npm run worker` — contribute this machine's GPU to a mesh.
 *
 * With no arguments it joins the coordinator running on this same machine,
 * reading the token straight off disk. Point it at another host with
 * `npm run worker -- --server http://192.168.1.20:8000 --token <token>`.
 */

import {
  COORDINATOR_PORT,
  ENGINE_DIR,
  log,
  paint,
  readCoordinatorState,
  run,
  venvPython,
  venvReady,
} from "./lib/env.mjs";
import { readSetupState, setupControlPlane, setupTrainingPlane } from "./bootstrap.mjs";

function argValue(flag, fallback) {
  const index = process.argv.indexOf(flag);
  return index !== -1 && process.argv[index + 1] ? process.argv[index + 1] : fallback;
}

async function main() {
  if (!venvReady()) await setupControlPlane();

  const setup = readSetupState();
  if (setup.trainingPlane !== "ready") {
    log("worker", "the training plane is not installed yet, installing now", "yellow");
    await setupTrainingPlane();
  }

  const server = argValue("--server", `http://127.0.0.1:${COORDINATOR_PORT}`);
  const token = argValue("--token", readCoordinatorState()?.mesh_token || "");
  const name = argValue("--name", "");
  const backend = argValue("--backend", "auto");

  if (!token) {
    console.error(
      paint("red", "\nNo mesh token. Start the host with `npm run dev`, or pass --token from its Join page.\n")
    );
    process.exit(1);
  }

  const args = ["worker.py", "--server-url", server, "--token", token, "--backend", backend, "--discover"];
  if (name) args.push("--name", name);

  log("worker", `joining ${server}`);
  await run(venvPython(), args, { cwd: ENGINE_DIR, env: { PYTHONUNBUFFERED: "1" }, allowFailure: true });
}

main().catch((error) => {
  console.error(paint("red", error.message));
  process.exit(1);
});
