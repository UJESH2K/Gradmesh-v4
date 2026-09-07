/**
 * `npm run start` — production mode.
 *
 * Same topology as dev, but serves the compiled Next build. Run `npm run build`
 * first. Everything else, including setup, is identical, so a machine that has
 * only ever run dev can switch to production without extra steps.
 */

import { existsSync } from "node:fs";
import path from "node:path";

import { downloadModels, readSetupState, setupControlPlane, setupTrainingPlane } from "./bootstrap.mjs";
import {
  COORDINATOR_PORT,
  ENGINE_DIR,
  IS_WINDOWS,
  REPO_ROOT,
  WEB_PORT,
  banner,
  log,
  paint,
  primaryLanAddress,
  spawnBackground,
  venvPython,
  waitForHttp,
} from "./lib/env.mjs";

const NEXT_BIN = path.join(REPO_ROOT, "node_modules", "next", "dist", "bin", "next");

const children = [];
let shuttingDown = false;

function shutdown(code = 0) {
  if (shuttingDown) return;
  shuttingDown = true;
  for (const child of children) {
    try {
      if (IS_WINDOWS && child.pid) {
        spawnBackground("taskkill", ["/pid", String(child.pid), "/T", "/F"], { stdio: "ignore" });
      } else {
        child.kill("SIGTERM");
      }
    } catch {
      // already gone
    }
  }
  setTimeout(() => process.exit(code), 400);
}

process.on("SIGINT", () => shutdown(0));
process.on("SIGTERM", () => shutdown(0));

async function main() {
  if (!existsSync(path.join(REPO_ROOT, ".next", "BUILD_ID"))) {
    console.error(paint("red", "\nNo production build found. Run `npm run build` first.\n"));
    process.exit(1);
  }

  await setupControlPlane();
  await downloadModels();

  children.push(
    spawnBackground(
      venvPython(),
      [
        "-m",
        "uvicorn",
        "coordinator.app:app",
        "--host",
        "0.0.0.0",
        "--port",
        String(COORDINATOR_PORT),
        "--log-level",
        "warning",
      ],
      {
        cwd: ENGINE_DIR,
        label: "coordinator",
        color: "magenta",
        env: { PYTHONUNBUFFERED: "1", GRADMESH_STATE_DIR: path.join(REPO_ROOT, ".gradmesh") },
      }
    )
  );

  const health = await waitForHttp(`http://127.0.0.1:${COORDINATOR_PORT}/health`, {
    timeoutMs: 60000,
  });
  log("coordinator", "ready", "green");
  const mdns = health?.mdns?.active ? health.mdns.hostname : null;

  if (readSetupState().trainingPlane !== "ready") {
    setupTrainingPlane({ quiet: true }).catch(() => {});
  }

  children.push(
    spawnBackground(
      process.execPath,
      // Resolving Next's entry point directly avoids a shell, which on Windows
      // is both a deprecation warning and an argument-escaping hazard.
      [NEXT_BIN, "start", "--hostname", "0.0.0.0", "--port", String(WEB_PORT)],
      {
        cwd: REPO_ROOT,
        label: "web",
        color: "cyan",
        env: {
          GRADMESH_COORDINATOR_URL: `http://127.0.0.1:${COORDINATOR_PORT}`,
          NEXT_TELEMETRY_DISABLED: "1",
        },
      }
    )
  );

  await waitForHttp(`http://127.0.0.1:${WEB_PORT}/api/health`, { timeoutMs: 120000 });
  const lan = primaryLanAddress();
  banner([
    `${paint("green", "●")} ${paint("bold", "GradMesh is serving")}`,
    ...(mdns ? [`  Other devices  ${paint("cyan", `http://${mdns}:${WEB_PORT}`)}`] : []),
    `  Dashboard      ${paint("cyan", `http://${lan}:${WEB_PORT}`)}`,
    `  Join page      ${paint("cyan", `http://${lan}:${WEB_PORT}/join`)}`,
  ]);
}

main().catch((error) => {
  console.error(paint("red", error.message));
  shutdown(1);
});
