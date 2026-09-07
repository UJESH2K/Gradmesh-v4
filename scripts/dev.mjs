/**
 * `npm run dev` — the whole product, one command.
 *
 * Brings up the Python coordinator and the Next.js dashboard together, binds
 * both to every interface so peers on the same Wi-Fi can reach them, installs
 * the training plane in the background, and prints the join command a
 * contributor pastes into their own terminal.
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
  lanAddresses,
  log,
  paint,
  portInUse,
  primaryLanAddress,
  readCoordinatorState,
  spawnBackground,
  venvPython,
  waitForHttp,
} from "./lib/env.mjs";

const NEXT_BIN = path.join(REPO_ROOT, "node_modules", "next", "dist", "bin", "next");

const children = [];
let shuttingDown = false;

function track(child, name) {
  children.push({ child, name });
  child.on("exit", (code) => {
    if (shuttingDown) return;
    log("gradmesh", `${name} exited with code ${code}, shutting down`, "red");
    shutdown(code ?? 1);
  });
  return child;
}

function shutdown(code = 0) {
  if (shuttingDown) return;
  shuttingDown = true;
  for (const { child } of children) {
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

process.on("SIGINT", () => {
  console.log("");
  log("gradmesh", "stopping the mesh", "yellow");
  shutdown(0);
});
process.on("SIGTERM", () => shutdown(0));

function requireNodeModules() {
  if (existsSync(path.join(REPO_ROOT, "node_modules", "next"))) return;
  console.error(
    paint("red", "\nDependencies are not installed yet. Run:\n\n  npm install\n\nthen `npm run dev` again.\n")
  );
  process.exit(1);
}

/** Noise filter: uvicorn logs one line per heartbeat, which is every worker every few seconds. */
function isInterestingCoordinatorLine(line) {
  if (/"(POST|GET) \/(heartbeat|get_batch)/.test(line)) return false;
  if (/GET \/health/.test(line)) return false;
  return true;
}

async function requirePorts() {
  const conflicts = [];
  if (await portInUse(COORDINATOR_PORT)) conflicts.push(COORDINATOR_PORT);
  if (await portInUse(WEB_PORT)) conflicts.push(WEB_PORT);
  if (conflicts.length === 0) return;

  const list = conflicts.join(" and ");
  console.error(
    paint(
      "red",
      `\nPort ${list} is already in use, most likely by a GradMesh that is still running.\n`
    )
  );
  console.error(
    paint("yellow", "Close the other terminal, or free the ports:\n") +
      (IS_WINDOWS
        ? `  Get-NetTCPConnection -LocalPort ${conflicts.join(",")} -State Listen | ` +
          "Stop-Process -Id { $_.OwningProcess } -Force\n"
        : `  lsof -ti tcp:${conflicts.join(",")} | xargs kill\n`) +
      `\nOr run on different ports:  PORT=3100 GRADMESH_COORDINATOR_PORT=8100 npm run dev\n`
  );
  process.exit(1);
}

async function main() {
  requireNodeModules();
  await requirePorts();

  banner([
    paint("bold", "GradMesh 4"),
    paint("gray", "Every GPU on your network, one training cluster."),
  ]);

  await setupControlPlane();
  await downloadModels();

  const python = venvPython();
  log("coordinator", `starting on 0.0.0.0:${COORDINATOR_PORT}`);
  track(
    spawnBackground(
      python,
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
        filter: isInterestingCoordinatorLine,
        env: { PYTHONUNBUFFERED: "1", GRADMESH_STATE_DIR: path.join(REPO_ROOT, ".gradmesh") },
      }
    ),
    "coordinator"
  );

  const health = await waitForHttp(`http://127.0.0.1:${COORDINATOR_PORT}/health`, {
    timeoutMs: 60000,
  });
  log("coordinator", "ready", "green");

  // The coordinator claims gradmesh.local over multicast DNS, which is what
  // lets a peer open the dashboard without anyone reading an IP off a screen.
  const mdns = health?.mdns?.active ? health.mdns.hostname : null;
  if (!mdns && health?.mdns?.error) {
    log("gradmesh", `multicast DNS unavailable: ${health.mdns.error}`, "yellow");
  }

  // The dashboard does not need torch, so this runs unattended while the UI is
  // already usable. Runs are blocked with a clear message until it finishes.
  const setupState = readSetupState();
  if (setupState.trainingPlane !== "ready") {
    log("setup", "installing the training plane in the background", "yellow");
    setupTrainingPlane({ quiet: true }).catch(() => {});
  }

  log("web", `starting on 0.0.0.0:${WEB_PORT}`);
  track(
    spawnBackground(
      process.execPath,
      // Resolving Next's entry point directly avoids a shell, which on Windows
      // is both a deprecation warning and an argument-escaping hazard.
      [NEXT_BIN, "dev", "--hostname", "0.0.0.0", "--port", String(WEB_PORT)],
      {
        cwd: REPO_ROOT,
        label: "web",
        color: "cyan",
        env: {
          GRADMESH_COORDINATOR_URL: `http://127.0.0.1:${COORDINATOR_PORT}`,
          NEXT_TELEMETRY_DISABLED: "1",
        },
        filter: (line) => !/^\s*[-‐]\s*(Local|Network):/.test(line),
      }
    ),
    "web"
  );

  await waitForHttp(`http://127.0.0.1:${WEB_PORT}/api/health`, { timeoutMs: 180000 });

  const lan = primaryLanAddress();
  const token = readCoordinatorState()?.mesh_token || "";
  const joinUrl = `http://${lan}:${WEB_PORT}/join`;

  const lines = [
    `${paint("green", "●")} ${paint("bold", "Mesh is live")}`,
    "",
    `  ${"Dashboard".padEnd(15)} ${paint("cyan", `http://localhost:${WEB_PORT}`)}`,
  ];

  if (mdns) {
    lines.push(
      `  ${"Other devices".padEnd(15)} ${paint("cyan", `http://${mdns}:${WEB_PORT}`)}  ${paint("gray", "no IP needed")}`
    );
  }
  lines.push(
    `  ${"On this Wi-Fi".padEnd(15)} ${paint("cyan", `http://${lan}:${WEB_PORT}`)}`,
    `  ${"Join page".padEnd(15)} ${paint("cyan", joinUrl)}`,
    "",
    paint("gray", "  Anyone on this network pastes one line to lend their GPU:"),
    `  ${paint("yellow", `irm http://${lan}:${WEB_PORT}/join.ps1 | iex`.padEnd(52))} ${paint("gray", "Windows")}`,
    `  ${paint("yellow", `curl -fsSL http://${lan}:${WEB_PORT}/join.sh | sh`.padEnd(52))} ${paint("gray", "macOS, Linux")}`
  );

  console.log("");
  banner(lines);

  const others = lanAddresses().slice(1);
  if (others.length) {
    log(
      "gradmesh",
      `other interfaces: ${others.map((entry) => `${entry.address} (${entry.name})`).join(", ")}`,
      "gray"
    );
  }
  if (!token) {
    log("gradmesh", "no mesh token yet, the coordinator will mint one on first use", "yellow");
  }
  console.log("");
  log("gradmesh", "press Ctrl+C to stop", "gray");
}

main().catch((error) => {
  console.error(paint("red", `\n${error.message}\n`));
  shutdown(1);
});
