/**
 * `npm run doctor` — tell the user exactly why the mesh will not start.
 *
 * Every check prints a fix, not just a status, because the person running this
 * is usually a contributor who did not set the project up.
 */

import { existsSync } from "node:fs";
import path from "node:path";

import { readSetupState } from "./bootstrap.mjs";
import {
  COORDINATOR_PORT,
  REPO_ROOT,
  STATE_DIR,
  WEB_PORT,
  detectGpuVendor,
  findSystemPython,
  lanAddresses,
  paint,
  readCoordinatorState,
  venvPython,
  venvReady,
} from "./lib/env.mjs";

const results = [];

function check(name, ok, detail, fix) {
  results.push({ name, ok, detail, fix });
}

async function reachable(url) {
  try {
    const response = await fetch(url, { signal: AbortSignal.timeout(2000) });
    return response.ok;
  } catch {
    return false;
  }
}

async function main() {
  const nodeMajor = Number(process.versions.node.split(".")[0]);
  check("Node.js", nodeMajor >= 20, `v${process.versions.node}`, "Install Node.js 20 or newer from nodejs.org");

  check(
    "npm dependencies",
    existsSync(path.join(REPO_ROOT, "node_modules", "next")),
    existsSync(path.join(REPO_ROOT, "node_modules")) ? "installed" : "missing",
    "Run: npm install"
  );

  const python = findSystemPython();
  check("Python", Boolean(python), python ? `${python.command} ${python.version}` : "not found", "Install Python 3.11 or 3.12 and reopen the terminal");

  check("Virtual environment", venvReady(), venvReady() ? venvPython() : "missing", "Run: npm run setup");

  const setup = readSetupState();
  check("Coordinator runtime", setup.controlPlane === "ready", setup.controlPlane, "Run: npm run setup");
  check(
    "Training runtime",
    setup.trainingPlane === "ready",
    setup.trainingPlane === "installing" ? "still installing" : setup.trainingPlane,
    "Run: npm run setup. This downloads PyTorch and takes a few minutes."
  );

  const models = setup.models || [];
  check("Base checkpoints", models.length > 0, models.join(", ") || "none", "Run: npm run setup while online");

  const state = readCoordinatorState();
  check("Mesh token", Boolean(state?.mesh_token), state?.mesh_token ? "present" : "not minted yet", "Start the host once with: npm run dev");

  const datasets = Object.keys(state?.datasets || {}).length;
  check("Datasets", datasets > 0, `${datasets} registered`, "Upload a YOLO dataset zip on the Datasets page");

  check("Accelerator", true, detectGpuVendor(), "");

  const addresses = lanAddresses();
  check(
    "Network",
    addresses.length > 0,
    addresses.map((entry) => entry.address).join(", ") || "no LAN address",
    "Connect to Wi-Fi or Ethernet so peers can reach this host"
  );

  check(
    "Coordinator running",
    await reachable(`http://127.0.0.1:${COORDINATOR_PORT}/health`),
    `port ${COORDINATOR_PORT}`,
    "Run: npm run dev"
  );
  check(
    "Dashboard running",
    await reachable(`http://127.0.0.1:${WEB_PORT}/api/health`),
    `port ${WEB_PORT}`,
    "Run: npm run dev"
  );

  console.log("");
  for (const result of results) {
    const mark = result.ok ? paint("green", "  ok  ") : paint("red", " fail ");
    console.log(`${mark} ${result.name.padEnd(22)} ${paint("gray", result.detail || "")}`);
    if (!result.ok && result.fix) {
      console.log(`       ${paint("yellow", result.fix)}`);
    }
  }

  const failures = results.filter((result) => !result.ok).length;
  console.log("");
  console.log(
    failures === 0
      ? paint("green", "Everything checks out.")
      : paint("yellow", `${failures} item${failures === 1 ? "" : "s"} need attention. State lives in ${STATE_DIR}`)
  );
  process.exit(failures === 0 ? 0 : 1);
}

main();
