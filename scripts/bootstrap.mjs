/**
 * One-time environment setup, run automatically by `npm run dev`.
 *
 * v3 asked every participant to create a virtualenv by hand, pick the right
 * requirements file for their GPU vendor, and download a checkpoint before
 * anything worked. That is the single biggest reason a demo failed. This script
 * does all of it, is safe to run repeatedly, and separates the fast control
 * plane from the slow training plane so the dashboard is usable in seconds
 * while torch downloads in the background.
 */

import { createWriteStream, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { rm, stat } from "node:fs/promises";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

import {
  ENGINE_DIR,
  REPO_ROOT,
  STATE_DIR,
  VENV_DIR,
  detectGpuVendor,
  ensureStateDir,
  findSystemPython,
  log,
  paint,
  run,
  venvPython,
  venvReady,
} from "./lib/env.mjs";

const SETUP_FILE = path.join(STATE_DIR, "setup.json");
const MODELS_DIR = path.join(STATE_DIR, "models");

const BASE_MODELS = [
  {
    name: "yolov8n.pt",
    url: "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
  },
  {
    name: "yolo11n.pt",
    url: "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
  },
];

export function readSetupState() {
  if (!existsSync(SETUP_FILE)) {
    return { controlPlane: "pending", trainingPlane: "pending", models: [], backend: null, messages: [] };
  }
  try {
    return JSON.parse(readFileSync(SETUP_FILE, "utf8"));
  } catch {
    return { controlPlane: "pending", trainingPlane: "pending", models: [], backend: null, messages: [] };
  }
}

export function writeSetupState(patch) {
  ensureStateDir();
  const next = { ...readSetupState(), ...patch, updatedAt: Date.now() };
  writeFileSync(SETUP_FILE, JSON.stringify(next, null, 2));
  return next;
}

function note(message) {
  const state = readSetupState();
  const messages = [...(state.messages || []), { at: Date.now(), message }].slice(-40);
  writeSetupState({ messages });
}

async function ensureVenv() {
  if (venvReady()) return;

  const python = findSystemPython();
  if (!python) {
    console.error(
      paint(
        "red",
        "\nGradMesh needs Python 3.9 or newer on PATH.\n" +
          "  Windows: winget install Python.Python.3.12\n" +
          "  macOS:   brew install python@3.12\n" +
          "  Linux:   sudo apt install python3 python3-venv\n"
      )
    );
    process.exit(1);
  }

  log("setup", `creating a virtual environment with Python ${python.version}`);
  await run(python.command, [...python.args, "-m", "venv", VENV_DIR]);
}

async function pipInstall(requirementsFile, label) {
  const python = venvPython();
  log("setup", `installing ${label}`);
  await run(python, ["-m", "pip", "install", "--upgrade", "pip", "--quiet", "--disable-pip-version-check"], {
    allowFailure: true,
  });
  await run(python, [
    "-m",
    "pip",
    "install",
    "--quiet",
    "--disable-pip-version-check",
    "-r",
    path.join(ENGINE_DIR, requirementsFile),
  ]);
}

async function downloadModels() {
  mkdirSync(MODELS_DIR, { recursive: true });
  const present = [];

  for (const model of BASE_MODELS) {
    const target = path.join(MODELS_DIR, model.name);
    if (existsSync(target)) {
      const info = await stat(target);
      if (info.size > 1_000_000) {
        present.push(model.name);
        continue;
      }
      await rm(target, { force: true });
    }

    log("setup", `downloading ${model.name}`);
    try {
      const response = await fetch(model.url, { redirect: "follow" });
      if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
      await pipeline(Readable.fromWeb(response.body), createWriteStream(target));
      present.push(model.name);
    } catch (error) {
      // A checkpoint is only needed when a run starts, so a flaky network at
      // setup time must not stop the dashboard from coming up.
      log("setup", `could not fetch ${model.name}: ${error.message}`, "yellow");
      await rm(target, { force: true });
    }
  }

  writeSetupState({ models: present });
  return present;
}

/** Requirements profile for whatever accelerator this machine actually has. */
export function trainingRequirements() {
  const vendor = detectGpuVendor();
  if (vendor === "cuda") return { file: "requirements-train.txt", label: "PyTorch CUDA build", backend: "cuda" };
  if (vendor === "xpu") return { file: "requirements-xpu.txt", label: "PyTorch Intel XPU build", backend: "xpu" };
  return { file: "requirements-train-cpu.txt", label: "PyTorch CPU build", backend: "cpu" };
}

export async function setupControlPlane() {
  ensureStateDir();
  await ensureVenv();
  writeSetupState({ controlPlane: "installing" });
  await pipInstall("requirements-control.txt", "the coordinator runtime");
  writeSetupState({ controlPlane: "ready" });
  note("Control plane ready.");
}

export async function setupTrainingPlane({ quiet = false } = {}) {
  const profile = trainingRequirements();
  writeSetupState({ trainingPlane: "installing", backend: profile.backend });
  if (!quiet) log("setup", `installing ${profile.label}, this can take a few minutes`);
  note(`Installing ${profile.label}.`);
  try {
    await pipInstall(profile.file, profile.label);
    writeSetupState({ trainingPlane: "ready" });
    note("Training plane ready. Runs can start now.");
    if (!quiet) log("setup", "training plane ready", "green");
  } catch (error) {
    writeSetupState({ trainingPlane: "failed", trainingError: String(error.message || error) });
    note(`Training plane install failed: ${error.message}`);
    if (!quiet) log("setup", `training plane install failed: ${error.message}`, "red");
  }
}

async function main() {
  const args = new Set(process.argv.slice(2));
  await setupControlPlane();
  await downloadModels();

  if (!args.has("--control-only")) {
    await setupTrainingPlane();
  }

  const state = readSetupState();
  log("setup", "done", "green");
  console.log(
    `${paint("gray", "           ")} control plane: ${state.controlPlane}, training plane: ${state.trainingPlane}, models: ${(state.models || []).join(", ") || "none"}`
  );
}

if (import.meta.url === `file://${process.argv[1]}` || process.argv[1]?.endsWith("bootstrap.mjs")) {
  main().catch((error) => {
    console.error(paint("red", `setup failed: ${error.message}`));
    process.exit(1);
  });
}

export { MODELS_DIR, REPO_ROOT, downloadModels };
