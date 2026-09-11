/**
 * `npm run report <suite-id>` — figures and tables from a finished sweep.
 *
 * Thin wrapper so nobody has to remember where the virtualenv lives. With no
 * argument it builds the most recent sweep, which is what you want right after
 * one finishes.
 */

import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";

import { ENGINE_DIR, STATE_DIR, log, paint, run, venvPython, venvReady } from "./lib/env.mjs";

function latestSuiteId() {
  const root = path.join(STATE_DIR, "benchmarks");
  if (!existsSync(root)) return null;
  const suites = readdirSync(root)
    .map((id) => ({ id, file: path.join(root, id, "suite.json") }))
    .filter((entry) => existsSync(entry.file))
    .map((entry) => {
      let createdAt = statSync(entry.file).mtimeMs;
      try {
        createdAt = (JSON.parse(readFileSync(entry.file, "utf8")).created_at ?? 0) * 1000 || createdAt;
      } catch {
        // Fall back to the file timestamp.
      }
      return { ...entry, createdAt };
    })
    .sort((a, b) => b.createdAt - a.createdAt);
  return suites[0]?.id ?? null;
}

async function main() {
  if (!venvReady()) {
    console.error(paint("red", "No Python environment. Run `npm run setup` first."));
    process.exit(1);
  }

  const requested = process.argv.slice(2).filter((arg) => !arg.startsWith("--"));
  const suite = requested[0] || latestSuiteId();
  if (!suite) {
    console.error(paint("yellow", "No sweeps found. Run one from the Testing lab first."));
    process.exit(1);
  }

  log("report", `building figures for sweep ${suite}`);
  const extra = process.argv.slice(2).filter((arg) => arg.startsWith("--"));
  await run(venvPython(), ["report.py", suite, ...extra], {
    cwd: ENGINE_DIR,
    env: { PYTHONUNBUFFERED: "1", GRADMESH_STATE_DIR: STATE_DIR },
  });
}

main().catch((error) => {
  console.error(paint("red", error.message));
  process.exit(1);
});
