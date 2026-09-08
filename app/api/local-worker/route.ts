import { spawn, spawnSync } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  openSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import path from "node:path";
import { NextResponse } from "next/server";

import { currentUser } from "@/lib/auth";
import { COORDINATOR_URL, STATE_DIR, meshToken, setupState } from "@/lib/config";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Contribute the host machine's own GPU without opening a second terminal.
 *
 * The agent runs **detached, with its own console and its own process group**,
 * and its output goes to a file rather than back through a pipe. Both of those
 * are load-bearing, and both were learned the hard way.
 *
 * A worker spawned as an ordinary child of the Next dev server died silently
 * about twenty seconds into every round, inside the CUDA init that Ultralytics
 * runs before training. No Python traceback, no exit message: the process was
 * simply gone, which is a native-level termination rather than an error. The
 * identical command from a terminal completed every round. Training is a long
 * job holding a GPU context, and it has no business being tied to the lifetime
 * or the console of a web dev server that restarts whenever a file changes.
 *
 * Because it is detached, the handle does not survive a dev-server reload, so
 * the pid is written to disk and status and stop work from the pid instead of
 * from a live ChildProcess object.
 */

const LOG_PATH = () => path.join(STATE_DIR, "worker.log");
const PID_PATH = () => path.join(STATE_DIR, "worker.pid");
const MAX_LOG_BYTES = 256 * 1024;

function readPid(): number | null {
  try {
    const raw = readFileSync(PID_PATH(), "utf8").trim();
    const pid = Number(raw);
    return Number.isFinite(pid) && pid > 0 ? pid : null;
  } catch {
    return null;
  }
}

/** Signal 0 tests for existence without touching the process. */
function pidAlive(pid: number | null): boolean {
  if (!pid) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function clearPid(): void {
  try {
    rmSync(PID_PATH(), { force: true });
  } catch {
    // Nothing to clean up.
  }
}

function tailLog(lines = 40): string[] {
  const file = LOG_PATH();
  if (!existsSync(file)) return [];
  try {
    const contents = readFileSync(file, "utf8");
    const size = statSync(file).size;
    const text = size > MAX_LOG_BYTES ? contents.slice(-MAX_LOG_BYTES) : contents;
    return text
      .split(/\r?\n/)
      .map((line) => line.replace(/\r/g, "").trim())
      .filter(Boolean)
      .slice(-lines);
  } catch {
    return [];
  }
}

export async function GET() {
  const user = await currentUser();
  if (!user) return NextResponse.json({ detail: "Sign in first." }, { status: 401 });

  const pid = readPid();
  const running = pidAlive(pid);
  if (!running && pid) clearPid();

  return NextResponse.json({
    running,
    pid: running ? pid : null,
    log: tailLog(),
    trainingPlane: setupState().trainingPlane,
  });
}

export async function POST() {
  const user = await currentUser();
  if (!user) return NextResponse.json({ detail: "Sign in first." }, { status: 401 });
  if (user.role !== "owner") {
    return NextResponse.json({ detail: "Only the mesh owner can do that." }, { status: 403 });
  }
  if (pidAlive(readPid())) return NextResponse.json({ running: true, alreadyRunning: true });

  const setup = setupState();
  if (setup.trainingPlane !== "ready") {
    return NextResponse.json(
      {
        detail:
          setup.trainingPlane === "installing"
            ? "PyTorch is still installing on this machine. Try again once setup finishes."
            : "The training runtime is not installed. Run `npm run setup` on this machine.",
      },
      { status: 409 }
    );
  }

  const token = meshToken();
  if (!token) {
    return NextResponse.json({ detail: "The coordinator has not minted a token yet." }, { status: 503 });
  }

  const engineDir = path.join(process.cwd(), "engine");
  const python =
    process.platform === "win32"
      ? path.join(process.cwd(), ".venv", "Scripts", "python.exe")
      : path.join(process.cwd(), ".venv", "bin", "python");

  if (!existsSync(python)) {
    return NextResponse.json({ detail: "No Python environment. Run `npm run setup`." }, { status: 409 });
  }

  mkdirSync(STATE_DIR, { recursive: true });
  const logPath = LOG_PATH();
  writeFileSync(logPath, `# GradMesh worker started ${new Date().toISOString()}\n`);
  const logFd = openSync(logPath, "a");

  const child = spawn(
    python,
    ["worker.py", "--server-url", COORDINATOR_URL, "--token", token, "--name", "this machine"],
    {
      cwd: engineDir,
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
      stdio: ["ignore", logFd, logFd],
      // Its own process group and, on Windows, its own console. This is what
      // stops the dev server's lifecycle from reaching the training process.
      detached: true,
      windowsHide: true,
    }
  );

  // Let this process exit without waiting on a job that may run for hours.
  child.unref();

  if (!child.pid) {
    return NextResponse.json({ detail: "Could not start the worker process." }, { status: 500 });
  }
  writeFileSync(PID_PATH(), String(child.pid));

  return NextResponse.json({ running: true, pid: child.pid, startedAt: Date.now() });
}

export async function DELETE() {
  const user = await currentUser();
  if (!user || user.role !== "owner") {
    return NextResponse.json({ detail: "Only the mesh owner can do that." }, { status: 403 });
  }

  const pid = readPid();
  if (pidAlive(pid) && pid) {
    if (process.platform === "win32") {
      // A detached process on Windows has its own tree, and Python's child
      // dataloader workers live inside it, so the tree has to go together.
      spawnSync("taskkill", ["/pid", String(pid), "/T", "/F"], { windowsHide: true });
    } else {
      try {
        // Negative pid targets the whole group created by detached: true.
        process.kill(-pid, "SIGTERM");
      } catch {
        try {
          process.kill(pid, "SIGTERM");
        } catch {
          // Already gone.
        }
      }
    }
  }
  clearPid();
  return NextResponse.json({ running: false });
}
