import { spawn, type ChildProcess } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { NextResponse } from "next/server";

import { currentUser } from "@/lib/auth";
import { COORDINATOR_URL, meshToken, setupState } from "@/lib/config";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Contribute the host machine's own GPU without opening a second terminal.
 *
 * The dashboard runs on the host, so it can start the worker process here. That
 * removes the last reason the owner needed two devices to see the mesh do
 * anything.
 *
 * The handle lives on globalThis so a dev-server hot reload does not lose track
 * of a process it started.
 */
type WorkerHandle = { child: ChildProcess; startedAt: number; log: string[] };
const registry = globalThis as unknown as { __gradmeshWorker?: WorkerHandle };

function isRunning(): boolean {
  const handle = registry.__gradmeshWorker;
  return Boolean(handle && handle.child.exitCode === null && !handle.child.killed);
}

export async function GET() {
  const user = await currentUser();
  if (!user) return NextResponse.json({ detail: "Sign in first." }, { status: 401 });

  const handle = registry.__gradmeshWorker;
  return NextResponse.json({
    running: isRunning(),
    startedAt: handle?.startedAt ?? null,
    log: handle?.log.slice(-40) ?? [],
    trainingPlane: setupState().trainingPlane,
  });
}

export async function POST() {
  const user = await currentUser();
  if (!user) return NextResponse.json({ detail: "Sign in first." }, { status: 401 });
  if (user.role !== "owner") {
    return NextResponse.json({ detail: "Only the mesh owner can do that." }, { status: 403 });
  }
  if (isRunning()) return NextResponse.json({ running: true, alreadyRunning: true });

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

  const child = spawn(
    python,
    ["worker.py", "--server-url", COORDINATOR_URL, "--token", token, "--name", "this machine"],
    { cwd: engineDir, env: { ...process.env, PYTHONUNBUFFERED: "1" }, stdio: ["ignore", "pipe", "pipe"] }
  );

  const handle: WorkerHandle = { child, startedAt: Date.now(), log: [] };
  registry.__gradmeshWorker = handle;

  const record = (chunk: Buffer) => {
    for (const line of chunk.toString("utf8").split(/\r?\n/)) {
      if (line.trim()) handle.log.push(line.trim());
    }
    if (handle.log.length > 200) handle.log.splice(0, handle.log.length - 200);
  };
  child.stdout?.on("data", record);
  child.stderr?.on("data", record);

  return NextResponse.json({ running: true, startedAt: handle.startedAt });
}

export async function DELETE() {
  const user = await currentUser();
  if (!user || user.role !== "owner") {
    return NextResponse.json({ detail: "Only the mesh owner can do that." }, { status: 403 });
  }

  const handle = registry.__gradmeshWorker;
  if (handle && isRunning()) {
    // The agent deregisters on SIGTERM, so the mesh sees it leave immediately
    // rather than waiting out a heartbeat timeout.
    handle.child.kill(process.platform === "win32" ? undefined : "SIGTERM");
  }
  registry.__gradmeshWorker = undefined;
  return NextResponse.json({ running: false });
}
