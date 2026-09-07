import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { randomBytes } from "node:crypto";
import { networkInterfaces } from "node:os";
import path from "node:path";

export const REPO_ROOT = process.cwd();
export const STATE_DIR = process.env.GRADMESH_STATE_DIR || path.join(REPO_ROOT, ".gradmesh");
export const COORDINATOR_URL = (
  process.env.GRADMESH_COORDINATOR_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");
export const WEB_PORT = Number(process.env.PORT || 3000);
export const COORDINATOR_PORT = Number(process.env.GRADMESH_COORDINATOR_PORT || 8000);

export type CoordinatorState = {
  mesh_token?: string;
  mesh_name?: string;
  datasets?: Record<string, unknown>;
  default_dataset_id?: string | null;
};

export function ensureStateDir(): string {
  if (!existsSync(STATE_DIR)) mkdirSync(STATE_DIR, { recursive: true });
  return STATE_DIR;
}

/**
 * The mesh token is minted by the Python coordinator, so the dashboard reads it
 * from the same file rather than keeping a second copy that could drift.
 */
export function readCoordinatorState(): CoordinatorState | null {
  const file = path.join(STATE_DIR, "coordinator.json");
  if (!existsSync(file)) return null;
  try {
    return JSON.parse(readFileSync(file, "utf8")) as CoordinatorState;
  } catch {
    return null;
  }
}

export function meshToken(): string {
  return readCoordinatorState()?.mesh_token || "";
}

export function meshName(): string {
  return readCoordinatorState()?.mesh_name || "GradMesh";
}

/** Session signing key. Persisted so logins survive a dev-server restart. */
export function sessionSecret(): string {
  if (process.env.GRADMESH_SESSION_SECRET) return process.env.GRADMESH_SESSION_SECRET;
  ensureStateDir();
  const file = path.join(STATE_DIR, "session.key");
  if (existsSync(file)) return readFileSync(file, "utf8").trim();
  const generated = randomBytes(32).toString("hex");
  writeFileSync(file, generated, { mode: 0o600 });
  return generated;
}

export type LanAddress = { name: string; address: string };

export function lanAddresses(): LanAddress[] {
  const found: LanAddress[] = [];
  for (const [name, entries] of Object.entries(networkInterfaces())) {
    for (const entry of entries || []) {
      if (entry.family !== "IPv4" || entry.internal) continue;
      found.push({ name, address: entry.address });
    }
  }
  const score = (address: string) =>
    address.startsWith("192.168.") ? 0 : address.startsWith("10.") ? 1 : address.startsWith("172.") ? 2 : 3;
  return found.sort((a, b) => score(a.address) - score(b.address));
}

export function primaryLanAddress(): string {
  return lanAddresses()[0]?.address || "127.0.0.1";
}

/**
 * The origin a peer on the same network should use. A request's Host header is
 * the most reliable source, because it is literally the address that worked for
 * whoever is looking at the page.
 */
export function meshOrigin(requestHost?: string | null): string {
  if (requestHost && !requestHost.startsWith("localhost") && !requestHost.startsWith("127.")) {
    return `http://${requestHost}`;
  }
  return `http://${primaryLanAddress()}:${WEB_PORT}`;
}

export function setupState() {
  const file = path.join(STATE_DIR, "setup.json");
  const fallback = {
    controlPlane: "pending",
    trainingPlane: "pending",
    models: [] as string[],
    backend: null as string | null,
    messages: [] as { at: number; message: string }[],
  };
  if (!existsSync(file)) return fallback;
  try {
    return { ...fallback, ...JSON.parse(readFileSync(file, "utf8")) };
  } catch {
    return fallback;
  }
}
