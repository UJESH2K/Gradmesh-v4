import "server-only";

import { COORDINATOR_URL, meshToken } from "./config";

export class CoordinatorError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
    this.name = "CoordinatorError";
  }
}

type RequestOptions = {
  method?: string;
  body?: unknown;
  signal?: AbortSignal;
  timeoutMs?: number;
  raw?: BodyInit;
  headers?: Record<string, string>;
};

/**
 * Every call to the Python control plane goes through here.
 *
 * The mesh token never reaches the browser on a normal page load: the dashboard
 * talks to Next, and Next attaches the token server-side. The one deliberate
 * exception is the Join page, which has to show the token because that is what
 * a contributor pastes into their own terminal.
 */
export async function coordinator<T = unknown>(
  path: string,
  options: RequestOptions = {}
): Promise<T> {
  const { method = "GET", body, raw, timeoutMs = 30000, headers = {} } = options;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(`${COORDINATOR_URL}${path}`, {
      method,
      headers: {
        "X-Mesh-Token": meshToken(),
        ...(body !== undefined ? { "Content-Type": "application/json" } : {}),
        ...headers,
      },
      body: raw ?? (body !== undefined ? JSON.stringify(body) : undefined),
      signal: options.signal ?? controller.signal,
      cache: "no-store",
    });

    if (!response.ok) {
      const detail = await readError(response);
      throw new CoordinatorError(detail, response.status);
    }

    if (response.status === 204) return undefined as T;
    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof CoordinatorError) throw error;
    if ((error as Error).name === "AbortError") {
      throw new CoordinatorError("The coordinator did not respond in time.", 504);
    }
    throw new CoordinatorError(
      "The coordinator is not reachable. Is `npm run dev` still running?",
      503
    );
  } finally {
    clearTimeout(timer);
  }
}

async function readError(response: Response): Promise<string> {
  try {
    const payload = await response.json();
    if (typeof payload?.detail === "string") return payload.detail;
    if (Array.isArray(payload?.detail)) {
      return payload.detail.map((item: { msg?: string }) => item.msg).filter(Boolean).join("; ");
    }
    return JSON.stringify(payload);
  } catch {
    return `The coordinator returned ${response.status}.`;
  }
}

/** Streaming and binary passthrough for the SSE feed, shard zips and artifacts. */
export async function coordinatorStream(path: string, init?: RequestInit): Promise<Response> {
  return fetch(`${COORDINATOR_URL}${path}`, {
    ...init,
    headers: { ...(init?.headers || {}), "X-Mesh-Token": meshToken() },
    cache: "no-store",
  });
}

export async function coordinatorHealthy(): Promise<boolean> {
  try {
    const response = await fetch(`${COORDINATOR_URL}/health`, {
      signal: AbortSignal.timeout(2500),
      cache: "no-store",
    });
    return response.ok;
  } catch {
    return false;
  }
}
