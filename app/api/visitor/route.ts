import { NextRequest, NextResponse } from "next/server";

import { COORDINATOR_URL, meshToken } from "@/lib/config";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Presence beacon from the public join page.
 *
 * Unauthenticated on purpose: the whole point is that a device announces itself
 * before anyone has installed or signed into anything. The mesh token is added
 * here rather than shipped to the browser, and the payload is capped and rate
 * limited so an open endpoint on the LAN cannot be used to flood the host.
 */

const WINDOW_MS = 10000;
const MAX_PER_WINDOW = 6;
const MAX_BODY_BYTES = 4096;

type Bucket = { count: number; resetAt: number };
const buckets = globalThis as unknown as { __gradmeshVisitorRate?: Map<string, Bucket> };
buckets.__gradmeshVisitorRate ??= new Map();

function allowed(key: string): boolean {
  const now = Date.now();
  const store = buckets.__gradmeshVisitorRate!;
  const bucket = store.get(key);

  if (!bucket || bucket.resetAt < now) {
    store.set(key, { count: 1, resetAt: now + WINDOW_MS });
    return true;
  }
  if (bucket.count >= MAX_PER_WINDOW) return false;
  bucket.count += 1;
  return true;
}

export async function POST(request: NextRequest) {
  const key =
    request.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ||
    request.headers.get("x-real-ip") ||
    "unknown";

  if (!allowed(key)) {
    return NextResponse.json({ detail: "Too many announcements." }, { status: 429 });
  }

  const raw = await request.text();
  if (raw.length > MAX_BODY_BYTES) {
    return NextResponse.json({ detail: "Payload too large." }, { status: 413 });
  }

  let body: unknown;
  try {
    body = JSON.parse(raw);
  } catch {
    return NextResponse.json({ detail: "Invalid JSON." }, { status: 400 });
  }

  const token = meshToken();
  if (!token) {
    return NextResponse.json({ detail: "The mesh is not running." }, { status: 503 });
  }

  try {
    const upstream = await fetch(`${COORDINATOR_URL}/visitors`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Mesh-Token": token },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(5000),
    });
    const payload = await upstream.json().catch(() => ({}));
    return NextResponse.json(payload, { status: upstream.status });
  } catch {
    return NextResponse.json({ detail: "The coordinator is not reachable." }, { status: 503 });
  }
}
