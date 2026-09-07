import { NextRequest, NextResponse } from "next/server";

import { coordinatorStream } from "@/lib/coordinator";
import { currentUser } from "@/lib/auth";

export const dynamic = "force-dynamic";
// Node runtime, not edge: the token comes off the local filesystem.
export const runtime = "nodejs";

/**
 * Proxies the coordinator's Server-Sent Events feed to the browser.
 *
 * The dashboard holds one of these per tab instead of polling, which is what
 * makes shard timings and straggler decisions appear the moment they happen.
 */
export async function GET(request: NextRequest) {
  const user = await currentUser();
  if (!user) return NextResponse.json({ detail: "Sign in first." }, { status: 401 });

  const replay = request.nextUrl.searchParams.get("replay") ?? "40";

  let upstream: Response;
  try {
    upstream = await coordinatorStream(`/events?replay=${encodeURIComponent(replay)}`, {
      signal: request.signal,
    });
  } catch {
    return NextResponse.json({ detail: "The coordinator is not reachable." }, { status: 503 });
  }

  if (!upstream.ok || !upstream.body) {
    return NextResponse.json({ detail: "The event stream is unavailable." }, { status: 503 });
  }

  return new NextResponse(upstream.body, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}
