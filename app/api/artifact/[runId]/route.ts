import { NextResponse } from "next/server";

import { coordinatorStream } from "@/lib/coordinator";
import { currentUser } from "@/lib/auth";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/** Streams a finished run's aggregated weights to the browser as a .pt file. */
export async function GET(_request: Request, context: { params: Promise<{ runId: string }> }) {
  const user = await currentUser();
  if (!user) return NextResponse.json({ detail: "Sign in first." }, { status: 401 });

  const { runId } = await context.params;
  const upstream = await coordinatorStream(`/runs/${encodeURIComponent(runId)}/artifact`);

  if (!upstream.ok || !upstream.body) {
    return NextResponse.json({ detail: "No weights are available for this run." }, { status: 404 });
  }

  return new NextResponse(upstream.body, {
    headers: {
      "Content-Type": "application/octet-stream",
      "Content-Disposition": `attachment; filename="gradmesh-${runId}.pt"`,
    },
  });
}
