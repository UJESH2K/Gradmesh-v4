import { NextResponse } from "next/server";

import { coordinatorStream } from "@/lib/coordinator";
import { currentUser } from "@/lib/auth";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Streams one sweep artifact to the browser with a filename worth keeping.
 *
 * Separate from the general mesh proxy because that route parses JSON and
 * re-serialises it, which would corrupt a CSV and pointlessly double-handle a
 * large results file.
 */
const ALLOWED: Record<string, string> = {
  "suite.json": "application/json",
  "results.csv": "text/csv",
  "summary.csv": "text/csv",
};

export async function GET(
  _request: Request,
  context: { params: Promise<{ suiteId: string; artifact: string }> }
) {
  const user = await currentUser();
  if (!user) return NextResponse.json({ detail: "Sign in first." }, { status: 401 });

  const { suiteId, artifact } = await context.params;
  const mime = ALLOWED[artifact];
  if (!mime) return NextResponse.json({ detail: "Unknown artifact." }, { status: 404 });

  const upstream = await coordinatorStream(
    `/benchmarks/${encodeURIComponent(suiteId)}/download/${encodeURIComponent(artifact)}`
  );
  if (!upstream.ok || !upstream.body) {
    return NextResponse.json({ detail: "That file has not been written yet." }, { status: 404 });
  }

  return new NextResponse(upstream.body, {
    headers: {
      "Content-Type": mime,
      "Content-Disposition": `attachment; filename="gradmesh-${suiteId}-${artifact}"`,
    },
  });
}
