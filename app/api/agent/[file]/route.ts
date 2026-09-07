import { readFile } from "node:fs/promises";
import path from "node:path";
import { NextResponse } from "next/server";

import { AGENT_FILE_SET } from "@/lib/agent";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Serves the worker agent's source to joining machines. A contributor never
 * clones the repository; the join script pulls exactly these files.
 */
export async function GET(_request: Request, context: { params: Promise<{ file: string }> }) {
  const { file } = await context.params;
  if (!AGENT_FILE_SET.has(file)) {
    return NextResponse.json({ detail: "Unknown agent file." }, { status: 404 });
  }

  try {
    const contents = await readFile(path.join(process.cwd(), "engine", file), "utf8");
    return new NextResponse(contents, {
      headers: { "Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-store" },
    });
  } catch {
    return NextResponse.json({ detail: "Agent file is missing on the host." }, { status: 500 });
  }
}
