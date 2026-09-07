import { NextRequest, NextResponse } from "next/server";

import { COORDINATOR_URL, meshToken } from "@/lib/config";
import { currentUser } from "@/lib/auth";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";
// A YOLO export can be gigabytes, so the body is streamed straight through to
// the coordinator rather than buffered here.
export const maxDuration = 3600;

export async function POST(request: NextRequest) {
  const user = await currentUser();
  if (!user) return NextResponse.json({ detail: "Sign in first." }, { status: 401 });
  if (user.role !== "owner") {
    return NextResponse.json({ detail: "Only the mesh owner can upload datasets." }, { status: 403 });
  }

  const contentType = request.headers.get("content-type");
  if (!contentType?.includes("multipart/form-data")) {
    return NextResponse.json({ detail: "Send the dataset as multipart/form-data." }, { status: 400 });
  }

  try {
    const upstream = await fetch(`${COORDINATOR_URL}/datasets`, {
      method: "POST",
      headers: { "X-Mesh-Token": meshToken(), "Content-Type": contentType },
      body: request.body,
      // Required by undici when streaming a request body.
      duplex: "half",
    } as RequestInit & { duplex: "half" });

    const payload = await upstream.json().catch(() => ({ detail: "Upload failed." }));
    return NextResponse.json(payload, { status: upstream.status });
  } catch {
    return NextResponse.json({ detail: "The coordinator is not reachable." }, { status: 503 });
  }
}
