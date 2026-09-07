import { NextRequest, NextResponse } from "next/server";

import { CoordinatorError, coordinator, coordinatorStream } from "@/lib/coordinator";
import { currentUser } from "@/lib/auth";

export const dynamic = "force-dynamic";

/**
 * The dashboard's single door to the coordinator.
 *
 * The browser never sees the mesh token: it calls this route with its session
 * cookie, and the token is attached here. Writes require the owner role, reads
 * only require a session.
 */

const READ_ONLY_METHODS = new Set(["GET", "HEAD"]);

async function authorize(method: string) {
  const user = await currentUser();
  if (!user) return { error: NextResponse.json({ detail: "Sign in first." }, { status: 401 }) };
  if (!READ_ONLY_METHODS.has(method) && user.role !== "owner") {
    return {
      error: NextResponse.json(
        { detail: "Only the mesh owner can change the mesh." },
        { status: 403 }
      ),
    };
  }
  return { user };
}

function targetPath(segments: string[], request: NextRequest): string {
  const search = request.nextUrl.search;
  return `/${segments.map(encodeURIComponent).join("/")}${search}`;
}

export async function GET(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  const guard = await authorize("GET");
  if (guard.error) return guard.error;
  const { path } = await context.params;

  // Binary endpoints stream through untouched.
  if (path.at(-1)?.endsWith(".zip") || path.at(-1) === "artifact") {
    const upstream = await coordinatorStream(targetPath(path, request));
    return new NextResponse(upstream.body, {
      status: upstream.status,
      headers: {
        "Content-Type": upstream.headers.get("Content-Type") || "application/octet-stream",
        "Content-Disposition": upstream.headers.get("Content-Disposition") || "attachment",
      },
    });
  }

  return handle(() => coordinator(targetPath(path, request)));
}

export async function POST(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  const guard = await authorize("POST");
  if (guard.error) return guard.error;
  const { path } = await context.params;
  const body = await readBody(request);
  return handle(() => coordinator(targetPath(path, request), { method: "POST", body }));
}

export async function PUT(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  const guard = await authorize("PUT");
  if (guard.error) return guard.error;
  const { path } = await context.params;
  const body = await readBody(request);
  return handle(() => coordinator(targetPath(path, request), { method: "PUT", body }));
}

export async function DELETE(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  const guard = await authorize("DELETE");
  if (guard.error) return guard.error;
  const { path } = await context.params;
  return handle(() => coordinator(targetPath(path, request), { method: "DELETE" }));
}

async function readBody(request: NextRequest): Promise<unknown> {
  const text = await request.text();
  if (!text) return undefined;
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

async function handle(work: () => Promise<unknown>) {
  try {
    return NextResponse.json((await work()) ?? {});
  } catch (error) {
    if (error instanceof CoordinatorError) {
      return NextResponse.json({ detail: error.message }, { status: error.status });
    }
    return NextResponse.json({ detail: "Unexpected coordinator failure." }, { status: 500 });
  }
}
