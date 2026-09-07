import { NextResponse } from "next/server";

import { currentUser } from "@/lib/auth";
import { setupState } from "@/lib/config";
import { coordinatorHealthy } from "@/lib/coordinator";

export const dynamic = "force-dynamic";

/** Install progress for the banner the dashboard shows while torch downloads. */
export async function GET() {
  const user = await currentUser();
  if (!user) return NextResponse.json({ detail: "Sign in first." }, { status: 401 });

  return NextResponse.json({
    ...setupState(),
    coordinator: (await coordinatorHealthy()) ? "up" : "down",
  });
}
