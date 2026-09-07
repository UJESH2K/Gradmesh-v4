import { NextResponse } from "next/server";

import { coordinatorHealthy } from "@/lib/coordinator";
import { primaryLanAddress, setupState } from "@/lib/config";
import { userCount } from "@/lib/auth";

export const dynamic = "force-dynamic";

/** Used by the dev launcher to know when the dashboard is actually serving. */
export async function GET() {
  return NextResponse.json({
    status: "ok",
    coordinator: (await coordinatorHealthy()) ? "up" : "down",
    setup: setupState(),
    accounts: userCount(),
    lan: primaryLanAddress(),
  });
}
