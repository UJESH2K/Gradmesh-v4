import { headers } from "next/headers";

import { currentUser } from "@/lib/auth";
import { meshOrigin } from "@/lib/config";

import DiscoverView from "./DiscoverView";

export const metadata = { title: "Discover devices" };
export const dynamic = "force-dynamic";

export default async function DiscoverPage() {
  const headerList = await headers();
  const user = await currentUser();
  return (
    <DiscoverView canManage={user?.role === "owner"} origin={meshOrigin(headerList.get("host"))} />
  );
}
