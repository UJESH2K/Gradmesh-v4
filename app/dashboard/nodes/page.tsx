import { currentUser } from "@/lib/auth";

import NodesView from "./NodesView";

export const metadata = { title: "Machines" };
export const dynamic = "force-dynamic";

export default async function NodesPage() {
  const user = await currentUser();
  return <NodesView canManage={user?.role === "owner"} />;
}
