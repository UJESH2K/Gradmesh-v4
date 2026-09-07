import { currentUser } from "@/lib/auth";

import PolicyEditor from "./PolicyEditor";

export const metadata = { title: "Scheduler" };
export const dynamic = "force-dynamic";

export default async function PolicyPage() {
  const user = await currentUser();
  return <PolicyEditor canManage={user?.role === "owner"} />;
}
