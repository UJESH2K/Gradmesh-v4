import { currentUser } from "@/lib/auth";

import TestingLab from "./TestingLab";

export const metadata = { title: "Testing parameters" };
export const dynamic = "force-dynamic";

export default async function TestingPage() {
  const user = await currentUser();
  return <TestingLab canManage={user?.role === "owner"} />;
}
