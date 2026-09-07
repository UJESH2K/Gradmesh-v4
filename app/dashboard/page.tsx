import { currentUser } from "@/lib/auth";

import Overview from "./Overview";

export const metadata = { title: "Overview" };
export const dynamic = "force-dynamic";

export default async function DashboardPage() {
  const user = await currentUser();
  return (
    <>
      <h1 className="page-title">Mesh overview</h1>
      <Overview canManage={user?.role === "owner"} />
    </>
  );
}
