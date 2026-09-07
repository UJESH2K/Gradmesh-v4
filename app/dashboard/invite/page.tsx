import { headers } from "next/headers";

import JoinPanel from "@/components/JoinPanel";
import { currentUser } from "@/lib/auth";
import { meshOrigin, meshToken } from "@/lib/config";

import InviteActions from "./InviteActions";

export const metadata = { title: "Invite a GPU" };
export const dynamic = "force-dynamic";

export default async function InvitePage() {
  const headerList = await headers();
  const user = await currentUser();
  const origin = meshOrigin(headerList.get("host"));
  const token = meshToken();

  return (
    <>
      <div>
        <h1 className="page-title">Invite a GPU</h1>
        <p className="small faint" style={{ marginTop: 4 }}>
          Send this to anyone on the same network. They paste one line and their machine appears in
          the dashboard.
        </p>
      </div>

      <section className="panel">
        {token ? (
          <JoinPanel origin={origin} token={token} />
        ) : (
          <div className="notice notice-warn">
            The coordinator has not minted a join token yet.
          </div>
        )}
      </section>

      <InviteActions canManage={user?.role === "owner"} joinUrl={`${origin}/join`} />
    </>
  );
}
