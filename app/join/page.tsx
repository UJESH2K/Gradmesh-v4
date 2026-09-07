import Link from "next/link";
import { headers } from "next/headers";

import JoinPanel from "@/components/JoinPanel";
import JoinPresence from "./JoinPresence";
import Logo from "@/components/Logo";
import { COORDINATOR_PORT, meshName, meshOrigin, meshToken } from "@/lib/config";

import "./join.css";

export const metadata = { title: "Contribute a GPU" };
export const dynamic = "force-dynamic";

export default async function JoinPage() {
  const headerList = await headers();
  const origin = meshOrigin(headerList.get("host"));
  const token = meshToken();
  // Same host the visitor already reached, on the coordinator's port.
  const coordinatorOrigin = `http://${new URL(origin).hostname}:${COORDINATOR_PORT}`;

  return (
    <main className="join-shell">
      <div className="join-card">
        <header className="row-between" style={{ marginBottom: 30 }}>
          <Link href="/" className="brand">
            <Logo />
            <span>{meshName()}</span>
          </Link>
          <span className="badge badge-live">
            <span className="dot dot-live" />
            Mesh is live
          </span>
        </header>

        <span className="eyebrow">Contribute a GPU</span>
        <h1 style={{ fontSize: "clamp(1.9rem, 4vw, 2.7rem)", marginTop: 12 }}>
          Lend this machine to the mesh.
        </h1>
        <p className="lead" style={{ marginTop: 14, maxWidth: 620 }}>
          Paste one line into a terminal on the machine with the GPU. It sets itself up and joins.
          Nothing is installed system-wide, and closing the terminal removes you from the mesh.
        </p>

        <JoinPresence coordinatorUrl={coordinatorOrigin} />

        <div style={{ marginTop: 32 }}>
          {token ? (
            <JoinPanel origin={origin} token={token} />
          ) : (
            <div className="notice notice-warn">
              This mesh has not finished starting up. Ask whoever is hosting to run{" "}
              <code className="code-inline">npm run dev</code>, then reload this page.
            </div>
          )}
        </div>

        <div className="notice" style={{ marginTop: 30 }}>
          <strong>What you are agreeing to.</strong> Your GPU will train a model on a dataset the
          host provides. Dataset slices and model weights move between machines on this network
          only. Your machine sends back model weights and timing numbers, nothing else from your
          computer.
        </div>

        <p className="small faint" style={{ marginTop: 26 }}>
          Hosting instead?{" "}
          <Link href="/dashboard" style={{ color: "var(--accent)" }}>
            Open the dashboard
          </Link>
        </p>
      </div>
    </main>
  );
}
