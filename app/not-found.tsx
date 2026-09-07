import Link from "next/link";

import Logo from "@/components/Logo";

export default function NotFound() {
  return (
    <main style={{ minHeight: "100vh", display: "grid", placeItems: "center", padding: 40 }}>
      <div className="stack" style={{ maxWidth: 420, textAlign: "center", alignItems: "center" }}>
        <Logo size={30} />
        <h1 style={{ fontSize: "2rem" }}>Nothing here</h1>
        <p className="muted">That page is not part of the mesh.</p>
        <div className="row" style={{ gap: 8 }}>
          <Link className="btn btn-primary" href="/dashboard">
            Dashboard
          </Link>
          <Link className="btn" href="/join">
            Contribute a GPU
          </Link>
        </div>
      </div>
    </main>
  );
}
