"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import Logo from "@/components/Logo";
import { useMesh } from "./MeshProvider";

const LINKS = [
  { href: "/dashboard", label: "Overview", exact: true },
  { href: "/dashboard/nodes", label: "Machines" },
  { href: "/dashboard/discover", label: "Discover devices" },
  { href: "/dashboard/training", label: "Live training" },
  { href: "/dashboard/runs", label: "Training runs" },
  { href: "/dashboard/datasets", label: "Datasets" },
  { href: "/dashboard/policy", label: "Scheduler" },
  { href: "/dashboard/testing", label: "Testing parameters" },
  { href: "/dashboard/invite", label: "Invite a GPU" },
];

export default function Sidebar({
  meshName,
  user,
}: {
  meshName: string;
  user: { name: string; email: string; role: string };
}) {
  const pathname = usePathname();
  const { mesh, connected } = useMesh();

  return (
    <aside className="dash-side">
      <Link href="/" className="dash-brand">
        <Logo />
        <span>{meshName}</span>
      </Link>

      <nav className="dash-nav">
        <span className="dash-nav-label">Mesh</span>
        {LINKS.map((link) => {
          const active = link.exact ? pathname === link.href : pathname.startsWith(link.href);
          const count =
            link.href === "/dashboard/nodes"
              ? mesh?.metrics.nodes_active
              : link.href === "/dashboard/runs"
                ? mesh?.active_runs.length || undefined
                : undefined;
          return (
            <Link key={link.href} href={link.href} className={`dash-link${active ? " is-active" : ""}`}>
              <span>{link.label}</span>
              {count ? <span className="badge badge-accent">{count}</span> : null}
            </Link>
          );
        })}
      </nav>

      <div className="dash-side-foot">
        <div className="row small" style={{ gap: 8, padding: "0 8px" }}>
          <span className={`dot${connected ? " dot-live" : ""}`} style={connected ? undefined : { background: "var(--text-faint)" }} />
          <span className="faint">{connected ? "Live" : "Reconnecting"}</span>
        </div>
        <div style={{ padding: "0 8px" }}>
          <div className="small truncate">{user.name}</div>
          <div className="small faint truncate">{user.role === "owner" ? "Mesh owner" : "Member"}</div>
        </div>
        <form action="/api/auth/signout" method="post">
          <button className="btn btn-ghost btn-sm" type="submit" style={{ width: "100%" }}>
            Sign out
          </button>
        </form>
      </div>
    </aside>
  );
}
