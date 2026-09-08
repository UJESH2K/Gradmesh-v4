"use client";

import type { ReactNode } from "react";

export function StatTile({
  label,
  value,
  foot,
  accent = false,
}: {
  label: string;
  value: ReactNode;
  foot?: ReactNode;
  accent?: boolean;
}) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <span className={`stat-value${accent ? " is-accent" : ""}`}>{value}</span>
      {foot ? <span className="stat-foot">{foot}</span> : null}
    </div>
  );
}

export function Meter({
  value,
  tone = "accent",
}: {
  value: number;
  tone?: "accent" | "cyan" | "warn" | "danger";
}) {
  const width = Math.max(0, Math.min(1, value)) * 100;
  const toneClass = tone === "accent" ? "" : ` is-${tone}`;
  return (
    <div className="meter">
      <div className={`meter-fill${toneClass}`} style={{ width: `${width}%` }} />
    </div>
  );
}

export function Panel({
  title,
  action,
  children,
  flush = false,
  hidden = false,
}: {
  title?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  flush?: boolean;
  /** Kept mounted but not shown, so a toggled panel does not refetch on return. */
  hidden?: boolean;
}) {
  if (hidden) return null;
  if (!title) return <div className="panel">{children}</div>;
  return (
    <section className="panel panel-flush">
      <header className="panel-header">
        <span className="panel-title">{title}</span>
        {action}
      </header>
      {flush ? children : <div className="panel-body">{children}</div>}
    </section>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}

export function Spark({ values, height = 74 }: { values: number[]; height?: number }) {
  const peak = Math.max(...values, 1);
  return (
    <div className="spark" style={{ height }}>
      {values.map((value, index) => (
        <div
          key={index}
          className="spark-bar"
          style={{ height: `${Math.max(4, (value / peak) * 100)}%` }}
          title={String(value)}
        />
      ))}
    </div>
  );
}

const TIER_TONE: Record<string, string> = {
  full: "badge-live",
  probation: "badge-warn",
  rejected: "badge-danger",
};

const TIER_LABEL: Record<string, string> = {
  full: "Admitted",
  probation: "Probation",
  rejected: "Not eligible",
};

export function TierBadge({ tier, title }: { tier?: string; title?: string }) {
  if (!tier) return null;
  return (
    <span className={`badge ${TIER_TONE[tier] || ""}`} title={title}>
      {TIER_LABEL[tier] || tier}
    </span>
  );
}

const STATUS_TONE: Record<string, string> = {
  running: "badge-live",
  planning: "badge-cyan",
  waiting: "badge-warn",
  done: "badge-accent",
  failed: "badge-danger",
  stopped: "",
};

export function StatusBadge({ status }: { status: string }) {
  return (
    <span className={`badge ${STATUS_TONE[status] ?? ""}`}>
      {status === "running" ? <span className="dot dot-live" /> : null}
      {status}
    </span>
  );
}
