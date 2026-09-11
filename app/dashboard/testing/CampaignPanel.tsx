"use client";

import { useEffect, useRef, useState } from "react";

import { Panel } from "@/components/dashboard/ui";
import { bytes, clock, percent, seconds } from "@/lib/format";
import type { Campaign, Suite } from "@/lib/types";

/**
 * The multi-network flow.
 *
 * A campaign is the same design run on more than one network. The point is that
 * only the network changes: identical machines, dataset, subsets, seeds and
 * matrix, so a difference in the results is a difference in the network rather
 * than in anything else.
 *
 * The prompt below is the hinge. When a leg finishes, the next thing that has to
 * happen is a human walking over and switching the Wi-Fi, and nothing can
 * proceed until they do. So it is loud, it says what just finished, and it will
 * not let the next leg start under the same label.
 */

export function NetworkChangePrompt({
  suite,
  onStartNextLeg,
  busy,
  nodesOnline,
}: {
  suite: Suite;
  onStartNextLeg: (label: string) => Promise<void>;
  busy: boolean;
  nodesOnline: number;
}) {
  const previousLabel = suite.config.network_label;
  const [label, setLabel] = useState("");
  const [dismissed, setDismissed] = useState(false);
  const notified = useRef<string | null>(null);

  // A sweep is a long job and nobody watches it finish. A desktop notification
  // is best effort: it needs permission, and browsers only grant it on a secure
  // origin, which on a LAN address means it will not fire. The banner is the
  // real mechanism; this is a bonus when the host itself is being used.
  useEffect(() => {
    if (notified.current === suite.id) return;
    notified.current = suite.id;
    try {
      if (typeof Notification === "undefined") return;
      const show = () =>
        new Notification("GradMesh: sweep finished", {
          body: `Leg ${suite.config.leg} on "${previousLabel}" is done. Switch networks to run the next leg.`,
          tag: `gradmesh-${suite.id}`,
        });
      if (Notification.permission === "granted") show();
      else if (Notification.permission === "default") {
        void Notification.requestPermission().then((result) => {
          if (result === "granted") show();
        });
      }
    } catch {
      // Notifications are unavailable. The banner already covers this.
    }
  }, [suite.id, suite.config.leg, previousLabel]);

  if (dismissed) return null;

  const trimmed = label.trim();
  const sameAsBefore = trimmed.toLowerCase() === previousLabel.toLowerCase();
  const canStart = trimmed.length > 0 && !sameAsBefore && !busy && nodesOnline > 0;
  const completed = suite.results.filter((r) => r.status === "done").length;

  return (
    <div className="handoff">
      <div className="handoff-head">
        <span className="badge badge-live">
          <span className="dot dot-live" />
          Leg {suite.config.leg} finished
        </span>
        <button
          type="button"
          className="btn btn-ghost btn-sm"
          onClick={() => setDismissed(true)}
          title="Hide this. The campaign stays open and you can start the next leg from the list."
        >
          Dismiss
        </button>
      </div>

      <h2 style={{ marginTop: 12 }}>Change the network now</h2>
      <p className="muted" style={{ marginTop: 8, maxWidth: 620 }}>
        {completed} trial{completed === 1 ? "" : "s"} completed on{" "}
        <strong>{previousLabel}</strong>. Move every machine onto the next network, wait for them to
        reappear, then start the next leg. The design is copied exactly, so only the network differs.
      </p>

      <ol className="handoff-steps">
        <li>Switch this host and every worker to the new network.</li>
        <li>
          Check the machine count below reads{" "}
          <strong>{suite.environment.nodes.length}</strong> again. Workers reconnect on their own,
          and <code className="code-inline">gradmesh.local</code> follows the host.
        </li>
        <li>Name the new network and start leg {suite.config.leg + 1}.</li>
      </ol>

      <div className="handoff-status">
        <span className={nodesOnline >= suite.environment.nodes.length ? "accent" : "faint"}>
          {nodesOnline} of {suite.environment.nodes.length} machines back online
        </span>
        {nodesOnline > 0 && nodesOnline < suite.environment.nodes.length ? (
          <span className="faint small">
            {" "}
            · missing machines make their cells skipped, not smaller, so the legs stay comparable
          </span>
        ) : null}
      </div>

      <div className="handoff-actions">
        <input
          className="input"
          placeholder="New network label, e.g. phone-hotspot"
          value={label}
          onChange={(event) => setLabel(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && canStart) void onStartNextLeg(trimmed);
          }}
        />
        <button
          className="btn btn-primary"
          type="button"
          disabled={!canStart}
          onClick={() => void onStartNextLeg(trimmed)}
        >
          {busy ? "Starting…" : `Start leg ${suite.config.leg + 1}`}
        </button>
      </div>

      {sameAsBefore ? (
        <p className="small" style={{ color: "var(--warn)", marginTop: 8 }}>
          That is the same label as the leg that just ran. Use a different one or the two legs
          cannot be told apart in the results.
        </p>
      ) : null}
      {nodesOnline === 0 ? (
        <p className="small faint" style={{ marginTop: 8 }}>
          No machine is online yet. The workers need to reconnect before the next leg can start.
        </p>
      ) : null}
    </div>
  );
}

export function CampaignComparison({ campaign }: { campaign: Campaign }) {
  const rows = campaign.comparison || [];
  if (rows.length < 2) return null;

  const baseline = rows[0];
  const delta = (value: number | null, base: number | null) =>
    value == null || base == null || base === 0 ? null : (value - base) / base;

  return (
    <Panel title={`Across networks: ${campaign.name}`} flush>
      <div className="table-scroll">
        <table className="table">
          <thead>
            <tr>
              <th className="num">Leg</th>
              <th>Network</th>
              <th className="num">Trials</th>
              <th className="num">Latency</th>
              <th className="num">Train time</th>
              <th className="num">Speedup</th>
              <th className="num">mAP@50</th>
              <th className="num">Network share</th>
              <th className="num">vs leg 1</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const change = delta(row.train_seconds, baseline.train_seconds);
              return (
                <tr key={row.suite_id}>
                  <td className="num">{row.leg}</td>
                  <td>
                    <strong>{row.network_label}</strong>
                  </td>
                  <td className="num">{row.trials}</td>
                  <td className="num">
                    {row.mean_latency_ms != null ? `${row.mean_latency_ms.toFixed(1)} ms` : "—"}
                  </td>
                  <td className="num">{row.train_seconds != null ? seconds(row.train_seconds) : "—"}</td>
                  <td className="num">
                    {row.speedup != null ? `${row.speedup.toFixed(2)}x` : "—"}
                  </td>
                  <td className="num">{row.map50 != null ? row.map50.toFixed(3) : "—"}</td>
                  <td className="num">
                    {row.comm_fraction != null ? percent(row.comm_fraction) : "—"}
                  </td>
                  <td className="num">
                    {change == null ? (
                      "—"
                    ) : (
                      <span className={change > 0.02 ? "" : "accent"}>
                        {change > 0 ? "+" : ""}
                        {(change * 100).toFixed(1)}%
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="panel-body" style={{ borderTop: "1px solid var(--line)" }}>
        <p className="small faint">
          Every leg ran the same design on the same machines, so the differences here are the
          network. Speedup averages the multi-machine trials only, since a single-machine trial is
          the baseline and is always 1.0. The last column compares training time against leg 1, so a
          positive number means that network was slower. Per-cell detail stays in each leg&apos;s own
          results, and{" "}
          <code className="code-inline">
            npm run report {rows[rows.length - 1]?.suite_id ?? ""}
          </code>{" "}
          builds the figures for one leg.
        </p>
      </div>
    </Panel>
  );
}

export function CampaignList({
  campaigns,
  openSuiteId,
  onOpen,
}: {
  campaigns: Campaign[];
  openSuiteId: string | null;
  onOpen: (id: string) => void;
}) {
  if (campaigns.length === 0) return null;

  return (
    <Panel title="Campaigns" flush>
      <div className="suite-list">
        {campaigns.map((campaign) => (
          <div key={campaign.campaign_id} className="campaign-block">
            <div className="row-between" style={{ padding: "12px 20px 4px" }}>
              <strong className="truncate">{campaign.name}</strong>
              <span className="small faint">
                {campaign.legs.length} leg{campaign.legs.length === 1 ? "" : "s"} ·{" "}
                {clock(campaign.created_at)}
              </span>
            </div>
            {campaign.legs.map((leg) => (
              <button
                key={leg.id}
                type="button"
                className={`suite-row${openSuiteId === leg.id ? " is-open" : ""}`}
                onClick={() => onOpen(leg.id)}
                style={{ paddingLeft: 32 }}
              >
                <span className="grow" style={{ minWidth: 0 }}>
                  <span className="row" style={{ gap: 8 }}>
                    <span className="badge">leg {leg.leg}</span>
                    <strong className="truncate">{leg.network_label}</strong>
                    <span
                      className={`badge ${
                        leg.status === "running"
                          ? "badge-live"
                          : leg.status === "done"
                            ? "badge-accent"
                            : "badge-warn"
                      }`}
                    >
                      {leg.status}
                    </span>
                  </span>
                  <span className="small faint">
                    {leg.completed_trials}/{leg.total_trials} trials
                  </span>
                </span>
              </button>
            ))}
          </div>
        ))}
      </div>
    </Panel>
  );
}

export function bytesLabel(value: number | null | undefined): string {
  return value ? bytes(value) : "—";
}
