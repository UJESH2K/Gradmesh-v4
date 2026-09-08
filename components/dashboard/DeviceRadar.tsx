"use client";

import { useMemo, useState } from "react";

import type { DiscoverState, NetworkDevice, Visitor } from "@/lib/types";

/**
 * The network, drawn the way a file-sharing app draws it: this host at the
 * centre, everything else orbiting at a distance that means something.
 *
 * Two rules keep it honest rather than decorative.
 *
 * Radius is measured network round trip, not invented. A device whose distance
 * could not be measured sits on the outer "unknown" ring rather than being
 * placed somewhere plausible-looking, because a made-up position on a map that
 * claims to show proximity is worse than an empty one.
 *
 * Angle is derived from a hash of the device's address, so a machine keeps the
 * same seat on every sweep. Positions that jump around between refreshes make
 * the view unreadable and make people distrust it.
 */

type RadarEntry = {
  id: string;
  label: string;
  sub: string;
  kind: "member" | "visitor" | "idle" | "self";
  proximity: "close" | "nearby" | "far" | "unknown";
  rtt: number | null;
  device?: NetworkDevice;
  visitor?: Visitor;
};

const RING_RADIUS: Record<RadarEntry["proximity"], number> = {
  close: 0.3,
  nearby: 0.55,
  far: 0.78,
  unknown: 0.93,
};

const RING_LABEL: Record<RadarEntry["proximity"], string> = {
  close: "Same link",
  nearby: "Nearby",
  far: "Distant",
  unknown: "Not measurable",
};

/** Stable pseudo-random angle from a string, so seats never shuffle. */
function angleFor(seed: string): number {
  let hash = 2166136261;
  for (let index = 0; index < seed.length; index += 1) {
    hash ^= seed.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return ((hash >>> 0) % 3600) / 3600;
}

function initials(label: string): string {
  const words = label.replace(/[^A-Za-z0-9 ]/g, " ").trim().split(/\s+/);
  if (words.length === 0) return "?";
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
  return (words[0][0] + words[1][0]).toUpperCase();
}

export default function DeviceRadar({
  state,
  scanning,
  onRescan,
}: {
  state: DiscoverState;
  scanning: boolean;
  onRescan: () => void;
}) {
  const [selected, setSelected] = useState<string | null>(null);

  const entries = useMemo<RadarEntry[]>(() => {
    const visitorByIp = new Map<string, Visitor>();
    for (const visitor of state.visitors) {
      if (visitor.ip) visitorByIp.set(visitor.ip, visitor);
    }

    const rows: RadarEntry[] = state.devices.map((device) => {
      const visitor = visitorByIp.get(device.ip);
      const kind: RadarEntry["kind"] = device.is_this_host
        ? "self"
        : device.is_member
          ? "member"
          : visitor
            ? "visitor"
            : "idle";

      // A device that has announced itself names itself far better than reverse
      // DNS ever will, so prefer that label when it exists.
      const label = visitor?.name || device.hostname || device.ip;
      const sub = visitor?.gpu || device.hostname || device.mac || device.ip;

      return {
        id: device.ip,
        label,
        sub,
        kind,
        proximity: device.proximity,
        rtt: device.rtt_ms,
        device,
        visitor,
      };
    });

    // A visitor whose address never showed up in the sweep still deserves a
    // seat: it is a real device that is demonstrably reachable.
    const seen = new Set(rows.map((row) => row.id));
    for (const visitor of state.visitors) {
      const id = visitor.ip || visitor.visitor_id;
      if (seen.has(id)) continue;
      rows.push({
        id,
        label: visitor.name,
        sub: visitor.gpu || visitor.platform || "browser",
        kind: "visitor",
        proximity: "unknown",
        rtt: null,
        visitor,
      });
    }

    return rows;
  }, [state]);

  const active = entries.find((entry) => entry.id === selected) || null;
  const orbiting = entries.filter((entry) => entry.kind !== "self");
  const self = entries.find((entry) => entry.kind === "self");

  return (
    <div className="radar-wrap">
      <div className={`radar${scanning ? " is-scanning" : ""}`}>
        {(["close", "nearby", "far", "unknown"] as const).map((ring) => (
          <div
            key={ring}
            className={`radar-ring radar-ring-${ring}`}
            style={{ inset: `${(1 - RING_RADIUS[ring]) * 50}%` }}
          >
            <span className="radar-ring-label">{RING_LABEL[ring]}</span>
          </div>
        ))}

        <div className="radar-sweep" />

        <button
          type="button"
          className="radar-core"
          onClick={() => setSelected(self ? self.id : null)}
          title={self ? `This host, ${self.label}` : "This host"}
        >
          <span className="radar-core-dot" />
          <span className="radar-core-label">This host</span>
        </button>

        {orbiting.map((entry) => {
          const turn = angleFor(entry.id);
          const radius = RING_RADIUS[entry.proximity] * 50;
          const x = 50 + Math.cos(turn * Math.PI * 2) * radius;
          const y = 50 + Math.sin(turn * Math.PI * 2) * radius;

          return (
            <button
              type="button"
              key={entry.id}
              className={`radar-node is-${entry.kind}${selected === entry.id ? " is-selected" : ""}`}
              style={{ left: `${x}%`, top: `${y}%` }}
              onClick={() => setSelected(selected === entry.id ? null : entry.id)}
              title={`${entry.label} · ${entry.rtt != null ? `${entry.rtt} ms` : "distance unknown"}`}
            >
              <span className="radar-avatar">{initials(entry.label)}</span>
              <span className="radar-node-label">{entry.label}</span>
            </button>
          );
        })}

        {orbiting.length === 0 ? (
          <div className="radar-empty">
            <p className="small faint">Nothing else answered on this network yet.</p>
          </div>
        ) : null}
      </div>

      <div className="radar-side">
        <div className="row-between">
          <span className="panel-title">
            {orbiting.length} device{orbiting.length === 1 ? "" : "s"} around you
          </span>
          <button className="btn btn-sm" type="button" onClick={onRescan} disabled={scanning}>
            {scanning ? "Sweeping…" : "Sweep again"}
          </button>
        </div>

        <div className="radar-legend">
          <span className="radar-key">
            <i className="radar-swatch is-member" /> Contributing
          </span>
          <span className="radar-key">
            <i className="radar-swatch is-visitor" /> Looking
          </span>
          <span className="radar-key">
            <i className="radar-swatch is-idle" /> Idle
          </span>
        </div>

        {active ? (
          <div className="radar-detail">
            <div className="row-between" style={{ marginBottom: 10 }}>
              <strong className="truncate">{active.label}</strong>
              <button className="btn btn-ghost btn-sm" onClick={() => setSelected(null)} type="button">
                Close
              </button>
            </div>

            <div className="node-facts">
              <div className="node-fact">
                <span>Address</span>
                <span>{active.device?.ip || "unknown"}</span>
              </div>
              <div className="node-fact">
                <span>Distance</span>
                <span>{active.rtt != null ? `${active.rtt} ms` : "not measurable"}</span>
              </div>
              <div className="node-fact">
                <span>Status</span>
                <span>
                  {active.kind === "member"
                    ? "Contributing"
                    : active.kind === "visitor"
                      ? "Join page open"
                      : active.kind === "self"
                        ? "This host"
                        : "Idle"}
                </span>
              </div>
              <div className="node-fact">
                <span>Graphics</span>
                <span className="truncate" title={active.visitor?.gpu || undefined}>
                  {active.visitor?.gpu || "unreported"}
                </span>
              </div>
              {active.visitor?.cores ? (
                <div className="node-fact">
                  <span>CPU cores</span>
                  <span>{active.visitor.cores}</span>
                </div>
              ) : null}
              {active.device?.mac ? (
                <div className="node-fact">
                  <span>Hardware</span>
                  <span className="truncate">{active.device.mac}</span>
                </div>
              ) : null}
            </div>

            {active.kind === "idle" ? (
              <p className="small faint" style={{ marginTop: 12 }}>
                This device is on the network but not participating. Open the join address on it to
                bring it in. There is no way to enrol a machine remotely, and there should not be.
              </p>
            ) : null}
            {active.kind === "visitor" && !active.visitor?.has_agent ? (
              <p className="small faint" style={{ marginTop: 12 }}>
                Someone has the join page open here. They still need to run the one-line command for
                the GPU to be usable.
              </p>
            ) : null}
          </div>
        ) : (
          <p className="small faint">
            Rings are measured network round trip, which tracks link quality rather than metres. A
            device that answers on no port cannot be measured and sits on the outer ring.
          </p>
        )}
      </div>
    </div>
  );
}
