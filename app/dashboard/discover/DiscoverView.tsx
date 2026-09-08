"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import CopyLine from "@/components/CopyLine";
import DeviceRadar from "@/components/dashboard/DeviceRadar";
import { useMesh } from "@/components/dashboard/MeshProvider";
import { Empty, Panel, StatTile } from "@/components/dashboard/ui";
import { ago } from "@/lib/format";
import type { DiscoverState } from "@/lib/types";

export default function DiscoverView({
  canManage,
  origin,
}: {
  canManage: boolean;
  origin: string;
}) {
  const { request, events } = useMesh();
  const [state, setState] = useState<DiscoverState | null>(null);
  const [scanning, setScanning] = useState(false);
  const [view, setView] = useState<"radar" | "list">("radar");
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(
    async (refresh = false) => {
      try {
        setState(
          await request<DiscoverState>(`/api/mesh/discover${refresh ? "?refresh=true" : ""}`)
        );
        setError(null);
      } catch (cause) {
        setError((cause as Error).message);
      }
    },
    [request]
  );

  useEffect(() => {
    void load();
  }, [load]);

  // A visitor announcing itself is the event this page exists to show, so it
  // refreshes on the stream rather than only on a timer.
  useEffect(() => {
    const latest = events.at(-1);
    if (latest && ["visitor.arrived", "network.scanned", "node.joined", "node.left"].includes(latest.kind)) {
      void load();
    }
  }, [events, load]);

  useEffect(() => {
    const timer = setInterval(() => void load(), 15000);
    return () => clearInterval(timer);
  }, [load]);

  async function rescan() {
    setScanning(true);
    setError(null);
    try {
      await request("/api/mesh/discover/scan", { method: "POST" });
      await load();
    } catch (cause) {
      setError((cause as Error).message);
    } finally {
      setScanning(false);
    }
  }

  if (!state) {
    return (
      <>
        <h1 className="page-title">Discover devices</h1>
        <div className="panel">
          <Empty>{error || "Sweeping the network…"}</Empty>
        </div>
      </>
    );
  }

  const { mdns, addresses, scan, devices, visitors, nodes } = state;
  const friendlyOrigin = mdns.active
    ? `http://${mdns.hostname}:${addresses.web_port}`
    : `http://${addresses.lan_ip}:${addresses.web_port}`;
  const ipOrigin = `http://${addresses.lan_ip}:${addresses.web_port}`;

  const contributing = devices.filter((device) => device.is_member && !device.is_this_host);
  const looking = devices.filter((device) => device.is_visitor && !device.is_member);
  const idle = devices.filter(
    (device) => !device.is_this_host && !device.is_member && !device.is_visitor
  );

  return (
    <>
      <div className="row-between">
        <div>
          <h1 className="page-title">Discover devices</h1>
          <p className="small faint" style={{ marginTop: 4 }}>
            Everything on {scan.subnet || "this network"}, and the address other devices should
            open to find this host.
          </p>
        </div>
        <div className="row" style={{ gap: 8 }}>
          <span className="small faint">
            {scan.at ? `Swept ${ago(scan.at)} in ${scan.duration_seconds}s` : "Not swept yet"}
          </span>
          <button
            className={`btn btn-sm${view === "radar" ? " btn-primary" : ""}`}
            type="button"
            onClick={() => setView("radar")}
          >
            Radar
          </button>
          <button
            className={`btn btn-sm${view === "list" ? " btn-primary" : ""}`}
            type="button"
            onClick={() => setView("list")}
          >
            List
          </button>
          <button className="btn btn-sm" type="button" onClick={rescan} disabled={scanning}>
            {scanning ? "Sweeping…" : "Sweep network"}
          </button>
        </div>
      </div>

      {error ? <div className="notice notice-danger">{error}</div> : null}

      <div className="grid grid-4">
        <StatTile label="Devices on network" value={devices.length} foot={scan.subnet || "—"} />
        <StatTile
          label="Contributing"
          value={nodes.filter((node) => node.active).length}
          foot="running the agent"
          accent={nodes.some((node) => node.active)}
        />
        <StatTile
          label="Looking right now"
          value={looking.length}
          foot="join page open, no agent yet"
        />
        <StatTile label="Idle" value={idle.length} foot="seen but not participating" />
      </div>

      {view === "radar" ? (
        <DeviceRadar state={state} scanning={scanning} onRescan={rescan} />
      ) : null}

      <Panel title="How other devices reach this host">
        <div className="stack">
          {mdns.active ? (
            <>
              <div className="notice notice-accent">
                <strong>No IP address needed.</strong> This host answers to{" "}
                <code className="code-inline">{mdns.hostname}</code> over multicast DNS. Any device
                on this network can open the address below directly. Works on Windows 10 and later,
                macOS, iOS, Android 12 and later, and most desktop Linux.
              </div>
              <CopyLine value={friendlyOrigin} label="Type this on the other device" />
              <CopyLine
                tone="muted"
                value={ipOrigin}
                label="Fallback, if the name does not resolve on an older device"
              />
            </>
          ) : (
            <>
              <div className="notice notice-warn">
                Multicast DNS is not advertising, so devices need the address rather than a name.
                {mdns.error ? ` ${mdns.error}` : ""}
              </div>
              <CopyLine value={ipOrigin} label="Type this on the other device" />
            </>
          )}

          <div className="row wrap" style={{ gap: 8 }}>
            <Link className="btn btn-sm" href="/dashboard/invite">
              Get the join command
            </Link>
            <a
              className="btn btn-sm"
              href={`${origin}/join`}
              target="_blank"
              rel="noreferrer"
            >
              Open the join page
            </a>
          </div>
        </div>
      </Panel>

      {looking.length > 0 || visitors.length > 0 ? (
        <Panel title={`Devices looking at the join page (${visitors.length})`} flush>
          <div className="table-scroll">
            <table className="table">
              <thead>
                <tr>
                  <th>Device</th>
                  <th>Address</th>
                  <th>Graphics</th>
                  <th className="num">Cores</th>
                  <th className="num">Memory</th>
                  <th className="num">Seen</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {visitors.map((visitor) => (
                  <tr key={visitor.visitor_id}>
                    <td>
                      <div className="truncate">{visitor.name}</div>
                      <div className="small faint truncate">{visitor.platform || "unknown"}</div>
                    </td>
                    <td className="mono small">{visitor.ip || "—"}</td>
                    <td className="small">
                      <div className="truncate" title={visitor.gpu || undefined}>
                        {visitor.gpu || "not reported"}
                      </div>
                      {visitor.webgpu ? (
                        <span className="badge badge-cyan" style={{ marginTop: 4 }}>
                          WebGPU
                        </span>
                      ) : null}
                    </td>
                    <td className="num">{visitor.cores ?? "—"}</td>
                    <td className="num">{visitor.memory_gb ? `${visitor.memory_gb} GB` : "—"}</td>
                    <td className="num small faint">{ago(visitor.last_seen)}</td>
                    <td>
                      {visitor.has_agent ? (
                        <span className="badge badge-live">
                          <span className="dot dot-live" />
                          Contributing
                        </span>
                      ) : (
                        <span className="badge badge-warn">Agent not installed</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="panel-body" style={{ borderTop: "1px solid var(--line)" }}>
            <p className="small faint">
              These devices have the join page open. A browser tab cannot train a model on its own,
              so each still needs the one-line command to contribute its GPU. This list exists so
              you can tell which physical machine is which before walking over to it.
            </p>
          </div>
        </Panel>
      ) : null}

      <Panel
        title={`Everything on the network (${devices.length})`}
        flush
        hidden={view === "radar"}
      >
        {devices.length === 0 ? (
          <Empty>Nothing found. Press Sweep network, or check that this host is on Wi-Fi.</Empty>
        ) : (
          <div className="table-scroll">
            <table className="table">
              <thead>
                <tr>
                  <th>Address</th>
                  <th>Name</th>
                  <th>Hardware address</th>
                  <th>Open ports</th>
                  <th>Role</th>
                </tr>
              </thead>
              <tbody>
                {devices.map((device) => (
                  <tr key={device.ip}>
                    <td className="mono">{device.ip}</td>
                    <td className="small">{device.hostname || <span className="faint">unnamed</span>}</td>
                    <td className="mono small faint">{device.mac || "—"}</td>
                    <td className="small mono faint">
                      {device.open_ports.length ? device.open_ports.join(", ") : "none"}
                    </td>
                    <td>
                      {device.is_this_host ? (
                        <span className="badge badge-accent">This host</span>
                      ) : device.is_member ? (
                        <span className="badge badge-live">
                          <span className="dot dot-live" />
                          Contributing
                        </span>
                      ) : device.is_visitor ? (
                        <span className="badge badge-cyan">Looking</span>
                      ) : (
                        <span className="badge">Idle</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      {canManage && idle.length > 0 ? (
        <Panel title="Bring an idle device in">
          <div className="stack">
            <p className="small muted">
              {idle.length} device{idle.length === 1 ? "" : "s"} on this network{" "}
              {idle.length === 1 ? "is" : "are"} not participating. There is no way to enrol a
              machine remotely, and there should not be. Open the address below on that device and
              it walks itself through the rest.
            </p>
            <CopyLine value={`${friendlyOrigin}/join`} label="Open this on the other device" />
          </div>
        </Panel>
      ) : null}
    </>
  );
}
