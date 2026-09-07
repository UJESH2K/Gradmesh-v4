"use client";

import { useEffect, useState } from "react";

/**
 * Tells the host "a device is here, looking at this page".
 *
 * This closes the discovery loop from the other direction. The host's Discover
 * page lists machines on the network, but a bare IP address and a MAC are hard
 * to match to a physical laptop. A device that has opened the join page can
 * name itself, so the host sees "Aadit's laptop, Windows, 16 cores, WebGPU via
 * NVIDIA" instead of "192.168.1.46, unknown".
 *
 * Nothing here is installed and nothing is persisted beyond a random id in
 * localStorage so the same device is not counted twice. Everything reported is
 * what the browser already hands to any page it loads.
 */

const ANNOUNCE_INTERVAL_MS = 30000;
const STORAGE_KEY = "gradmesh.visitor";

export type VisitorProfile = {
  cores: number | null;
  memoryGb: number | null;
  gpu: string | null;
  webgpu: boolean;
  platform: string;
};

function visitorId(): string {
  try {
    const existing = localStorage.getItem(STORAGE_KEY);
    if (existing) return existing;
    const generated = crypto.randomUUID().replace(/-/g, "").slice(0, 24);
    localStorage.setItem(STORAGE_KEY, generated);
    return generated;
  } catch {
    // Private browsing blocks storage. A per-tab id still beats nothing.
    return Math.random().toString(36).slice(2, 14).padEnd(12, "0");
  }
}

/** What the browser can tell us about the machine, including WebGPU. */
export async function readProfile(): Promise<VisitorProfile> {
  const nav = navigator as Navigator & {
    deviceMemory?: number;
    gpu?: {
      requestAdapter: (options?: unknown) => Promise<{
        info?: { vendor?: string; architecture?: string; device?: string; description?: string };
      } | null>;
    };
  };

  let gpu: string | null = null;
  let webgpu = false;

  try {
    if (nav.gpu) {
      const adapter = await nav.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (adapter) {
        webgpu = true;
        const info = adapter.info;
        const parts = [info?.vendor, info?.architecture, info?.device, info?.description]
          .filter(Boolean)
          .map((part) => String(part).trim());
        gpu = parts.length ? Array.from(new Set(parts)).join(" ") : "WebGPU adapter";
      }
    }
  } catch {
    // WebGPU is unavailable or blocked. Not an error worth showing anyone.
  }

  if (!gpu) {
    // WebGL's debug renderer string is the fallback, and it usually names the
    // actual card even when WebGPU is off.
    try {
      const canvas = document.createElement("canvas");
      const context = (canvas.getContext("webgl2") ||
        canvas.getContext("webgl")) as WebGLRenderingContext | null;
      const debugInfo = context?.getExtension("WEBGL_debug_renderer_info");
      if (context && debugInfo) {
        const renderer = context.getParameter(
          (debugInfo as { UNMASKED_RENDERER_WEBGL: number }).UNMASKED_RENDERER_WEBGL
        );
        if (renderer) gpu = String(renderer);
      }
    } catch {
      // Nothing to report.
    }
  }

  const platform =
    (nav as Navigator & { userAgentData?: { platform?: string } }).userAgentData?.platform ||
    navigator.platform ||
    "unknown";

  return {
    cores: navigator.hardwareConcurrency ?? null,
    memoryGb: nav.deviceMemory ?? null,
    gpu,
    webgpu,
    platform,
  };
}

export default function VisitorBeacon({
  onProfile,
  coordinatorUrl,
}: {
  onProfile?: (profile: VisitorProfile) => void;
  coordinatorUrl?: string;
}) {
  const [, setAnnounced] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setInterval> | null = null;

    (async () => {
      const profile = await readProfile();
      if (cancelled) return;
      onProfile?.(profile);

      const body = {
        visitor_id: visitorId(),
        name: deviceName(profile),
        platform: profile.platform,
        user_agent: navigator.userAgent.slice(0, 300),
        cores: profile.cores,
        memory_gb: profile.memoryGb,
        gpu: profile.gpu,
        webgpu: profile.webgpu,
        screen: `${window.screen.width}x${window.screen.height}`,
      };

      // Announce straight to the coordinator rather than through the Next
      // proxy. A proxied request arrives from the host itself, so every visitor
      // would be recorded as 127.0.0.1 and could not be matched to a row in the
      // network scan. The coordinator accepts an untokened beacon from its own
      // subnet precisely so this can work. The proxy stays as a fallback for
      // the case where port 8000 is not reachable from the visitor.
      const direct = coordinatorUrl ? `${coordinatorUrl}/visitors` : null;

      const announce = () => {
        const payload = {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          keepalive: true,
        } satisfies RequestInit;

        const attempt = direct ? fetch(direct, payload) : Promise.reject(new Error("no direct url"));

        attempt
          .then((response) => {
            if (!response.ok) throw new Error(String(response.status));
            setAnnounced(true);
          })
          .catch(() => fetch("/api/visitor", payload).catch(() => {}));
      };

      announce();
      timer = setInterval(announce, ANNOUNCE_INTERVAL_MS);
    })();

    return () => {
      cancelled = true;
      if (timer) clearInterval(timer);
    };
  }, [onProfile, coordinatorUrl]);

  return null;
}

/** A name a human would recognise, built from what the browser exposes. */
function deviceName(profile: VisitorProfile): string {
  const ua = navigator.userAgent;
  const device = /iPhone/.test(ua)
    ? "iPhone"
    : /iPad/.test(ua)
      ? "iPad"
      : /Android/.test(ua)
        ? "Android device"
        : /Macintosh/.test(ua)
          ? "Mac"
          : /Windows/.test(ua)
            ? "Windows PC"
            : /Linux/.test(ua)
              ? "Linux machine"
              : "Device";

  const browser = /Edg\//.test(ua)
    ? "Edge"
    : /Chrome\//.test(ua)
      ? "Chrome"
      : /Firefox\//.test(ua)
        ? "Firefox"
        : /Safari\//.test(ua)
          ? "Safari"
          : "browser";

  return `${device} · ${browser}${profile.cores ? ` · ${profile.cores} cores` : ""}`;
}
