"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import type { MeshEvent, MeshState } from "@/lib/types";

type SetupState = {
  controlPlane: string;
  trainingPlane: string;
  models: string[];
  backend: string | null;
  messages: { at: number; message: string }[];
  coordinator?: string;
};

type MeshContextValue = {
  mesh: MeshState | null;
  events: MeshEvent[];
  setup: SetupState | null;
  connected: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  request: <T>(path: string, init?: RequestInit) => Promise<T>;
};

const MeshContext = createContext<MeshContextValue | null>(null);

const REFRESH_DEBOUNCE_MS = 220;
const SLOW_POLL_MS = 10000;

/**
 * Live mesh state for the whole dashboard.
 *
 * One Server-Sent Events connection per tab drives everything. An event tells
 * us *that* something changed; we then refetch the single mesh snapshot rather
 * than trying to apply deltas, which keeps the client from ever disagreeing
 * with the coordinator. A slow poll runs underneath as a safety net for the
 * case where the stream drops without firing an error.
 */
export function MeshProvider({ children }: { children: React.ReactNode }) {
  const [mesh, setMesh] = useState<MeshState | null>(null);
  const [events, setEvents] = useState<MeshEvent[]>([]);
  const [setup, setSetup] = useState<SetupState | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const debounce = useRef<ReturnType<typeof setTimeout> | null>(null);
  const inFlight = useRef(false);

  const request = useCallback(async <T,>(path: string, init?: RequestInit): Promise<T> => {
    const response = await fetch(path, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload?.detail || `Request failed with ${response.status}`);
    }
    return payload as T;
  }, []);

  const refresh = useCallback(async () => {
    if (inFlight.current) return;
    inFlight.current = true;
    try {
      const next = await request<MeshState>("/api/mesh/mesh");
      setMesh(next);
      setError(null);
    } catch (cause) {
      setError((cause as Error).message);
    } finally {
      inFlight.current = false;
    }
  }, [request]);

  const scheduleRefresh = useCallback(() => {
    if (debounce.current) clearTimeout(debounce.current);
    debounce.current = setTimeout(() => {
      void refresh();
    }, REFRESH_DEBOUNCE_MS);
  }, [refresh]);

  useEffect(() => {
    void refresh();
    request<SetupState>("/api/setup").then(setSetup).catch(() => {});
  }, [refresh, request]);

  useEffect(() => {
    const source = new EventSource("/api/stream?replay=40");

    source.onopen = () => setConnected(true);
    source.onerror = () => setConnected(false);

    const onMessage = (event: MessageEvent<string>) => {
      try {
        const parsed = JSON.parse(event.data) as MeshEvent;
        if (parsed.kind === "stream.ready") {
          setConnected(true);
          return;
        }
        setEvents((previous) => [...previous.slice(-199), parsed]);
        scheduleRefresh();
      } catch {
        // A malformed frame is not worth tearing the stream down for.
      }
    };

    // The coordinator names every event, so listeners are registered per kind
    // rather than relying on the default `message` type.
    const kinds = [
      "coordinator.ready",
      "node.joined",
      "node.offline",
      "node.left",
      "node.evicted",
      "run.created",
      "run.waiting",
      "run.stopped",
      "run.failed",
      "run.completed",
      "round.started",
      "round.completed",
      "round.failed",
      "shard.assigned",
      "shard.completed",
      "shard.failed",
      "shard.dropped",
      "shard.speculated",
      "shard.superseded",
      "dataset.added",
      "dataset.removed",
      "dataset.default",
      "policy.updated",
      "mesh.token_rotated",
      "visitor.arrived",
      "network.scanned",
      "supervisor.error",
      "stream.ready",
    ];
    for (const kind of kinds) source.addEventListener(kind, onMessage as EventListener);
    source.addEventListener("message", onMessage as EventListener);

    return () => {
      for (const kind of kinds) source.removeEventListener(kind, onMessage as EventListener);
      source.removeEventListener("message", onMessage as EventListener);
      source.close();
    };
  }, [scheduleRefresh]);

  useEffect(() => {
    const timer = setInterval(() => {
      void refresh();
      request<SetupState>("/api/setup").then(setSetup).catch(() => {});
    }, SLOW_POLL_MS);
    return () => clearInterval(timer);
  }, [refresh, request]);

  const value = useMemo<MeshContextValue>(
    () => ({ mesh, events, setup, connected, error, refresh, request }),
    [mesh, events, setup, connected, error, refresh, request]
  );

  return <MeshContext.Provider value={value}>{children}</MeshContext.Provider>;
}

export function useMesh(): MeshContextValue {
  const value = useContext(MeshContext);
  if (!value) throw new Error("useMesh must be used inside MeshProvider");
  return value;
}
