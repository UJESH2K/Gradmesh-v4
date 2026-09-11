"use client";

import { useCallback, useEffect, useState } from "react";

import { useMesh } from "@/components/dashboard/MeshProvider";
import { Panel } from "@/components/dashboard/ui";
import type { StandardCatalogue } from "@/lib/types";

/**
 * One-click import of a standard detection dataset.
 *
 * Uploading your own zip is still the right move for your own problem. This
 * exists because a paper wants a dataset a reader already knows, with published
 * numbers to compare against, and because hunting for a download link is a poor
 * use of an afternoon.
 */
export default function DatasetImporter({
  canManage,
  onImported,
}: {
  canManage: boolean;
  onImported: () => void;
}) {
  const { request, events } = useMesh();
  const [catalogue, setCatalogue] = useState<StandardCatalogue | null>(null);
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);

  const load = useCallback(async () => {
    try {
      setCatalogue(await request<StandardCatalogue>("/api/mesh/datasets/standard"));
    } catch (cause) {
      setError((cause as Error).message);
    }
  }, [request]);

  useEffect(() => {
    if (open) void load();
  }, [open, load]);

  useEffect(() => {
    const latest = events.at(-1);
    if (!latest || !latest.kind.startsWith("dataset.")) return;
    void load();
    if (latest.kind === "dataset.added") {
      setBusyKey(null);
      onImported();
    }
    if (latest.kind === "dataset.import_failed") {
      setBusyKey(null);
      setError(String(latest.data.error || "Import failed."));
    }
  }, [events, load, onImported]);

  async function start(key: string) {
    setBusyKey(key);
    setError(null);
    try {
      await request("/api/mesh/datasets/standard", {
        method: "POST",
        body: JSON.stringify({ key, make_default: true }),
      });
    } catch (cause) {
      setError((cause as Error).message);
      setBusyKey(null);
    }
  }

  const importing = catalogue?.import?.running ? catalogue.import : null;

  return (
    <Panel
      title="Standard datasets"
      action={
        <button className="btn btn-sm" type="button" onClick={() => setOpen((value) => !value)}>
          {open ? "Hide" : "Browse"}
        </button>
      }
    >
      {!open ? (
        <p className="small faint">
          Import a dataset with published baselines instead of uploading a zip. Fetched on demand
          and registered automatically.
        </p>
      ) : (
        <div className="stack">
          {error ? <div className="notice notice-danger">{error}</div> : null}
          {importing ? (
            <div className="notice notice-accent">
              <strong>Downloading {importing.key}.</strong> {importing.message || "Working"}. This
              can take a long time on a large set. It keeps going if you leave this page.
            </div>
          ) : null}

          <div className="dataset-cards">
            {(catalogue?.catalogue ?? []).map((entry) => (
              <div key={entry.key} className="dataset-card">
                <div className="row-between" style={{ gap: 8 }}>
                  <strong>{entry.name}</strong>
                  {entry.already_downloaded ? (
                    <span className="badge badge-accent">on disk</span>
                  ) : (
                    <span className="badge">
                      {entry.download_mb >= 1000
                        ? `${(entry.download_mb / 1000).toFixed(1)} GB`
                        : `${entry.download_mb} MB`}
                    </span>
                  )}
                </div>
                <p className="small faint" style={{ marginTop: 6 }}>
                  {entry.blurb}
                </p>
                <div className="row-between" style={{ marginTop: 10 }}>
                  <span className="small faint">
                    {entry.images.toLocaleString()} images · {entry.classes} classes
                  </span>
                  <button
                    className="btn btn-sm"
                    type="button"
                    disabled={!canManage || Boolean(busyKey) || Boolean(importing)}
                    onClick={() => void start(entry.key)}
                  >
                    {busyKey === entry.key ? "Starting…" : entry.already_downloaded ? "Register" : "Import"}
                  </button>
                </div>
                <p className="small faint" style={{ marginTop: 6 }}>
                  Best for {entry.good_for}
                </p>
              </div>
            ))}
          </div>

          <div className="notice">
            <strong>Not ImageNet.</strong> It is a classification dataset: one label per image and
            no bounding boxes, so there is nothing for a detection model to learn and nothing to
            convert. Its detection subset has boxes but runs to about 150 GB, ships XML that needs
            converting, and sits behind a signed agreement, so it cannot be fetched by a script.
            COCO is the citable standard with boxes; VisDrone is the practical choice for a scaling
            ladder.
          </div>
        </div>
      )}
    </Panel>
  );
}
