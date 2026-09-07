import QRCode from "qrcode";

import CopyLine from "./CopyLine";

const COORDINATOR_PORT = Number(process.env.GRADMESH_COORDINATOR_PORT || 8000);

/**
 * Everything a contributor needs, on one screen.
 *
 * The commands are generated from the address the visitor actually reached this
 * page on, so what they copy is guaranteed to be routable from where they are
 * standing. That is the failure mode of every "find the host IP with ipconfig"
 * instruction.
 */
export default async function JoinPanel({
  origin,
  token,
  compact = false,
}: {
  origin: string;
  token: string;
  compact?: boolean;
}) {
  const hostname = new URL(origin).hostname;
  const windowsCommand = `irm ${origin}/join.ps1 | iex`;
  const unixCommand = `curl -fsSL ${origin}/join.sh | sh`;
  const qr = await QRCode.toString(`${origin}/join`, {
    type: "svg",
    margin: 1,
    width: 168,
    color: { dark: "#e9edf1", light: "#00000000" },
  });

  return (
    <div className="stack">
      <div className="grid" style={{ gridTemplateColumns: compact ? "1fr" : "minmax(0, 1fr) auto" }}>
        <div className="stack">
          <CopyLine value={windowsCommand} label="Windows — PowerShell" />
          <CopyLine value={unixCommand} label="macOS and Linux — Terminal" />
        </div>
        {!compact ? (
          <div className="stack-sm" style={{ alignItems: "center" }}>
            <div
              aria-label="QR code to this join page"
              dangerouslySetInnerHTML={{ __html: qr }}
              style={{ lineHeight: 0 }}
            />
            <span className="small faint">Scan to open this page</span>
          </div>
        ) : null}
      </div>

      <div className="divider" />

      <div className="grid grid-2">
        <div className="stack-sm">
          <span className="eyebrow">What the command does</span>
          <ul className="small muted" style={{ margin: 0, paddingLeft: 18, lineHeight: 1.75 }}>
            <li>Finds Python, and tells you how to install it if it is missing.</li>
            <li>Creates a private environment under your home folder. Nothing else is touched.</li>
            <li>Detects your GPU and installs the matching PyTorch build.</li>
            <li>Measures what your device can actually do, then joins the mesh.</li>
          </ul>
        </div>

        <div className="stack-sm">
          <span className="eyebrow">Connection details</span>
          <div className="small mono muted stack-sm" style={{ gap: 4 }}>
            <div>
              host <span className="accent">{hostname}</span>
            </div>
            <div>
              dashboard <span className="accent">{origin}</span>
            </div>
            <div>
              coordinator <span className="accent">{`http://${hostname}:${COORDINATOR_PORT}`}</span>
            </div>
          </div>
          <span className="hint">
            Leaving is Ctrl+C in that terminal. Your machine drops out of the mesh immediately and
            the round replans without it.
          </span>
        </div>
      </div>

      {!compact ? (
        <details>
          <summary className="small faint" style={{ cursor: "pointer" }}>
            Prefer to run it yourself instead of piping a script?
          </summary>
          <div className="stack-sm" style={{ marginTop: 12 }}>
            <p className="small muted">
              Clone the repository on your machine and point the worker at this host. The token
              below is what authorises a machine to receive dataset shards, so share it only with
              people you want on the mesh.
            </p>
            <CopyLine
              tone="muted"
              value={`npm run worker -- --server http://${hostname}:${COORDINATOR_PORT} --token ${token}`}
            />
          </div>
        </details>
      ) : null}
    </div>
  );
}
