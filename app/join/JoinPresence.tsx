"use client";

import { useState } from "react";

import VisitorBeacon, { type VisitorProfile } from "@/components/VisitorBeacon";

/**
 * Shows the visitor what this device actually is, and what it can and cannot do
 * from a browser tab.
 *
 * Being straight about the limit here is worth more than pretending: a tab can
 * see the GPU through WebGPU, but it cannot load a PyTorch checkpoint or write
 * CUDA kernels, so it cannot train the model. Saying so, next to the numbers
 * that prove the device was detected, makes the one-line install feel like the
 * obvious next step rather than an unexplained hurdle.
 */
export default function JoinPresence({ coordinatorUrl }: { coordinatorUrl: string }) {
  const [profile, setProfile] = useState<VisitorProfile | null>(null);

  return (
    <>
      <VisitorBeacon onProfile={setProfile} coordinatorUrl={coordinatorUrl} />
      {profile ? (
        <div className="panel" style={{ marginTop: 26 }}>
          <div className="row-between" style={{ marginBottom: 14 }}>
            <span className="panel-title">This device</span>
            <span className="badge badge-live">
              <span className="dot dot-live" />
              Visible to the host
            </span>
          </div>

          <div className="node-facts">
            <div className="node-fact">
              <span>Platform</span>
              <span>{profile.platform}</span>
            </div>
            <div className="node-fact">
              <span>CPU cores</span>
              <span>{profile.cores ?? "unknown"}</span>
            </div>
            <div className="node-fact">
              <span>Memory</span>
              <span>{profile.memoryGb ? `${profile.memoryGb} GB or more` : "unknown"}</span>
            </div>
            <div className="node-fact">
              <span>Graphics</span>
              <span className="truncate" title={profile.gpu || undefined}>
                {profile.gpu || "not reported"}
              </span>
            </div>
          </div>

          <p className="small faint" style={{ marginTop: 16 }}>
            The host can already see this device on the network. A browser tab cannot load the
            training runtime, so the command below is what actually lends the GPU. It installs
            nothing outside your home folder and stops the moment you close the terminal.
          </p>

          <div className="notice" style={{ marginTop: 14 }}>
            <strong>If this machine has no supported GPU</strong>, the command still works. The
            agent measures the device, reports honestly, and the mesh marks it as ineligible with
            the reason shown. It will not be sent training work and will not slow anyone down. A
            browser can only guess at graphics hardware, so the agent is the one that decides.
          </div>
        </div>
      ) : null}
    </>
  );
}
