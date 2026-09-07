"use client";

import { useEffect, useRef } from "react";
import Link from "next/link";

import CopyLine from "./CopyLine";
import Logo from "./Logo";
import MeshCanvas from "./MeshCanvas";

type Props = {
  origin: string;
  signedIn: boolean;
  needsFirstAccount: boolean;
  meshName: string;
};

const FEATURES = [
  {
    title: "Measured, not trusted",
    body: "Every machine runs a throughput probe before it can hold a shard. The coordinator ranks devices on what they actually deliver, not on a name string.",
  },
  {
    title: "Shards sized to the device",
    body: "A round costs whatever the slowest worker costs. So shards are sized in proportion to measured throughput, and every worker is predicted to finish at the same instant.",
  },
  {
    title: "Stragglers do not stall rounds",
    body: "Each shard carries a deadline derived from its own prediction. A soft miss clones the work onto an idle peer. A hard miss releases the barrier without it.",
  },
  {
    title: "Aggregation that respects contribution",
    body: "Unequal shards need weighted FedAvg. Each update is weighted by the samples behind it, damped by that worker's track record.",
  },
  {
    title: "One line to contribute",
    body: "A contributor pastes a single command. It finds their accelerator, installs the right PyTorch build, and joins. No repository to clone, no requirements to read.",
  },
  {
    title: "Your data stays on your network",
    body: "Shards, checkpoints and gradients move over your LAN between machines you can see. Nothing leaves for a third-party cluster.",
  },
];

const STEPS = [
  { title: "Start the mesh", body: "Run one command on the machine holding your dataset. The coordinator and the dashboard come up together." },
  { title: "Share the link", body: "Anyone on the same Wi-Fi opens the join page and pastes one line. Their GPU appears in the dashboard within seconds." },
  { title: "Upload a dataset", body: "Drop in a YOLO export. It becomes the mesh default, and every machine that joins later trains against it." },
  { title: "Watch it train", body: "Live shard timings, per-device throughput, straggler decisions and speedup, streamed as they happen." },
];

export default function Landing({ origin, signedIn, needsFirstAccount, meshName }: Props) {
  const rootRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    let cleanup = () => {};
    let cancelled = false;

    (async () => {
      const [{ gsap }, { ScrollTrigger }] = await Promise.all([
        import("gsap"),
        import("gsap/ScrollTrigger"),
      ]);
      if (cancelled) return;

      gsap.registerPlugin(ScrollTrigger);

      const context = gsap.context(() => {
        // Hero: staggered entrance, no scroll trigger, it is above the fold.
        gsap.from("[data-hero]", {
          y: 26,
          opacity: 0,
          duration: 0.85,
          ease: "power3.out",
          stagger: 0.09,
        });

        // Everything below reveals as it enters the viewport.
        gsap.utils.toArray<HTMLElement>(".js-reveal").forEach((element) => {
          gsap.from(element, {
            y: 30,
            opacity: 0,
            duration: 0.7,
            ease: "power2.out",
            scrollTrigger: { trigger: element, start: "top 86%", once: true },
          });
        });

        gsap.utils.toArray<HTMLElement>("[data-stagger]").forEach((group) => {
          gsap.from(group.children, {
            y: 24,
            opacity: 0,
            duration: 0.6,
            ease: "power2.out",
            stagger: 0.07,
            scrollTrigger: { trigger: group, start: "top 84%", once: true },
          });
        });

        // The two schedules animate their bars, so the difference between an
        // even split and a proportional one is something you watch happen.
        gsap.utils.toArray<HTMLElement>("[data-bar]").forEach((bar) => {
          gsap.from(bar, {
            scaleX: 0,
            duration: 1.05,
            ease: "power2.inOut",
            scrollTrigger: { trigger: bar.closest(".split-demo"), start: "top 74%", once: true },
          });
        });

        // Nav gains a hairline once the hero scrolls away.
        ScrollTrigger.create({
          start: 64,
          onUpdate: (self) => {
            document.querySelector(".nav")?.classList.toggle("is-stuck", self.scroll() > 64);
          },
        });
      }, rootRef);

      cleanup = () => context.revert();
    })();

    return () => {
      cancelled = true;
      cleanup();
    };
  }, []);

  const dashboardHref = signedIn ? "/dashboard" : needsFirstAccount ? "/login?first=1" : "/login";
  const windowsCommand = `irm ${origin}/join.ps1 | iex`;

  return (
    <div ref={rootRef}>
      <header className="nav">
        <div className="container nav-inner">
          <Link href="/" className="brand">
            <Logo />
            <span>{meshName}</span>
          </Link>
          <nav className="nav-links">
            <a href="#how">How it works</a>
            <a href="#scheduler">Scheduler</a>
            <a href="#trust">Trust</a>
            <Link href="/join">Contribute a GPU</Link>
          </nav>
          <div className="row" style={{ gap: 8 }}>
            <Link className="btn btn-sm" href="/join">
              Lend a GPU
            </Link>
            <Link className="btn btn-primary btn-sm" href={dashboardHref}>
              {signedIn ? "Open dashboard" : needsFirstAccount ? "Claim this mesh" : "Sign in"}
            </Link>
          </div>
        </div>
      </header>

      <section className="hero">
        <MeshCanvas />
        <div className="container hero-inner">
          <span className="hero-tag" data-hero>
            <span className="dot dot-live" />
            Running on this machine right now
          </span>

          <h1 data-hero>
            Every GPU on your network,
            <br />
            <span className="glow">one training cluster.</span>
          </h1>

          <p className="hero-sub" data-hero>
            The compute you need is already in the room. GradMesh finds the idle GPUs on your
            network, measures what each one can actually do, and trains a single model across all
            of them. One command to start. One line for anyone else to join.
          </p>

          <div className="hero-actions" data-hero>
            <Link className="btn btn-primary btn-lg" href={dashboardHref}>
              {signedIn ? "Open the dashboard" : "Open the dashboard"}
            </Link>
            <Link className="btn btn-lg" href="/join">
              Contribute this machine
            </Link>
          </div>

          <div className="hero-command" data-hero>
            <CopyLine value={windowsCommand} label="Anyone on this network can join with one line" />
          </div>
        </div>
      </section>

      <div className="container">
        <div className="rule" />
      </div>

      <section className="section" id="problem">
        <div className="container">
          <div className="section-head js-reveal">
            <span className="eyebrow">The problem</span>
            <h2>Distributed training punishes you for having mixed hardware.</h2>
            <p>
              Split a dataset evenly across a fast card and a slow one and every round costs what the
              slow one costs. Add a third, weaker machine and the mesh gets slower, not faster. That
              is why lending a friend&apos;s GPU has never actually been worth the setup.
            </p>
          </div>

          <div className="split-demo">
            <div className="split-card js-reveal">
              <div className="split-title">
                <div>
                  <div className="panel-title">Even split</div>
                  <div className="small faint">Every worker gets a third of the data</div>
                </div>
                <span className="badge badge-danger">42s round</span>
              </div>
              <Gantt
                rows={[
                  { name: "RTX 3060", work: 26, idle: 74, label: "11s" },
                  { name: "Arc A370M", work: 55, idle: 45, label: "23s" },
                  { name: "GTX 1650", work: 100, idle: 0, label: "42s" },
                ]}
              />
              <p className="small faint" style={{ marginTop: 16 }}>
                Two machines sit idle for most of the round waiting on the third.
              </p>
            </div>

            <div className="split-card is-good js-reveal">
              <div className="split-title">
                <div>
                  <div className="panel-title">GradMesh</div>
                  <div className="small faint">Shards sized to measured throughput</div>
                </div>
                <span className="badge badge-live">17s round</span>
              </div>
              <Gantt
                good
                rows={[
                  { name: "RTX 3060", work: 100, idle: 0, label: "17s" },
                  { name: "Arc A370M", work: 97, idle: 3, label: "17s" },
                  { name: "GTX 1650", work: 96, idle: 4, label: "16s" },
                ]}
              />
              <p className="small faint" style={{ marginTop: 16 }}>
                Everyone finishes together, so the round costs what the mesh costs.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="section" id="scheduler">
        <div className="container">
          <div className="section-head js-reveal">
            <span className="eyebrow">What makes it work</span>
            <h2>A scheduler that treats heterogeneity as the normal case.</h2>
            <p>
              Consumer hardware is never uniform. GradMesh plans for that instead of tolerating it,
              with four policies that run every single round.
            </p>
          </div>

          <div className="grid grid-3" data-stagger>
            {FEATURES.map((feature, index) => (
              <article className="feature" key={feature.title}>
                <span className="feature-index">{String(index + 1).padStart(2, "0")}</span>
                <h3>{feature.title}</h3>
                <p>{feature.body}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="section" id="how">
        <div className="container">
          <div className="section-head js-reveal">
            <span className="eyebrow">How it works</span>
            <h2>Four steps, none of which involve a requirements file.</h2>
            <p>
              The whole point of this version is that nobody has to understand the pipeline to
              contribute to it.
            </p>
          </div>

          <div className="flow js-reveal">
            {STEPS.map((step, index) => (
              <div className="flow-step" key={step.title}>
                <span className="num-badge">{index + 1}</span>
                <h3>{step.title}</h3>
                <p className="small muted">{step.body}</p>
              </div>
            ))}
          </div>

          <div className="grid grid-2 js-reveal" style={{ marginTop: 22 }}>
            <div className="panel">
              <div className="stack-sm">
                <span className="eyebrow">On the host</span>
                <pre className="code">{`npm install\nnpm run dev`}</pre>
                <p className="small faint">
                  Creates the Python environment, downloads the base checkpoints, starts the
                  coordinator and the dashboard, and prints your network address.
                </p>
              </div>
            </div>
            <div className="panel">
              <div className="stack-sm">
                <span className="eyebrow">On every other machine</span>
                <pre className="code">{`irm ${origin}/join.ps1 | iex`}</pre>
                <p className="small faint">
                  Detects the accelerator, installs the matching PyTorch build, and joins the mesh.
                  There is a shell equivalent for macOS and Linux.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="section" id="trust">
        <div className="container">
          <div className="section-head js-reveal">
            <span className="eyebrow">Trust</span>
            <h2>A mesh you can actually explain to the person lending you a GPU.</h2>
          </div>

          <div className="grid grid-3" data-stagger>
            <article className="feature">
              <h3>Nothing leaves the network</h3>
              <p>
                Dataset shards, checkpoints and gradient updates travel between machines on your own
                LAN. There is no upstream service in the path.
              </p>
            </article>
            <article className="feature">
              <h3>Joining takes a token</h3>
              <p>
                Workers present a mesh token before they receive any data. Rotate it from the
                dashboard and every uninvited machine drops out.
              </p>
            </article>
            <article className="feature">
              <h3>Every decision is legible</h3>
              <p>
                Admission, shard size and drops all carry a written reason. Contributors can see
                exactly what their machine did and why.
              </p>
            </article>
          </div>
        </div>
      </section>

      <section className="section">
        <div className="container">
          <div className="cta js-reveal">
            <span className="eyebrow">Ready</span>
            <h2 style={{ marginTop: 14 }}>
              The cluster you were going to rent is already on your Wi-Fi.
            </h2>
            <div
              className="row"
              style={{ justifyContent: "center", marginTop: 28, flexWrap: "wrap" }}
            >
              <Link className="btn btn-primary btn-lg" href={dashboardHref}>
                {needsFirstAccount ? "Claim this mesh" : "Open the dashboard"}
              </Link>
              <Link className="btn btn-lg" href="/join">
                Lend a GPU instead
              </Link>
            </div>
          </div>
        </div>
      </section>

      <footer className="footer">
        <div className="container row-between">
          <div className="row" style={{ gap: 9 }}>
            <Logo size={18} />
            <span>GradMesh 4</span>
          </div>
          <span className="small">
            Collaborative GPU training over an ordinary network. Research prototype.
          </span>
        </div>
      </footer>
    </div>
  );
}

function Gantt({
  rows,
  good = false,
}: {
  rows: { name: string; work: number; idle: number; label: string }[];
  good?: boolean;
}) {
  return (
    <div className="gantt">
      {rows.map((row) => (
        <div className="gantt-row" key={row.name}>
          <span className="small muted truncate">{row.name}</span>
          <div className="gantt-track">
            <div
              className={`gantt-bar${good ? " is-good" : ""}`}
              data-bar
              style={{ width: `${row.work}%` }}
            />
          </div>
          <span className="small mono faint" style={{ textAlign: "right" }}>
            {row.label}
          </span>
        </div>
      ))}
    </div>
  );
}
