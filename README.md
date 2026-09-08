# GradMesh 4

**Every GPU on your network, one training cluster.**

GradMesh turns the idle consumer GPUs already sitting on a local network into a
single coordinated training cluster. One machine hosts. Everyone else pastes one
line. The mesh measures what each device can actually do, sizes the work to
match, and trains a shared model across all of them.

```
npm install
npm run dev
```

That is the whole host setup. It creates the Python environment, downloads the
base checkpoints, starts the coordinator and the dashboard, and prints the
address peers should open.

---

## What changed from v3

v3 proved the idea: a FastAPI coordinator, worker agents, round-synchronised
YOLO training with FedAvg aggregation, over a LAN. It worked, and the training
pipeline it validated is **carried into v4 unchanged**.

What it could not do was get out of its own way. Running it meant two laptops,
four terminals, a requirements file chosen by hand per GPU vendor, a dataset path
hard-coded to one ZIP, and a scheduler that split the dataset evenly and then
waited for the slowest machine.

v4 fixes the product around that pipeline.

| | v3 | v4 |
|---|---|---|
| Host setup | venv, pick a requirements file, firewall rule, two terminals | `npm run dev` |
| Joining a GPU | copy the repo, install deps, find the host IP, run a CLI | paste one line |
| Finding the host | read an IP off the other screen | open `http://gradmesh.local:3000` |
| Seeing the network | nothing | Discover page sweeps the subnet in ~3s |
| Dataset | one hard-coded ZIP path | upload in the browser, registry of many |
| Shard sizing | equal split | proportional to measured throughput |
| Slow machine | the whole round waits | speculative re-execution, then dropped |
| Aggregation | unweighted mean | sample-weighted FedAvg, damped by reliability |
| Visibility | poll a static HTML page | live event stream, per-shard timing |
| Watching a run | nothing until it ended | live screen, progress bar, pinned indicator |
| Seeing who is nearby | nothing | radar view ranked by measured network distance |
| Access control | none | dashboard accounts plus a mesh join token |

---

## Quick start

### Host, the machine with the dataset

```bash
npm install
npm run dev
```

The launcher prints something like:

```
● Mesh is live

  Dashboard       http://localhost:3000
  Other devices   http://gradmesh.local:3000  no IP needed
  On this Wi-Fi   http://192.168.1.85:3000
  Join page       http://192.168.1.85:3000/join

  Anyone on this network pastes one line to lend their GPU:
  irm http://192.168.1.85:3000/join.ps1 | iex          Windows
  curl -fsSL http://192.168.1.85:3000/join.sh | sh     macOS, Linux
```

Open the dashboard, create the first account, which becomes the mesh owner, and
upload a dataset on the Datasets page.

The host can contribute its own GPU without a second terminal: **Use this
machine** in the top bar starts a worker as a child process.

### Contributors

On the machine with the GPU, open **`http://gradmesh.local:3000`**. No IP
address, no QR code, no scanning: the host claims that name over multicast DNS,
so it resolves on any device on the same network. Then paste the one line. It finds Python, creates a private environment under the home
folder, installs the PyTorch build matching whatever accelerator is present,
measures the device, and joins. Ctrl+C leaves the mesh and the current round
replans without that machine.

Nothing is installed system-wide. Everything lives in `~/.gradmesh/agent`.

### Firewall

Windows prompts on first run. If the prompt was dismissed, allow the two ports
once from an Administrator PowerShell:

```powershell
New-NetFirewallRule -DisplayName "GradMesh" -Direction Inbound -Protocol TCP -LocalPort 3000,8000 -Action Allow
```

---

## Commands

| Command | What it does |
|---|---|
| `npm run dev` | Everything: setup, coordinator, dashboard, LAN addresses |
| `npm run setup` | Just the environment. Safe to re-run |
| `npm run build` then `npm start` | Production mode |
| `npm run worker` | Contribute this machine to a mesh. `-- --server http://host:8000 --token <token>` for a remote one |
| `npm run coordinator` | Control plane only, with FastAPI docs at `/docs` |
| `npm run doctor` | Diagnose a broken setup, with a fix printed per failure |
| `npm run test:scheduler` | Assert the scheduler's behaviour. No PyTorch required |

---

## Finding each other

Two different problems, solved separately.

**Other devices finding the host.** The coordinator advertises itself over
multicast DNS and claims the name `gradmesh.local`. Any device on the same
network opens `http://gradmesh.local:3000` and lands on the dashboard, with
nothing configured on either side. This works on Windows 10 and later, macOS,
iOS, Android 12 and later, and most desktop Linux. If a network blocks multicast
the launcher says so and prints the address instead.

**The host seeing the network.** The **Discover devices** page, in the sidebar
and the top bar, sweeps the local `/24` in about three seconds and shows every
device on the Wi-Fi, split into what is contributing, what has the join page
open right now, and what is idle. It draws them as a radar with this host at the
centre, in the spirit of a file-sharing app, with a table view behind a toggle.

Two rules keep that radar honest. **Radius is measured, not decorative:** it
comes from a real TCP round trip, timed with dedicated blocking connects rather
than read off the batched sweep, whose numbers are dominated by scheduler delay
and varied between 8 ms and 114 ms for the same router. A device that answers on
no port has no measurable distance and sits on an outer ring labelled as such,
rather than being placed somewhere plausible. **Angle is a hash of the address,**
so a machine keeps the same seat on every sweep instead of jumping around.

Round trip tracks link quality, not metres. A wired machine in the next building
answers faster than a phone on weak Wi-Fi two metres away, and the rings are
labelled accordingly.

The sweep uses two cheap signals rather than a slow ping scan. The operating
system's ARP table already lists every device this host has exchanged frames
with, and a TCP connection that is *refused* proves a host exists just as well
as one that is accepted, so probing a few common ports finds devices that answer
nothing at all. Roughly fifteen hundred non-blocking connects run through one
selector rather than a thread each, which is the difference between three
seconds and eleven.

**Devices announcing themselves.** Anything that opens the join page tells the
host it is there, before installing anything. That row carries the device's
name, platform, core count and GPU as reported by WebGPU or WebGL, so an entry
in the scan reads "iPhone, Safari, 6 cores" rather than an unlabelled address.
The beacon posts straight to the coordinator, which accepts it without a token
from its own subnet, because a request relayed through the dashboard would
arrive from the host itself and every visitor would be recorded as `127.0.0.1`.

## Can a browser contribute without installing anything?

No, and it is worth being precise about why, because the answer changes what is
worth building next.

A browser tab can *see* the GPU through WebGPU, which is how the join page names
your graphics card without installing anything. What it cannot do is load a
PyTorch checkpoint, run Ultralytics, or write CUDA and XPU kernels. Those are
native libraries with direct driver access, and no browser exposes that. There
is no version of the current pipeline that runs in a tab.

What the browser genuinely does today:

- Resolves `gradmesh.local` so nobody types an address.
- Announces the device to the host, with its real hardware profile.
- Hands over a single line that sets everything up.

That line is not an install in the usual sense. It clones nothing, touches
nothing outside `~/.gradmesh`, and disappears when the terminal closes. It is
about as close to zero-install as native GPU training gets.

The honest zero-install path, if it is ever worth taking, is a different
training stack rather than a different transport: an ONNX Runtime Web or
`transformers.js` model trained through WebGPU compute shaders, aggregated as
raw tensors instead of PyTorch state dicts. That is a second engine, not a
feature, and it would train a much smaller model than YOLO. Worth prototyping
only if browser-only contribution turns out to be the thing that decides whether
people join.

## Watching a run

**The pinned indicator.** Bottom right of every dashboard page, whenever
anything is training: a progress ring, the round counter, elapsed time, and a
bar. Clicking it goes to the live screen. It hides on the training page itself
and comes back on its own for the next run if dismissed.

**The training screen** at `/dashboard/training` is the one the product revolves
around. It carries overall progress as a single percentage and a bar ticked once
per round, per-machine shard bars against each shard's own prediction, the last
round's timings, and the live event feed.

Progress is reported at two granularities on purpose. Round progress alone sits
at "1 of 4" for minutes and looks stuck, so the shards finished inside the
current round fill in the gap between round boundaries.

### The 3D element

`components/dashboard/TrainingRig.tsx` renders it. Drop a GLB at
`public/models/training-rig.glb` and it is picked up on the next page load, with
no code change and no rebuild. Until that file exists it renders a procedural GPU
fan, so the screen works either way.

What the model should contain, and how it behaves, is written up in
`public/models/README.md`. In short: baked animation clips that loop cleanly,
authored around the origin, embedded textures. Every clip in the file is played
at once, so a robot arm with one clip per joint needs no configuration. Playback
speed is driven by how much of the mesh is busy, so the rig visibly runs harder
under load, and its accent light changes colour with the run state.

## Architecture

```
                        Browser
                           |
                   Next.js dashboard          <- accounts, live view, uploads
                           |  server-side, holds the mesh token
                           v
                  FastAPI coordinator          <- the control plane
                           |
      +--------------------+--------------------+
      |                    |                    |
   Worker 1             Worker 2             Worker N
   RTX 3060             Arc A370M            GTX 1650
      |                    |                    |
   87 images            33 images            18 images     <- sized to throughput
      |                    |                    |
      +--------------------+--------------------+
                           |
                 weighted FedAvg aggregation
                           |
                    next round replans
```

Two processes, started together:

- **`engine/`** is the Python control plane and the training agent. The
  coordinator owns the node registry, admission, round planning, shard
  materialisation, the round barrier, aggregation and run history.
- **The Next.js app** is the dashboard and the join surface. The browser never
  holds the mesh token; it calls Next, and Next attaches the token server-side.

State lives in `.gradmesh/` at the repository root: datasets, run artifacts,
checkpoints, accounts and the join token. Deleting that directory resets the
mesh.

### The training pipeline is untouched

`engine/worker.py`'s `train_batch` is the v3 function. It downloads a shard,
loads the global state dict through the same shape-compatible filter, and calls
the same Ultralytics training options, including the Intel XPU path through
`ultralytics_xpu.xpu_train` with AMP disabled and `foreach=False`.
`accelerator.py`, `federated_training.py` and `ultralytics_xpu.py` are carried
over verbatim.

Everything v4 adds sits around that call.

---

## The scheduler

This is the part that makes a mixed-hardware mesh worth building, and it is
documented in full in [RESEARCH.md](RESEARCH.md).

A round costs `max_i(t_i)`. Splitting a dataset evenly across a fast card and a
slow one means every round costs what the slow one costs, and adding a third,
weaker machine makes the mesh slower rather than faster. Four policies run every
round to fix that.

**1. Admission control.** Every worker runs a short FP32 throughput probe at
startup and reports it. The coordinator ranks devices on measured GFLOP/s and
memory bandwidth rather than a name string. A device that cannot fit the model
is rejected with a written reason; one that is far below the mesh median is
admitted on probation with a capped shard.

**2. Predictive proportional sharding.** With throughput `r_i` and shard size
`s_i`, worker time is `s_i / r_i`. Makespan is minimised when every `s_i / r_i`
is equal, which gives `s_i` proportional to `r_i`. Shards are sized that way,
then clipped by the probation cap and a skew cap, with trimmed samples
redistributed by water-filling and rounding repaired by largest remainder.

**3. Straggler mitigation.** Each shard carries a soft and hard deadline derived
from its own prediction. Past the soft deadline, an idle peer starts a duplicate
and whichever lands first wins. Past the hard deadline, the barrier releases
without it and the machine's reliability drops. Losing more than a third of the
round's samples aborts and replans instead of aggregating a hole.

**4. Contribution-weighted aggregation.** Unequal shards require weighted
FedAvg: each update is weighted by the samples behind it, scaled by that
worker's reliability, normalised to sum to one.

After every round the coordinator updates an exponentially weighted moving
average of each machine's real samples-per-second and replans. The mesh gets
better at scheduling itself the longer it runs.

Every value driving this is exposed on the Scheduler page, with a live preview
of how the next round would be split.

---

## Trust model

GradMesh is designed for a network you can see, and it says so rather than
implying more.

- **Workers authenticate with a mesh token.** No token, no dataset shards. The
  token is minted by the coordinator and stored in `.gradmesh/coordinator.json`.
- **The join page shows the token to anyone who can reach the host.** That is
  deliberate: on a LAN, being on the network is the trust boundary, and hiding
  the token behind a login would only stop the people you are trying to invite.
  Rotate it from the Invite page to revoke access, which disconnects every
  machine immediately.
- **Dashboard accounts are separate from that.** The first account is the owner
  and can start runs, upload datasets, change policy and evict machines. Later
  accounts are members with a read-only view. Passwords are scrypt-hashed;
  sessions are HMAC-signed cookies.
- **Nothing leaves the network.** Shards, checkpoints and weight updates move
  between machines on your LAN. There is no upstream service in the path.
- **The presence beacon is open on the local subnet.** A device announcing that
  it is looking at the join page does so without a token, because it has not
  installed anything yet. It is rate limited, it carries no dataset access, and
  it writes only to a list of who is on the network, which anyone able to reach
  this host could already observe.

### Guardrails for mixed hardware

Two failure modes are specific to a mesh of machines nobody controls, and both
are handled rather than left to chance.

**A machine with no usable GPU** still joins, is measured, and is marked
ineligible with the reason shown in the dashboard. It receives no shards and
slows nobody down. The join page says this will happen before anyone installs
anything, because a browser can only guess at graphics hardware and the agent is
what actually decides.

**Batch size is resolved per device, not per run.** The person starting the run
picks one number, but a batch of 8 at 640px is comfortable on a 12 GB card and an
instant out-of-memory crash on a 4 GB laptop. The requested value is treated as a
ceiling and each machine is given whatever it can hold, from its device memory
and the image size. Without that, one contributor with a smaller GPU fails every
round, loses reliability for it, and is eventually dropped from a mesh they could
have helped with.

What is **not** solved: a malicious worker can return poisoned weights. There is
no result verification, and reliability scoring only catches failure, not lying.
Do not run this across an untrusted network.

---

## Dataset format

A YOLO export, zipped:

```
dataset/
├── images/train/    ├── labels/train/
├── images/val/      └── labels/val/     (optional)
└── data.yaml                            (optional, supplies class names)
```

`images/train` and `labels/train` are required; label files match image files by
name. Class names come from any `data.yaml` in the archive, or default to a
single `object` class. Both nesting conventions are accepted, so
`train/images/` works as well as `images/train/`.

The first dataset uploaded becomes the mesh default, which is what every machine
that joins later trains against. Uploads are validated before registration, and
the archive is path-checked so it cannot write outside its own directory.

---

## Benchmarking

The comparison the research question needs is built into the run form.

1. Start a run with mode **Single GPU baseline**. It plans onto the strongest
   machine only.
2. Start the same run with mode **Mesh**. Same dataset, checkpoint, image size,
   batch size and round count.
3. Compare on the Runs page: wall clock, speedup, efficiency, straggler gap per
   round, aggregation cost per round, and per-machine prediction error.

Every finished run writes `summary.json` next to its weights under
`.gradmesh/runs/<id>/`, and the aggregated checkpoint downloads as a `.pt`.

---

## Troubleshooting

Run `npm run doctor` first. It checks Node, npm dependencies, Python, the
virtual environment, both dependency planes, checkpoints, the mesh token,
datasets, the accelerator, the network and both ports, and prints a fix for each
failure.

**`gradmesh.local` does not resolve.** Some corporate and guest networks block
multicast DNS, and Windows 8 and older have no mDNS resolver. Use the address
the launcher prints instead. The Discover page shows whether the advertisement
is active and why it is not.

**A contributor's machine will not join.** They must be on the same network as
the host. Check with `Test-NetConnection <host-ip> -Port 8000` on Windows or
`nc -vz <host-ip> 8000` elsewhere. If it fails, the firewall rule above is
missing. The agent also accepts `--discover`, which scans the local subnet when
the host's address has changed.

**Runs will not start.** Aggregation needs PyTorch on the host. The dashboard
banner shows install progress; wait for it, or run `npm run setup`.

**A worker runs out of GPU memory.** Batch size is already capped per device, so
this should be rare. If it still happens, lower the batch size on the run form:
it is a ceiling for the whole run, and each machine gets at most that.

**Use this machine does nothing, or the worker keeps dying.** The agent started
from the dashboard runs detached, with its own process group and console, so the
dev server cannot take it down. If it still will not start, check
`.gradmesh/worker.log`, which holds its full output, and `npm run doctor`.

**Intel Arc workers.** The agent installs the `+xpu` PyTorch build and takes the
XPU trainer path automatically. Ultralytics 8.4.46 rejects the string `xpu`, so a
validated `torch.device("xpu:0")` is passed through the custom trainer, with AMP
and final validation disabled and `foreach=False` for Adam-family optimizers and
gradient clipping. This is the v3 behaviour, carried over unchanged.

---

## Scaling beyond one network

Everything above assumes a single LAN, which is the right scope for the current
claim. Going further is four separate problems, in the order they bite.

**1. Transport.** Today shards and weights move over plain HTTP between machines
that can see each other. Off the LAN, that needs TLS, and the two peers usually
cannot see each other at all. The realistic path is WebRTC data channels or a
WireGuard mesh, with the coordinator acting only as a rendezvous point rather
than relaying the traffic. NAT traversal is the actual work.

**2. Bandwidth.** A round currently transfers the entire model state dict per
worker, twice. YOLOv8n is about 12 MB, which is nothing on a LAN and painful on
a home upload link. Beyond one network the aggregation has to send deltas rather
than weights, quantised and sparsified. This is well-studied and it is the
single change that most widens the range of workable models.

**3. Verification.** On your own Wi-Fi you know whose machines these are. Open
to strangers, a worker can return plausible but poisoned weights and nothing
would catch it. The standard answer is redundant assignment with cross-checking
plus outlier rejection at aggregation, which costs real throughput. Until that
exists, this should not run across a network you do not control.

**4. Scheduling at a different scale.** The current scheduler assumes uniform,
sub-millisecond latency, which is true on a LAN and false everywhere else. The
fitness function already carries a latency term, but shard sizing does not yet
account for transfer time, and across the internet transfer would often dominate
compute. That turns the sizing problem from "equalise compute time" into
"equalise compute plus transfer", which is a genuinely different optimisation.

The nearest useful step is not any of those. It is more machines on one network:
three or four heterogeneous GPUs in one room is where the scheduler's claims
become measurable rather than merely predicted, and that is the experiment the
paper needs.

## Limitations

1. Round-synchronised weight averaging is not step-level gradient
   synchronisation. For a single machine with multiple GPUs, PyTorch DDP over
   NCCL remains the right tool.
2. Communication cost scales with model size. This is tuned for small detection
   models on a LAN, not for large models over the internet.
3. There is no worker result verification.
4. In-flight round state is in memory. A coordinator restart loses the current
   round, deliberately: a half-finished barrier is not resumable and pretending
   otherwise would produce silently wrong aggregation.
5. Speedup is bounded by communication, serialisation and synchronisation. Two
   GPUs do not give 2x, and the dashboard reports what actually happened rather
   than what was hoped for.

---

## Research

The central question, unchanged from v3:

> Can heterogeneous, independently owned consumer GPUs be coordinated over an
> ordinary network to reduce deep-learning training time while maintaining
> comparable model quality, without datacenter-grade infrastructure?

v4's contribution is the scheduling policy that makes the answer depend on
something other than owning matched hardware. The model, the assertions that
pin it down, and the measurement methodology are in
[RESEARCH.md](RESEARCH.md).
<<<<<<< HEAD
#
