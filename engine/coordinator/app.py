"""GradMesh v4 coordinator.

One FastAPI process owns the whole control plane: node registry, admission,
round planning, shard materialisation, the round barrier, aggregation, run
history and the live event stream.

What changed from v3, and why:

* Shards are sized per node from measured throughput instead of split evenly, so
  a slow laptop no longer sets the pace for the whole mesh.
* A background supervisor enforces per-shard deadlines, so a machine that goes
  to sleep mid-round costs the run seconds rather than the lease timeout.
* Aggregation is sample-weighted, which is what unequal shards require.
* Datasets are uploaded through the dashboard and stored in a registry, so the
  strawberry ZIP is one dataset among many rather than a hard-coded path.
* Every transition is published to an event bus that the dashboard streams.

The training pipeline itself is unchanged. Workers still receive a shard zip, a
base checkpoint and a global state dict, and still run the same Ultralytics call
that v3 validated, including the Intel XPU path.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import sys
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional

# The engine directory holds the v3 modules that must stay importable by name.
ENGINE_DIR = Path(__file__).resolve().parent.parent
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from coordinator import aggregation, discovery, sharding, store
from coordinator.events import bus

from coordinator.scheduler import (
    DEFAULT_POLICY,
    Deadline,
    MeshPolicy,
    admit,
    aggregation_weights,
    deadline_for,
    efficiency,
    fitness,
    mesh_reference,
    plan_round,
    should_abort_round,
    straggler_action,
    update_reliability,
    update_throughput,
)

# Rounds close on the aggregator thread and deadlines fire on an executor thread,
# so every publish goes through the thread-safe path.
emit = bus.publish_threadsafe

HEARTBEAT_TIMEOUT_SECONDS = float(os.getenv("GRADMESH_HEARTBEAT_TIMEOUT", "20"))
SUPERVISOR_INTERVAL_SECONDS = 2.0
MAX_ROUND_ATTEMPTS = 2

# ---------------------------------------------------------------------------
# In-memory mesh state
# ---------------------------------------------------------------------------

state_lock = RLock()
nodes: Dict[str, dict] = {}
runs: Dict[str, dict] = {}
batches: Dict[str, dict] = {}
aggregator = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gradmesh-agg")
scanner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gradmesh-scan")

# Devices that opened the join page but have not installed the agent. This is
# how the host sees "my other laptop is here, looking at the join screen right
# now" without anything being installed on that laptop.
visitors: Dict[str, dict] = {}
VISITOR_TTL_SECONDS = 75.0
_visitor_rate: Dict[str, List[float]] = {}
VISITOR_RATE_WINDOW = 10.0
VISITOR_RATE_LIMIT = 8

advertiser = discovery.MulticastAdvertiser(
    hostname=os.getenv("GRADMESH_MDNS_NAME", "gradmesh"),
    web_port=int(os.getenv("GRADMESH_WEB_PORT", "3000")),
    api_port=int(os.getenv("GRADMESH_COORDINATOR_PORT", "8000")),
)

_scan_cache: Dict[str, Any] = {"result": None, "at": 0.0, "running": False}
SCAN_CACHE_SECONDS = 25.0


def current_policy() -> MeshPolicy:
    return MeshPolicy.from_dict(store.policy())


def _node_view(node: dict, mesh: dict, policy: MeshPolicy) -> dict:
    view = {key: value for key, value in node.items() if key != "secret"}
    view["fitness"] = round(fitness(node, mesh, policy), 4)
    return view


def snapshot_nodes() -> List[dict]:
    policy = current_policy()
    now = time.time()
    with state_lock:
        _refresh_liveness(now)
        pool = list(nodes.values())
        mesh = mesh_reference(pool, now)
        return [_node_view(node, mesh, policy) for node in pool]


def _refresh_liveness(now: float) -> None:
    for node in nodes.values():
        was_active = node.get("active", False)
        node["active"] = (now - float(node.get("last_seen") or 0)) <= HEARTBEAT_TIMEOUT_SECONDS
        if was_active and not node["active"]:
            emit(
                "node.offline", {"node_id": node["node_id"], "name": node.get("display_name")}
            )


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def require_mesh_token(x_mesh_token: Optional[str] = Header(default=None)) -> str:
    """Workers and the dashboard proxy must present the mesh join token.

    This is a shared secret for a LAN, not an identity system. It stops a random
    device on a coffee-shop network from registering as a worker and receiving
    dataset shards, which is the actual threat when you hand out a URL.
    """
    expected = store.mesh_token()
    if not x_mesh_token or not _constant_time_equals(x_mesh_token, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing mesh token")
    return x_mesh_token


def _constant_time_equals(left: str, right: str) -> bool:
    import hmac

    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def _is_local_client(host: Optional[str]) -> bool:
    """True when the caller is on this machine or this /24."""
    if not host:
        return False
    if host in {"127.0.0.1", "::1", "localhost"}:
        return True
    own = discovery.local_ipv4()
    if own.startswith("127."):
        return False
    return host.rsplit(".", 1)[0] == own.rsplit(".", 1)[0]


def allow_local_or_token(
    request: Request, x_mesh_token: Optional[str] = Header(default=None)
) -> str:
    """Auth for the presence beacon only.

    The beacon has to be callable by a browser that has not installed anything
    and does not hold the token, and it must see the visitor's real address,
    which a server-side proxy would replace with its own. So a request from this
    subnet is accepted without a token. It carries no dataset access and writes
    only to a list of who is looking at the join page, which anyone able to reach
    this host could observe anyway.
    """
    expected = store.mesh_token()
    if x_mesh_token and _constant_time_equals(x_mesh_token, expected):
        return x_mesh_token
    client = request.client.host if request.client else None
    if _is_local_client(client):
        return "local"
    raise HTTPException(status_code=401, detail="Invalid or missing mesh token")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    node_id: str
    display_name: Optional[str] = None
    gpu: str = "unknown"
    gpu_memory_mb: int = Field(default=8000, ge=1)
    max_batch_size: int = Field(default=4, ge=1, le=128)
    backend: str = "cpu"
    supports_training: bool = True
    capability: Optional[dict] = None
    owner: Optional[str] = None
    agent_version: str = "4.0.0"


class HeartbeatRequest(BaseModel):
    node_id: str
    load: Optional[float] = None
    active_batches: Optional[int] = None
    allocated_memory_mb: Optional[int] = None
    training_epoch: Optional[int] = None
    training_total_epochs: Optional[int] = None
    latency_ms: Optional[float] = None


class RoundResultRequest(BaseModel):
    node_id: str
    batch_id: str
    round_index: int = Field(ge=0)
    weights_b64: str = Field(..., min_length=1)
    metrics: Optional[dict] = None


class BatchFailureRequest(BaseModel):
    node_id: str
    batch_id: str
    error: str = Field(..., min_length=1)


class CreateRunRequest(BaseModel):
    name: str = Field(default="mesh-run", min_length=1, max_length=80)
    dataset_id: Optional[str] = None
    base_model: str = Field(default="yolov8n.pt", min_length=1)
    rounds: int = Field(default=4, ge=1, le=200)
    imgsz: int = Field(default=640, ge=32, le=2048)
    batch_size: int = Field(default=8, ge=1, le=128)
    mode: str = Field(default="mesh")  # mesh or solo
    notes: Optional[str] = None


class PolicyRequest(BaseModel):
    values: Dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    bus.bind_loop(asyncio.get_running_loop())
    store.load()
    supervisor = asyncio.create_task(_supervisor_loop())

    # Claiming gradmesh.local is what lets a peer open the dashboard without
    # anybody reading an IP address off a screen. Best effort: a network that
    # blocks multicast falls back to the address, it does not fail startup.
    #
    # This runs in a worker thread because zeroconf's synchronous API blocks on
    # its own event loop, which it refuses to do from inside another one.
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, advertiser.start)

    emit("coordinator.ready", {"version": "4.0.0", "mdns": advertiser.as_dict()})
    try:
        yield
    finally:
        supervisor.cancel()
        try:
            await supervisor
        except asyncio.CancelledError:
            pass
        await loop.run_in_executor(None, advertiser.stop)
        aggregator.shutdown(wait=False)
        scanner.shutdown(wait=False)


app = FastAPI(title="GradMesh Coordinator", version="4.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Node lifecycle
# ---------------------------------------------------------------------------


@app.post("/register_node")
def register_node(req: RegisterRequest, _: str = Depends(require_mesh_token)):
    now = time.time()
    with state_lock:
        existing = nodes.get(req.node_id) or {}
        nodes[req.node_id] = {
            "node_id": req.node_id,
            "display_name": req.display_name or req.gpu,
            "gpu": req.gpu,
            "backend": req.backend,
            "gpu_memory_mb": req.gpu_memory_mb,
            "max_batch_size": req.max_batch_size,
            "supports_training": req.supports_training,
            "capability": req.capability or existing.get("capability") or {},
            "owner": req.owner,
            "agent_version": req.agent_version,
            "joined_at": existing.get("joined_at", now),
            "last_seen": now,
            "active": True,
            "load": 0.0,
            "latency_ms": existing.get("latency_ms", 0.0),
            "allocated_memory_mb": 0,
            "active_batches": 0,
            "completed_rounds": existing.get("completed_rounds", 0),
            "failed_rounds": existing.get("failed_rounds", 0),
            "samples_trained": existing.get("samples_trained", 0),
            "seconds_trained": existing.get("seconds_trained", 0.0),
            # A fresh node starts trusted enough to receive work but not enough
            # to outweigh a node that has actually finished rounds.
            "reliability": existing.get("reliability", 0.7),
            "throughput_sps": existing.get("throughput_sps", 0.0),
            "training_epoch": 0,
            "training_total_epochs": 0,
        }
    policy = current_policy()
    # Admission is relative to the rest of the mesh, so judge the new node
    # against the whole pool rather than against itself.
    with state_lock:
        pool = [dict(node) for node in nodes.values()]
    decision = admit(pool, policy)[req.node_id]
    emit(
        "node.joined",
        {
            "node_id": req.node_id,
            "name": req.display_name,
            "gpu": req.gpu,
            "backend": req.backend,
            "tier": decision.tier,
            "reason": decision.reason,
            "gflops": (req.capability or {}).get("gflops"),
        },
    )
    return {
        "status": "registered",
        "node_id": req.node_id,
        "admission": decision.as_dict(),
        "heartbeat_seconds": max(3.0, HEARTBEAT_TIMEOUT_SECONDS / 4),
    }


@app.post("/heartbeat")
def heartbeat(req: HeartbeatRequest, _: str = Depends(require_mesh_token)):
    with state_lock:
        node = nodes.get(req.node_id)
        if node is None:
            raise HTTPException(status_code=404, detail="Unknown node. Register first.")
        node["last_seen"] = time.time()
        node["active"] = True
        if req.load is not None:
            node["load"] = req.load
        if req.active_batches is not None:
            node["active_batches"] = req.active_batches
        if req.allocated_memory_mb is not None:
            node["allocated_memory_mb"] = req.allocated_memory_mb
        if req.training_epoch is not None:
            node["training_epoch"] = req.training_epoch
        if req.training_total_epochs is not None:
            node["training_total_epochs"] = req.training_total_epochs
        if req.latency_ms is not None:
            node["latency_ms"] = req.latency_ms
    return {"status": "ok"}


@app.post("/leave")
def leave(req: HeartbeatRequest, _: str = Depends(require_mesh_token)):
    with state_lock:
        node = nodes.pop(req.node_id, None)
    if node:
        emit("node.left", {"node_id": req.node_id, "name": node.get("display_name")})
    return {"status": "ok"}


@app.delete("/nodes/{node_id}")
def evict_node(node_id: str, _: str = Depends(require_mesh_token)):
    with state_lock:
        node = nodes.pop(node_id, None)
        for batch in batches.values():
            if batch.get("node_id") == node_id and batch["status"] in {"queued", "assigned"}:
                batch["status"] = "dropped"
                batch["error"] = "node was evicted by the mesh owner"
    if node is None:
        raise HTTPException(status_code=404, detail="Unknown node")
    emit("node.evicted", {"node_id": node_id, "name": node.get("display_name")})
    return {"status": "evicted"}


# ---------------------------------------------------------------------------
# Work dispatch
# ---------------------------------------------------------------------------


@app.get("/get_batch/{node_id}")
def get_batch(node_id: str, _: str = Depends(require_mesh_token)):
    with state_lock:
        node = nodes.get(node_id)
        if node is None:
            raise HTTPException(status_code=404, detail="Unknown node. Register first.")
        _refresh_liveness(time.time())

        for batch in batches.values():
            if batch["status"] != "queued" or batch.get("node_id") != node_id:
                continue
            run = runs.get(batch["run_id"])
            if run is None or run["status"] not in {"running", "planning"}:
                continue

            batch["status"] = "assigned"
            batch["assigned_at"] = time.time()
            node["active_batches"] = 1
            node["allocated_memory_mb"] = batch["memory_mb"]

            emit(
                "shard.assigned",
                {
                    "run_id": run["id"],
                    "batch_id": batch["batch_id"],
                    "node_id": node_id,
                    "name": node.get("display_name"),
                    "round": batch["round_index"],
                    "samples": batch["samples"],
                    "predicted_seconds": batch["predicted_seconds"],
                },
            )
            return {"batch": _batch_payload(batch, run)}

    return {"batch": None}


def _batch_payload(batch: dict, run: dict) -> dict:
    """The wire shape the v3 worker already understands, plus v4 fields."""
    return {
        "batch_id": batch["batch_id"],
        "job_id": run["id"],
        "kind": "training",
        "round_index": batch["round_index"],
        "shard_index": batch["shard_index"],
        "shard_url": "/runs/%s/shards/%d.zip" % (run["id"], batch["shard_index"]),
        "weights_url": "/runs/%s/weights" % run["id"],
        "base_model": run["base_model"],
        "imgsz": run["imgsz"],
        "batch_size": min(run["batch_size"], batch.get("max_batch_size", run["batch_size"])),
        "estimated_memory_mb": batch["memory_mb"],
        "epochs": 1,
        "job_name": run["name"],
        "class_names": run["class_names"],
        "current_round": run["current_round"],
        "total_rounds": run["rounds"],
        "samples": batch["samples"],
        "deadline_seconds": batch["hard_deadline_seconds"],
    }


@app.post("/submit_training_round_result")
def submit_round_result(req: RoundResultRequest, _: str = Depends(require_mesh_token)):
    finished_at = time.time()
    with state_lock:
        batch = batches.get(req.batch_id)
        if batch is None:
            raise HTTPException(status_code=404, detail="Unknown batch")
        if batch.get("node_id") != req.node_id:
            raise HTTPException(status_code=409, detail="Batch is not assigned to this node")
        if batch["round_index"] != req.round_index:
            raise HTTPException(status_code=400, detail="round_index does not match batch")

        if batch["status"] in {"done", "superseded"}:
            # This shard already has a winner. Accept and discard.
            return {"status": "superseded", "batch_id": req.batch_id}

        elapsed = max(0.001, finished_at - float(batch.get("assigned_at") or finished_at))
        batch["status"] = "done"
        batch["finished_at"] = finished_at
        batch["elapsed_seconds"] = elapsed
        batch["weights_b64"] = req.weights_b64
        batch["metrics"] = req.metrics or {}

        node = nodes.get(req.node_id)
        if node is not None:
            policy = current_policy()
            node["throughput_sps"] = update_throughput(node, batch["samples"], elapsed, policy)
            node["reliability"] = update_reliability(node, True, policy)
            node["completed_rounds"] = node.get("completed_rounds", 0) + 1
            node["samples_trained"] = node.get("samples_trained", 0) + batch["samples"]
            node["seconds_trained"] = node.get("seconds_trained", 0.0) + elapsed
            node["active_batches"] = 0
            node["allocated_memory_mb"] = 0

        # A speculative pair races for the same shard. The loser must leave the
        # barrier immediately, otherwise the round waits on work it no longer
        # needs.
        _supersede_twins_locked(batch)

        run = runs.get(batch["run_id"])
        run_id = run["id"] if run else None

    emit(
        "shard.completed",
        {
            "run_id": run_id,
            "batch_id": req.batch_id,
            "node_id": req.node_id,
            "round": req.round_index,
            "seconds": round(elapsed, 2),
            "samples": batch["samples"],
            "throughput_sps": round(batch["samples"] / elapsed, 3),
        },
    )

    if run_id:
        _maybe_close_round(run_id)
    return {"status": "received", "batch_id": req.batch_id}


@app.post("/submit_training_batch_failure")
def submit_batch_failure(req: BatchFailureRequest, _: str = Depends(require_mesh_token)):
    with state_lock:
        batch = batches.get(req.batch_id)
        if batch is None:
            raise HTTPException(status_code=404, detail="Unknown batch")
        if batch["status"] in {"done", "dropped"}:
            return {"status": "ignored"}
        batch["status"] = "failed"
        batch["error"] = req.error[:4000]
        batch["finished_at"] = time.time()

        node = nodes.get(req.node_id)
        if node is not None:
            node["reliability"] = update_reliability(node, False, current_policy())
            node["failed_rounds"] = node.get("failed_rounds", 0) + 1
            node["active_batches"] = 0
            node["allocated_memory_mb"] = 0
        run_id = batch["run_id"]

    emit(
        "shard.failed",
        {"run_id": run_id, "batch_id": req.batch_id, "node_id": req.node_id, "error": req.error[:400]},
    )
    _maybe_close_round(run_id)
    return {"status": "recorded"}


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


@app.post("/runs")
def create_run(req: CreateRunRequest, _: str = Depends(require_mesh_token)):
    dataset = store.get_dataset(req.dataset_id) if req.dataset_id else store.default_dataset()
    if dataset is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset is available. Upload one from the Datasets page first.",
        )

    model_path = store.MODELS_DIR / Path(req.base_model).name
    if not model_path.is_file():
        raise HTTPException(
            status_code=400,
            detail="Base checkpoint %s is not available yet. It downloads during setup."
            % Path(req.base_model).name,
        )

    if not aggregation.torch_ready():
        raise HTTPException(
            status_code=503,
            detail="The training plane is still installing. Aggregation needs torch on the host.",
        )

    run_id = uuid.uuid4().hex[:12]
    now = time.time()
    run = {
        "id": run_id,
        "name": req.name,
        "status": "planning",
        "mode": req.mode,
        "created_at": now,
        "started_at": now,
        "finished_at": None,
        "dataset_id": dataset["id"],
        "dataset_name": dataset["name"],
        "class_names": dataset.get("class_names") or ["object"],
        "total_samples": int(dataset.get("train_count") or 0),
        "base_model": Path(req.base_model).name,
        "rounds": req.rounds,
        "current_round": 0,
        "imgsz": req.imgsz,
        "batch_size": req.batch_size,
        "notes": req.notes,
        "weights_b64": None,
        "round_history": [],
        "plan": None,
        "error": None,
        "shards": [],
    }
    with state_lock:
        runs[run_id] = run

    emit("run.created", {"run_id": run_id, "name": req.name, "rounds": req.rounds})

    try:
        _start_round(run_id)
    except HTTPException:
        with state_lock:
            runs.pop(run_id, None)
        raise
    except Exception as exc:
        with state_lock:
            run["status"] = "failed"
            run["error"] = "%s: %s" % (type(exc).__name__, exc)
        emit("run.failed", {"run_id": run_id, "error": run["error"]})
        raise HTTPException(status_code=500, detail=run["error"])

    return _run_view(run_id)


@app.get("/runs")
def list_runs(_: str = Depends(require_mesh_token)):
    with state_lock:
        live = [_run_summary(run) for run in runs.values()]
    archived = [run for run in store.list_runs() if run["id"] not in {r["id"] for r in live}]
    combined = live + archived
    combined.sort(key=lambda item: item.get("created_at") or 0, reverse=True)
    return {"runs": combined}


@app.get("/runs/{run_id}")
def get_run(run_id: str, _: str = Depends(require_mesh_token)):
    with state_lock:
        if run_id in runs:
            return _run_view(run_id)
    archived = store.get_run(run_id)
    if archived is None:
        raise HTTPException(status_code=404, detail="Unknown run")
    return archived


@app.post("/runs/{run_id}/stop")
def stop_run(run_id: str, _: str = Depends(require_mesh_token)):
    with state_lock:
        run = runs.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Unknown run")
        run["status"] = "stopped"
        run["finished_at"] = time.time()
        for batch in batches.values():
            if batch["run_id"] == run_id and batch["status"] in {"queued", "assigned"}:
                batch["status"] = "dropped"
                batch["error"] = "run stopped by the mesh owner"
    emit("run.stopped", {"run_id": run_id})
    _archive_run(run_id)
    return {"status": "stopped"}


@app.get("/runs/{run_id}/shards/{shard_index}.zip")
def get_shard(run_id: str, shard_index: int, _: str = Depends(require_mesh_token)):
    with state_lock:
        run = runs.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Unknown run")
        shards = run.get("shards") or []
        if shard_index < 0 or shard_index >= len(shards):
            raise HTTPException(status_code=404, detail="Unknown shard")
        zip_path = Path(shards[shard_index]["zip_path"])
    if not zip_path.is_file():
        raise HTTPException(status_code=404, detail="Shard archive is missing")
    return FileResponse(zip_path, media_type="application/zip", filename=zip_path.name)


@app.get("/runs/{run_id}/weights")
def get_run_weights(run_id: str, _: str = Depends(require_mesh_token)):
    with state_lock:
        run = runs.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Unknown run")
        return {
            "job_id": run_id,
            "base_model": run["base_model"],
            "weights_b64": run.get("weights_b64"),
            "round": run["current_round"],
        }


@app.get("/runs/{run_id}/artifact")
def download_artifact(run_id: str, _: str = Depends(require_mesh_token)):
    """Download the aggregated global weights as a .pt file."""
    path = store.run_dir(run_id) / "global.pt"
    if not path.is_file():
        record = store.get_run(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="No artifact for this run yet")
        raise HTTPException(status_code=404, detail="This run finished without producing weights")
    return FileResponse(path, media_type="application/octet-stream", filename="%s.pt" % run_id)


# ---------------------------------------------------------------------------
# Round machinery
# ---------------------------------------------------------------------------


def _start_round(run_id: str) -> None:
    """Plan, materialise and queue one round of shards."""
    policy = current_policy()

    with state_lock:
        run = runs.get(run_id)
        if run is None or run["status"] in {"stopped", "failed", "done"}:
            return
        _refresh_liveness(time.time())
        pool = [dict(node) for node in nodes.values()]
        round_index = run["current_round"]
        total_samples = run["total_samples"]
        mode = run["mode"]

    candidates = [node for node in pool if node.get("active") and node.get("supports_training", True)]

    if mode == "solo" and candidates:
        mesh = mesh_reference(candidates)
        candidates = [max(candidates, key=lambda node: fitness(node, mesh, policy))]

    plan = plan_round(candidates, total_samples, policy)

    if not plan.assignments:
        with state_lock:
            run["status"] = "waiting"
            run["plan"] = plan.as_dict()
        emit(
            "run.waiting",
            {
                "run_id": run_id,
                "reason": "no eligible worker is online",
                "rejected": plan.rejected,
            },
        )
        return

    dataset = store.get_dataset(run["dataset_id"])
    if dataset is None:
        raise RuntimeError("The dataset for this run was deleted")

    sizes = [assignment.samples for assignment in plan.assignments]
    shard_dir = store.run_dir(run_id) / ("round_%d" % round_index)

    # Shards are rebuilt every round rather than reused, even when the plan did
    # not move. The seed is the round index, so each round reshuffles the pool
    # before splitting it. Reusing last round's shards would hand every worker
    # the same images every time, which is exactly the class-correlated split
    # that biases local gradients before aggregation sees them.
    if shard_dir.exists():
        shutil.rmtree(shard_dir, ignore_errors=True)
    shards = sharding.build_proportional_shards(
        Path(dataset["extracted_path"]),
        shard_dir,
        sizes,
        run["class_names"],
        seed=round_index,
    )
    _prune_old_rounds(run_id, round_index)

    now = time.time()
    created: List[dict] = []
    throughput_by_node = {node["node_id"]: float(node.get("throughput_sps") or 0.0) for node in pool}

    for assignment, shard in zip(plan.assignments, shards):
        # A node that has never finished a round is still paying setup costs, so
        # it gets the cold-start grace period rather than a probe-derived one.
        cold = throughput_by_node.get(assignment.node_id, 0.0) <= 0.0
        deadline = deadline_for(assignment.predicted_seconds, policy, cold=cold)
        batch_id = uuid.uuid4().hex[:12]
        created.append(
            {
                "batch_id": batch_id,
                "run_id": run_id,
                "node_id": assignment.node_id,
                "round_index": round_index,
                "shard_index": shard["shard_index"],
                "samples": shard["train_count"],
                "status": "queued",
                "tier": assignment.tier,
                "fitness": assignment.fitness,
                "predicted_seconds": assignment.predicted_seconds,
                "soft_deadline_seconds": deadline.soft_seconds,
                "hard_deadline_seconds": deadline.hard_seconds,
                "queued_at": now,
                "assigned_at": None,
                "finished_at": None,
                "elapsed_seconds": None,
                "memory_mb": max(1024, run["batch_size"] * 160),
                "max_batch_size": next(
                    (n.get("max_batch_size", run["batch_size"]) for n in pool if n["node_id"] == assignment.node_id),
                    run["batch_size"],
                ),
                "weights_b64": None,
                "metrics": None,
                "error": None,
                "speculative_for": None,
            }
        )

    with state_lock:
        run["status"] = "running"
        run["plan"] = plan.as_dict()
        run["shards"] = shards
        run["shard_sizes"] = sizes
        run["round_started_at"] = now
        for batch in created:
            batches[batch["batch_id"]] = batch

    emit(
        "round.started",
        {
            "run_id": run_id,
            "round": round_index,
            "total_rounds": run["rounds"],
            "plan": plan.as_dict(),
        },
    )


def _prune_old_rounds(run_id: str, keep_round: int) -> None:
    """Shard copies are the largest thing on disk. Keep only the live round."""
    root = store.run_dir(run_id)
    for child in root.glob("round_*"):
        if child.name != ("round_%d" % keep_round):
            shutil.rmtree(child, ignore_errors=True)


def _supersede_twins_locked(winner: dict) -> None:
    """Retire the other half of a speculative pair once one side finishes."""
    for other in batches.values():
        if other is winner:
            continue
        if other["run_id"] != winner["run_id"]:
            continue
        if other["round_index"] != winner["round_index"]:
            continue
        if other["shard_index"] != winner["shard_index"]:
            continue
        if other["status"] not in {"queued", "assigned"}:
            continue
        other["status"] = "superseded"
        other["finished_at"] = time.time()
        node = nodes.get(other["node_id"])
        if node is not None:
            node["active_batches"] = 0
            node["allocated_memory_mb"] = 0
        emit(
            "shard.superseded",
            {
                "run_id": other["run_id"],
                "batch_id": other["batch_id"],
                "node_id": other["node_id"],
                "winner": winner["batch_id"],
            },
        )


def _round_batches(run_id: str, round_index: int) -> List[dict]:
    return [
        batch
        for batch in batches.values()
        if batch["run_id"] == run_id and batch["round_index"] == round_index
    ]


def _maybe_close_round(run_id: str) -> None:
    """Close the barrier once every shard has resolved one way or another."""
    with state_lock:
        run = runs.get(run_id)
        if run is None or run["status"] not in {"running", "waiting"}:
            return
        round_index = run["current_round"]
        round_batches = _round_batches(run_id, round_index)
        if not round_batches:
            return
        if any(batch["status"] in {"queued", "assigned"} for batch in round_batches):
            return
        if run.get("closing"):
            return
        run["closing"] = True

        # Count each shard index once. A speculative pair is two batches for one
        # slice of the dataset, so counting both would double the round's
        # sample total and understate how much data was actually lost.
        done = [
            batch
            for batch in round_batches
            if batch["status"] == "done" and batch.get("weights_b64")
        ]
        finished_shards = {batch["shard_index"] for batch in done}
        lost = [
            batch
            for batch in round_batches
            if batch["status"] in {"failed", "dropped"}
            and batch["shard_index"] not in finished_shards
        ]
        dropped_samples = sum(batch["samples"] for batch in _unique_by_shard(lost))
        total = dropped_samples + sum(batch["samples"] for batch in done)
        policy = current_policy()

    if not done:
        _fail_round(run_id, "every shard in this round failed or was dropped")
        return
    if should_abort_round(dropped_samples, total, policy):
        _fail_round(
            run_id,
            "%d of %d samples were lost to offline workers, which is too much for a valid aggregation"
            % (dropped_samples, total),
        )
        return

    aggregator.submit(_aggregate_round, run_id, round_index)


def _unique_by_shard(items: List[dict]) -> List[dict]:
    seen: Dict[int, dict] = {}
    for item in items:
        seen.setdefault(item["shard_index"], item)
    return list(seen.values())


def _fail_round(run_id: str, reason: str) -> None:
    with state_lock:
        run = runs.get(run_id)
        if run is None:
            return
        attempts = run.get("round_attempts", 0) + 1
        run["round_attempts"] = attempts
        run["closing"] = False
        should_retry = attempts < MAX_ROUND_ATTEMPTS

    emit("round.failed", {"run_id": run_id, "reason": reason, "retrying": should_retry})

    if should_retry:
        _clear_round(run_id)
        _start_round(run_id)
        return

    with state_lock:
        run = runs.get(run_id)
        if run is not None:
            run["status"] = "failed"
            run["error"] = reason
            run["finished_at"] = time.time()
    emit("run.failed", {"run_id": run_id, "error": reason})
    _archive_run(run_id)


def _clear_round(run_id: str) -> None:
    with state_lock:
        run = runs.get(run_id)
        if run is None:
            return
        for batch_id in [b["batch_id"] for b in _round_batches(run_id, run["current_round"])]:
            batches.pop(batch_id, None)


def _aggregate_round(run_id: str, round_index: int) -> None:
    """Sample-weighted FedAvg. Runs off the event loop because torch blocks."""
    started = time.time()
    with state_lock:
        run = runs.get(run_id)
        if run is None:
            return
        round_batches = _round_batches(run_id, round_index)
        done = _unique_by_shard(
            [
                batch
                for batch in round_batches
                if batch["status"] == "done" and batch.get("weights_b64")
            ]
        )
        results = [
            {
                "batch_id": batch["batch_id"],
                "samples": batch["samples"],
                "reliability": (nodes.get(batch["node_id"]) or {}).get("reliability", 1.0),
                "weights_b64": batch["weights_b64"],
            }
            for batch in done
        ]
        round_started_at = run.get("round_started_at", started)

    try:
        weights = aggregation_weights(results)
        encoded = aggregation.aggregate(results, weights)
    except Exception as exc:
        _fail_round(run_id, "aggregation failed: %s: %s" % (type(exc).__name__, exc))
        return

    aggregation_seconds = time.time() - started
    makespan = max((batch.get("elapsed_seconds") or 0.0) for batch in done) if done else 0.0
    fastest = min((batch.get("elapsed_seconds") or 0.0) for batch in done) if done else 0.0
    serial_estimate = sum(batch.get("elapsed_seconds") or 0.0 for batch in done)
    wall_clock = time.time() - round_started_at

    record = {
        "round": round_index,
        "workers": len(done),
        "samples": sum(batch["samples"] for batch in done),
        "dropped_shards": len(round_batches) - len(done),
        "makespan_seconds": round(makespan, 2),
        "fastest_seconds": round(fastest, 2),
        "straggler_gap_seconds": round(makespan - fastest, 2),
        "aggregation_seconds": round(aggregation_seconds, 2),
        "wall_clock_seconds": round(wall_clock, 2),
        # Serial time is the sum of the shard times actually observed, which is
        # what one machine would have spent doing all of this work at the
        # measured per-shard rates.
        "serial_estimate_seconds": round(serial_estimate, 2),
        "speedup": round(serial_estimate / wall_clock, 3) if wall_clock > 0 else 0.0,
        "efficiency": round(efficiency(serial_estimate / wall_clock if wall_clock else 0.0, max(1, len(done))), 3),
        "shards": [
            {
                "node_id": batch["node_id"],
                "node_name": (nodes.get(batch["node_id"]) or {}).get("display_name"),
                "samples": batch["samples"],
                "seconds": round(batch.get("elapsed_seconds") or 0.0, 2),
                "predicted_seconds": batch["predicted_seconds"],
                "tier": batch["tier"],
                "weight": round(weights.get(batch["batch_id"], 0.0), 4),
                "metrics": batch.get("metrics"),
            }
            for batch in done
        ],
    }

    with state_lock:
        run = runs.get(run_id)
        if run is None:
            return
        run["weights_b64"] = encoded
        run["round_history"].append(record)
        run["current_round"] = round_index + 1
        run["round_attempts"] = 0
        run["closing"] = False
        finished = run["current_round"] >= run["rounds"]
        if finished:
            run["status"] = "done"
            run["finished_at"] = time.time()

    _clear_round(run_id)
    _write_artifact(run_id, encoded)
    emit("round.completed", {"run_id": run_id, **record})

    if finished:
        emit("run.completed", {"run_id": run_id, "summary": _run_summary(runs[run_id])})
        _archive_run(run_id)
    else:
        _start_round(run_id)


def _write_artifact(run_id: str, encoded_weights: str) -> None:
    try:
        path = store.run_dir(run_id) / "global.pt"
        path.write_bytes(base64.b64decode(encoded_weights.encode("ascii")))
    except Exception:
        # An unwritable artifact must not fail a run that otherwise succeeded.
        pass


def _archive_run(run_id: str) -> None:
    with state_lock:
        run = runs.get(run_id)
        if run is None:
            return
        record = _run_summary(run)
        record["round_history"] = run["round_history"]
        record["plan"] = run.get("plan")
        record["error"] = run.get("error")
        record["notes"] = run.get("notes")
    store.put_run(record)
    try:
        (store.run_dir(run_id) / "summary.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


def _run_summary(run: dict) -> dict:
    history = run.get("round_history") or []
    total_wall = sum(item.get("wall_clock_seconds", 0.0) for item in history)
    total_serial = sum(item.get("serial_estimate_seconds", 0.0) for item in history)
    peak_workers = max((item.get("workers", 0) for item in history), default=0)
    return {
        "id": run["id"],
        "name": run["name"],
        "status": run["status"],
        "mode": run["mode"],
        "created_at": run["created_at"],
        "finished_at": run.get("finished_at"),
        "dataset_id": run["dataset_id"],
        "dataset_name": run["dataset_name"],
        "base_model": run["base_model"],
        "rounds": run["rounds"],
        "current_round": run["current_round"],
        "imgsz": run["imgsz"],
        "batch_size": run["batch_size"],
        "total_samples": run["total_samples"],
        "wall_clock_seconds": round(total_wall, 2),
        "serial_estimate_seconds": round(total_serial, 2),
        "speedup": round(total_serial / total_wall, 3) if total_wall > 0 else 0.0,
        "efficiency": round(efficiency(total_serial / total_wall if total_wall else 0.0, max(1, peak_workers)), 3),
        "peak_workers": peak_workers,
        "has_artifact": (store.run_dir(run["id"]) / "global.pt").is_file(),
    }


def _run_view(run_id: str) -> dict:
    with state_lock:
        run = runs[run_id]
        view = _run_summary(run)
        view["round_history"] = run["round_history"]
        view["plan"] = run.get("plan")
        view["error"] = run.get("error")
        view["notes"] = run.get("notes")
        view["class_names"] = run["class_names"]
        view["live_shards"] = [
            {
                "batch_id": batch["batch_id"],
                "node_id": batch["node_id"],
                "node_name": (nodes.get(batch["node_id"]) or {}).get("display_name"),
                "status": batch["status"],
                "samples": batch["samples"],
                "tier": batch["tier"],
                "round": batch["round_index"],
                "predicted_seconds": batch["predicted_seconds"],
                "elapsed_seconds": round(time.time() - batch["assigned_at"], 1)
                if batch.get("assigned_at") and batch["status"] == "assigned"
                else batch.get("elapsed_seconds"),
                "soft_deadline_seconds": batch["soft_deadline_seconds"],
                "hard_deadline_seconds": batch["hard_deadline_seconds"],
                "error": batch.get("error"),
            }
            for batch in _round_batches(run_id, run["current_round"])
        ]
        return view


# ---------------------------------------------------------------------------
# Straggler supervisor
# ---------------------------------------------------------------------------


async def _supervisor_loop() -> None:
    """Enforce deadlines and keep waiting runs moving when workers arrive."""
    while True:
        try:
            await asyncio.sleep(SUPERVISOR_INTERVAL_SECONDS)
            await asyncio.get_running_loop().run_in_executor(None, _supervise_once)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # keep the supervisor alive through any bug
            emit("supervisor.error", {"error": "%s: %s" % (type(exc).__name__, exc)})


def _supervise_once() -> None:
    now = time.time()
    policy = current_policy()
    to_close: List[str] = []
    to_start: List[str] = []

    with state_lock:
        _refresh_liveness(now)

        idle_nodes = [
            node
            for node in nodes.values()
            if node.get("active") and node.get("active_batches", 0) == 0 and node.get("supports_training", True)
        ]

        for batch in list(batches.values()):
            if batch["status"] == "queued":
                node = nodes.get(batch["node_id"])
                if node is None or not node.get("active"):
                    stale = now - float(batch.get("queued_at") or now)
                    if stale > HEARTBEAT_TIMEOUT_SECONDS:
                        batch["status"] = "dropped"
                        batch["error"] = "assigned worker never came online"
                        emit(
                            "shard.dropped",
                            {"run_id": batch["run_id"], "batch_id": batch["batch_id"], "reason": batch["error"]},
                        )
                        to_close.append(batch["run_id"])
                continue

            if batch["status"] != "assigned":
                continue

            elapsed = now - float(batch.get("assigned_at") or now)
            node = nodes.get(batch["node_id"])
            # Use the deadline the round was planned with, so a policy change
            # mid-round cannot retroactively make a running shard late.
            deadline = Deadline(
                soft_seconds=batch["soft_deadline_seconds"],
                hard_seconds=batch["hard_deadline_seconds"],
            )

            if node is None or not node.get("active"):
                batch["status"] = "dropped"
                batch["error"] = "worker went offline mid-round"
                if node is not None:
                    node["reliability"] = update_reliability(node, False, policy)
                emit(
                    "shard.dropped",
                    {"run_id": batch["run_id"], "batch_id": batch["batch_id"], "reason": batch["error"]},
                )
                to_close.append(batch["run_id"])
                continue

            backup = next(
                (
                    candidate
                    for candidate in idle_nodes
                    if candidate["node_id"] != batch["node_id"] and not batch.get("speculated")
                ),
                None,
            )
            action = straggler_action(elapsed, deadline, backup is not None)

            if action == "drop":
                batch["status"] = "dropped"
                batch["error"] = "missed the hard deadline of %.0fs" % deadline.hard_seconds
                node["reliability"] = update_reliability(node, False, policy)
                node["active_batches"] = 0
                node["allocated_memory_mb"] = 0
                emit(
                    "shard.dropped",
                    {
                        "run_id": batch["run_id"],
                        "batch_id": batch["batch_id"],
                        "node_id": batch["node_id"],
                        "reason": batch["error"],
                    },
                )
                to_close.append(batch["run_id"])
            elif action == "speculate" and backup is not None:
                batch["speculated"] = True
                clone_id = uuid.uuid4().hex[:12]
                clone = dict(batch)
                clone.update(
                    {
                        "batch_id": clone_id,
                        "node_id": backup["node_id"],
                        "status": "queued",
                        "queued_at": now,
                        "assigned_at": None,
                        "speculative_for": batch["batch_id"],
                        "speculated": False,
                    }
                )
                batches[clone_id] = clone
                idle_nodes = [n for n in idle_nodes if n["node_id"] != backup["node_id"]]
                emit(
                    "shard.speculated",
                    {
                        "run_id": batch["run_id"],
                        "original": batch["batch_id"],
                        "clone": clone_id,
                        "from_node": batch["node_id"],
                        "to_node": backup["node_id"],
                        "elapsed_seconds": round(elapsed, 1),
                    },
                )

        for run in runs.values():
            if run["status"] == "waiting" and idle_nodes:
                to_start.append(run["id"])

    for run_id in dict.fromkeys(to_close):
        _maybe_close_round(run_id)
    for run_id in dict.fromkeys(to_start):
        _start_round(run_id)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


@app.get("/datasets")
def get_datasets(_: str = Depends(require_mesh_token)):
    return {"datasets": store.list_datasets()}


@app.post("/datasets")
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    make_default: bool = Form(default=False),
    _: str = Depends(require_mesh_token),
):
    """Accept a YOLO dataset ZIP, validate it, and register it for the mesh."""
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Upload a .zip archive")

    dataset_id = uuid.uuid4().hex[:12]
    target = store.dataset_dir(dataset_id)
    extracted = target / "data"
    target.mkdir(parents=True, exist_ok=True)
    archive_path = target / "dataset.zip"

    size = 0
    with archive_path.open("wb") as stream:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            stream.write(chunk)

    try:
        with zipfile.ZipFile(archive_path) as archive:
            _safe_extract(archive, extracted)
        listing = sharding.list_split_images(extracted)
        train_count = len(listing["train"])
        val_count = len(listing["val"])
        if train_count == 0:
            raise ValueError("The archive contains no training images")
        class_names = sharding.infer_class_names(extracted) or ["object"]
    except Exception as exc:
        shutil.rmtree(target, ignore_errors=True)
        raise HTTPException(
            status_code=400,
            detail="Could not read this dataset: %s. Expected images/train and labels/train inside the zip."
            % exc,
        )

    record = {
        "id": dataset_id,
        "name": name.strip() or file.filename,
        "filename": file.filename,
        "created_at": time.time(),
        "bytes": size,
        "train_count": train_count,
        "val_count": val_count,
        "class_names": class_names,
        "extracted_path": str(extracted),
        "archive_path": str(archive_path),
    }
    store.put_dataset(record, make_default=make_default)
    emit("dataset.added", {"id": dataset_id, "name": record["name"], "images": train_count})
    return record


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    """Reject archives that try to escape the extraction root."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    for member in archive.infolist():
        target = (root / member.filename).resolve()
        if not str(target).startswith(str(root)):
            raise ValueError("archive contains a path outside the extraction root")
    archive.extractall(root)


@app.post("/datasets/{dataset_id}/default")
def make_default_dataset(dataset_id: str, _: str = Depends(require_mesh_token)):
    if store.get_dataset(dataset_id) is None:
        raise HTTPException(status_code=404, detail="Unknown dataset")
    store.set_default_dataset(dataset_id)
    emit("dataset.default", {"id": dataset_id})
    return {"status": "ok"}


@app.delete("/datasets/{dataset_id}")
def remove_dataset(dataset_id: str, _: str = Depends(require_mesh_token)):
    with state_lock:
        in_use = any(
            run["dataset_id"] == dataset_id and run["status"] in {"running", "planning", "waiting"}
            for run in runs.values()
        )
    if in_use:
        raise HTTPException(status_code=409, detail="This dataset is in use by a running job")
    if not store.delete_dataset(dataset_id):
        raise HTTPException(status_code=404, detail="Unknown dataset")
    emit("dataset.removed", {"id": dataset_id})
    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# Models, policy, mesh state, events
# ---------------------------------------------------------------------------


@app.get("/models")
def list_models(_: str = Depends(require_mesh_token)):
    models = [
        {"name": path.name, "bytes": path.stat().st_size}
        for path in sorted(store.MODELS_DIR.glob("*.pt"))
    ]
    return {"models": models}


@app.get("/models/{model_name}")
def get_model(model_name: str, _: str = Depends(require_mesh_token)):
    path = (store.MODELS_DIR / Path(model_name).name).resolve()
    if path.parent != store.MODELS_DIR.resolve() or path.suffix != ".pt" or not path.is_file():
        raise HTTPException(status_code=404, detail="Model checkpoint not found")
    return FileResponse(path, media_type="application/octet-stream", filename=path.name)


@app.get("/policy")
def get_policy(_: str = Depends(require_mesh_token)):
    return {"policy": current_policy().as_dict(), "defaults": DEFAULT_POLICY.as_dict()}


@app.put("/policy")
def put_policy(req: PolicyRequest, _: str = Depends(require_mesh_token)):
    allowed = set(vars(DEFAULT_POLICY))
    unknown = set(req.values) - allowed
    if unknown:
        raise HTTPException(status_code=400, detail="Unknown policy keys: %s" % ", ".join(sorted(unknown)))
    store.set_policy(req.values)
    emit("policy.updated", {"values": req.values})
    return {"policy": current_policy().as_dict()}


@app.post("/token/rotate")
def rotate_token(_: str = Depends(require_mesh_token)):
    """Mint a new join token and drop every machine that used the old one.

    This is the revocation story for a LAN mesh: the token is what a worker
    presents to receive dataset shards, so rotating it and clearing the registry
    means an uninvited machine cannot keep pulling data.
    """
    new_token = store.rotate_mesh_token()
    with state_lock:
        removed = list(nodes.keys())
        nodes.clear()
        for batch in batches.values():
            if batch["status"] in {"queued", "assigned"}:
                batch["status"] = "dropped"
                batch["error"] = "the mesh token was rotated"
    emit("mesh.token_rotated", {"removed": len(removed)})
    return {"token": new_token, "removed_nodes": len(removed)}


@app.get("/mesh")
def mesh_state(_: str = Depends(require_mesh_token)):
    """Everything the dashboard needs for a cold render, in one request."""
    policy = current_policy()
    node_views = snapshot_nodes()
    decisions = admit([n for n in node_views], policy)

    for view in node_views:
        decision = decisions.get(view["node_id"])
        if decision:
            view["tier"] = decision.tier
            view["admission_reason"] = decision.reason

    with state_lock:
        active_runs = [_run_summary(run) for run in runs.values() if run["status"] in {"running", "planning", "waiting"}]
        live_batches = [
            batch for batch in batches.values() if batch["status"] in {"queued", "assigned"}
        ]

    dataset = store.default_dataset()
    preview = plan_round(node_views, int((dataset or {}).get("train_count") or 0), policy)

    total_gflops = sum(float((n.get("capability") or {}).get("gflops") or 0.0) for n in node_views if n.get("active"))
    total_memory = sum(int(n.get("gpu_memory_mb") or 0) for n in node_views if n.get("active"))

    return {
        "mesh_name": store.load().get("mesh_name", "GradMesh"),
        "nodes": node_views,
        "metrics": {
            "nodes_total": len(node_views),
            "nodes_active": sum(1 for n in node_views if n.get("active")),
            "nodes_training": sum(1 for n in node_views if n.get("active_batches", 0) > 0),
            "nodes_admitted": sum(1 for d in decisions.values() if d.tier != "rejected"),
            "total_gflops": round(total_gflops, 1),
            "total_memory_mb": total_memory,
            "active_shards": len(live_batches),
            "stream_subscribers": bus.subscriber_count,
        },
        "active_runs": active_runs,
        "plan_preview": preview.as_dict(),
        "policy": policy.as_dict(),
        "torch_ready": aggregation.torch_ready(),
        "default_dataset": dataset,
    }


@app.get("/events")
async def events(replay: int = Query(default=40, ge=0, le=200), _: str = Depends(require_mesh_token)):
    return StreamingResponse(
        bus.subscribe(replay=replay),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class VisitorRequest(BaseModel):
    visitor_id: str = Field(..., min_length=6, max_length=64)
    name: Optional[str] = None
    platform: Optional[str] = None
    user_agent: Optional[str] = None
    cores: Optional[int] = None
    memory_gb: Optional[float] = None
    gpu: Optional[str] = None
    webgpu: Optional[bool] = None
    screen: Optional[str] = None


def _prune_visitors_locked(now: float) -> None:
    for visitor_id in [
        key for key, value in visitors.items() if now - value["last_seen"] > VISITOR_TTL_SECONDS
    ]:
        visitors.pop(visitor_id, None)


@app.post("/visitors")
def announce_visitor(req: VisitorRequest, request: Request, _: str = Depends(allow_local_or_token)):
    """A browser on the network says hello.

    Opening the join page is the earliest moment the host can know a device
    exists, and it costs the visitor nothing. The record is deliberately thin:
    what the browser already exposes to any page it loads, plus whether WebGPU
    is available, which is the honest upper bound on what a tab could ever
    contribute without installing the agent.
    """
    now = time.time()
    client_ip = request.client.host if request.client else None

    # The endpoint is reachable without a token, so cap how often one address
    # can write to the list regardless of how many visitor ids it invents.
    bucket = _visitor_rate.setdefault(client_ip or "unknown", [])
    bucket[:] = [stamp for stamp in bucket if now - stamp < VISITOR_RATE_WINDOW]
    if len(bucket) >= VISITOR_RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many announcements.")
    bucket.append(now)

    with state_lock:
        _prune_visitors_locked(now)
        existing = visitors.get(req.visitor_id) or {}
        is_new = not existing
        has_agent = any(
            (node.get("capability") or {}).get("host") == req.name for node in nodes.values()
        )
        visitors[req.visitor_id] = {
            "visitor_id": req.visitor_id,
            "name": req.name or "Unnamed device",
            "platform": req.platform,
            "user_agent": req.user_agent,
            "cores": req.cores,
            "memory_gb": req.memory_gb,
            "gpu": req.gpu,
            "webgpu": bool(req.webgpu),
            "screen": req.screen,
            "ip": client_ip,
            "first_seen": existing.get("first_seen", now),
            "last_seen": now,
            "has_agent": has_agent,
        }

    if is_new:
        emit(
            "visitor.arrived",
            {"visitor_id": req.visitor_id, "name": req.name, "ip": client_ip, "gpu": req.gpu},
        )
    return {"status": "ok", "mdns": advertiser.as_dict()}


def _visitor_list_locked(now: float) -> List[dict]:
    _prune_visitors_locked(now)
    return sorted(visitors.values(), key=lambda item: item["last_seen"], reverse=True)


def _run_scan() -> dict:
    try:
        result = discovery.scan_network()
    finally:
        _scan_cache["running"] = False
    _scan_cache["result"] = result
    _scan_cache["at"] = time.time()
    emit("network.scanned", {"devices": len(result.get("devices", []))})
    return result


@app.get("/discover")
def discover(refresh: bool = Query(default=False), _: str = Depends(require_mesh_token)):
    """Everything the Discover page needs: how to reach this host, who is on the
    network, who is already contributing, and who is merely looking."""
    now = time.time()
    cached = _scan_cache["result"]
    stale = cached is None or (now - float(_scan_cache["at"])) > SCAN_CACHE_SECONDS

    if (refresh or stale) and not _scan_cache["running"]:
        _scan_cache["running"] = True
        if cached is None:
            # Nothing to show yet, so the first caller waits for a real answer
            # rather than being handed an empty page.
            cached = _run_scan()
        else:
            scanner.submit(_run_scan)

    with state_lock:
        _refresh_liveness(now)
        node_summaries = [
            {
                "node_id": node["node_id"],
                "display_name": node.get("display_name"),
                "gpu": node.get("gpu"),
                "backend": node.get("backend"),
                "active": node.get("active"),
                "host": (node.get("capability") or {}).get("host"),
            }
            for node in nodes.values()
        ]
        visitor_records = _visitor_list_locked(now)

    scan = cached or {"devices": [], "subnet": None, "self_ip": discovery.local_ipv4()}
    visitor_ips = {visitor["ip"] for visitor in visitor_records if visitor.get("ip")}
    agent_ips = {
        visitor["ip"] for visitor in visitor_records if visitor.get("ip") and visitor.get("has_agent")
    }

    devices = []
    for device in scan.get("devices", []):
        entry = dict(device)
        entry["is_visitor"] = device["ip"] in visitor_ips
        entry["is_member"] = bool(device.get("is_coordinator")) or device["ip"] in agent_ips
        devices.append(entry)

    return {
        "mdns": advertiser.as_dict(),
        "addresses": {
            "lan_ip": scan.get("self_ip"),
            "web_port": advertiser.web_port,
            "api_port": advertiser.api_port,
        },
        "scan": {
            "subnet": scan.get("subnet"),
            "at": _scan_cache["at"],
            "running": bool(_scan_cache["running"]),
            "duration_seconds": scan.get("duration_seconds", 0.0),
            "scanned": scan.get("scanned", 0),
        },
        "devices": devices,
        "visitors": visitor_records,
        "nodes": node_summaries,
        "idle_count": sum(
            1
            for device in devices
            if not device["is_this_host"] and not device["is_member"] and not device["is_visitor"]
        ),
    }


@app.post("/discover/scan")
def rescan(_: str = Depends(require_mesh_token)):
    """Force a fresh sweep. Blocks for a couple of seconds by design: someone
    pressed a button and expects the list to be current when it returns."""
    _scan_cache["running"] = True
    return _run_scan()


@app.get("/health")
def health():
    """Unauthenticated so the dev launcher can wait for readiness."""
    return {
        "status": "ok",
        "version": "4.0.0",
        "torch_ready": aggregation.torch_ready(),
        "nodes_active": sum(1 for node in nodes.values() if node.get("active")),
        "mdns": advertiser.as_dict(),
    }
