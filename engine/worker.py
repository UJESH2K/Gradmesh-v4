"""GradMesh worker agent.

Runs on any machine that wants to lend its GPU to the mesh. It registers with
the coordinator, reports a measured capability profile, then polls for shards
and trains them.

The training call itself is deliberately unchanged from v3. train_batch()
downloads a shard, loads the global state dict with the same shape-compatible
filter, runs the same Ultralytics train options, and takes the same Intel XPU
path through ultralytics_xpu.xpu_train. That pipeline is validated; v4 only
changes what surrounds it.

What is new: a capability probe at startup, the mesh token on every request,
coordinator auto-discovery on the local subnet, and clean deregistration on
exit so the mesh does not have to wait out a heartbeat timeout.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import socket
import sys
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib import error, request

ENGINE_DIR = Path(__file__).resolve().parent
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

from accelerator import Accelerator, detect_accelerator
from federated_training import decode_state_dict, encode_state_dict

torch = None
ACCELERATOR: Accelerator | None = None

SERVER_URL = os.getenv("GRADMESH_SERVER", "http://127.0.0.1:8000").rstrip("/")
MESH_TOKEN = os.getenv("GRADMESH_TOKEN", "")
POLL_SECONDS = float(os.getenv("GRADMESH_POLL_SECONDS", "1.5"))
HEARTBEAT_SECONDS = float(os.getenv("GRADMESH_HEARTBEAT_SECONDS", "5"))
NODE_ID = os.getenv("GRADMESH_NODE_ID", "")
MAX_BATCH_SIZE = int(os.getenv("GRADMESH_MAX_BATCH_SIZE", "4"))
GPU_MEMORY_MB = int(os.getenv("GRADMESH_GPU_MEMORY_MB", "8000"))
DISPLAY_NAME = os.getenv("GRADMESH_NAME", "")
OWNER = os.getenv("GRADMESH_OWNER", "")
BACKEND = os.getenv("GRADMESH_BACKEND", "auto")

AGENT_VERSION = "4.0.0"
STATE_FILE = Path.home() / ".gradmesh" / "worker.json"


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


def _headers(extra: dict | None = None) -> dict:
    headers = {"X-Mesh-Token": MESH_TOKEN, "User-Agent": "gradmesh-worker/%s" % AGENT_VERSION}
    headers.update(extra or {})
    return headers


def http_post(path: str, payload: dict, timeout: int = 15) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        SERVER_URL + path,
        data=data,
        headers=_headers({"Content-Type": "application/json"}),
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return json.loads(body) if body else {}


def http_get(path: str, timeout: int = 15) -> dict:
    req = request.Request(SERVER_URL + path, headers=_headers(), method="GET")
    with request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return json.loads(body) if body else {}


def download_bytes(path: str, timeout: int = 300) -> bytes:
    req = request.Request(SERVER_URL + path, headers=_headers(), method="GET")
    with request.urlopen(req, timeout=timeout) as response:
        return response.read()


def download_model(model_name: str, destination: Path) -> Path:
    model_path = Path(model_name)
    if model_path.is_file():
        return model_path

    safe_name = model_path.name
    if model_path.suffix.lower() != ".pt":
        raise FileNotFoundError("Model must be a .pt checkpoint: %s" % model_name)

    # Cache checkpoints across rounds so a multi-round run pays the transfer once.
    cache = Path.home() / ".gradmesh" / "models"
    cache.mkdir(parents=True, exist_ok=True)
    cached = cache / safe_name
    if cached.is_file() and cached.stat().st_size > 0:
        return cached

    destination.mkdir(parents=True, exist_ok=True)
    payload = download_bytes("/models/" + safe_name, timeout=600)
    cached.write_bytes(payload)
    return cached


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def local_ipv4() -> str:
    """The address this machine uses to reach the LAN, without sending traffic."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("10.255.255.255", 1))
        return probe.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        probe.close()


def discover_coordinator(port: int = 8000, timeout: float = 0.35) -> str | None:
    """Scan the local /24 for a GradMesh coordinator.

    Used when a contributor's saved host address has changed, which happens
    every time the router hands out a new lease. Checking 254 addresses in
    parallel takes well under a second on a normal home network.
    """
    base = local_ipv4()
    if base.startswith("127."):
        return None
    prefix = base.rsplit(".", 1)[0]

    def check(host: str) -> str | None:
        url = "http://%s:%d/health" % (host, port)
        try:
            req = request.Request(url, method="GET")
            with request.urlopen(req, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
                if payload.get("status") == "ok" and "version" in payload:
                    return "http://%s:%d" % (host, port)
        except Exception:
            return None
        return None

    candidates = ["%s.%d" % (prefix, octet) for octet in range(1, 255)]
    with ThreadPoolExecutor(max_workers=64) as pool:
        for result in pool.map(check, candidates):
            if result:
                return result
    return None


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def stable_node_id() -> str:
    """Reuse the same node id across restarts so reputation survives a reboot."""
    if NODE_ID:
        return NODE_ID
    try:
        if STATE_FILE.is_file():
            saved = json.loads(STATE_FILE.read_text(encoding="utf-8")).get("node_id")
            if saved:
                return saved
    except Exception:
        pass
    generated = uuid.uuid4().hex[:16]
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps({"node_id": generated}), encoding="utf-8")
    except Exception:
        pass
    return generated


# ---------------------------------------------------------------------------
# Registration and heartbeat
# ---------------------------------------------------------------------------


def register(capability: dict) -> dict:
    return http_post(
        "/register_node",
        {
            "node_id": NODE_ID,
            "display_name": DISPLAY_NAME or socket.gethostname(),
            "gpu": capability.get("device_name", "unknown"),
            "gpu_memory_mb": int(capability.get("total_memory_mb") or GPU_MEMORY_MB),
            "max_batch_size": MAX_BATCH_SIZE,
            "backend": capability.get("backend", "cpu"),
            "supports_training": bool(capability.get("supports_training")),
            "capability": capability,
            "owner": OWNER or None,
            "agent_version": AGENT_VERSION,
        },
    )


def current_load(active_batches: int) -> float:
    if ACCELERATOR is not None and ACCELERATOR.supports_training:
        return 0.1 if active_batches == 0 else 0.9
    return 0.05 if active_batches == 0 else 0.8


def send_heartbeat(
    active_batches: int = 0,
    allocated_memory_mb: int = 0,
    training_epoch: int | None = None,
    training_total_epochs: int | None = None,
) -> None:
    started = time.perf_counter()
    http_post(
        "/heartbeat",
        {
            "node_id": NODE_ID,
            "load": current_load(active_batches),
            "active_batches": active_batches,
            "allocated_memory_mb": allocated_memory_mb,
            "training_epoch": training_epoch,
            "training_total_epochs": training_total_epochs,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        },
        timeout=10,
    )


def deregister() -> None:
    try:
        http_post("/leave", {"node_id": NODE_ID}, timeout=5)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_batch(batch: dict) -> dict:
    """Train one shard for one round.

    Preserved from the v3 worker. The only additions are the shard sample count
    echoed back for sample-weighted aggregation, and timing for the
    coordinator's throughput model.
    """
    if YOLO is None:
        raise RuntimeError("ultralytics is required for training batches")

    started = time.time()
    shard_bytes = download_bytes(batch["shard_url"], timeout=600)
    weights_payload = http_get(batch["weights_url"], timeout=120)

    with TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        shard_zip_path = temp_root / "shard.zip"
        shard_zip_path.write_bytes(shard_bytes)

        extract_dir = temp_root / "shard"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(shard_zip_path, "r") as archive:
            archive.extractall(extract_dir)

        data_yaml = extract_dir / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError("training shard is missing data.yaml")

        # Shards are generated on the coordinator, so their YAML carries the
        # coordinator's absolute path. Each worker extracts into a different
        # temporary directory and must point YOLO at that copy.
        import yaml

        with data_yaml.open("r", encoding="utf-8") as stream:
            dataset_config = yaml.safe_load(stream) or {}
        dataset_config["path"] = str(extract_dir.resolve())
        with data_yaml.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(dataset_config, stream, sort_keys=False)

        model_path = download_model(batch.get("base_model", "yolov8n.pt"), temp_root / "models")
        model = YOLO(str(model_path))

        weights_b64 = weights_payload.get("weights_b64")
        if weights_b64:
            state_dict = decode_state_dict(weights_b64)
            # Filter incompatible tensors, such as a class head whose shape
            # changed, to avoid load failures.
            current_state = model.model.state_dict()
            compatible_state = {
                key: value
                for key, value in state_dict.items()
                if key in current_state
                and hasattr(value, "shape")
                and hasattr(current_state[key], "shape")
                and tuple(value.shape) == tuple(current_state[key].shape)
            }
            model.model.load_state_dict(compatible_state, strict=False)

        if ACCELERATOR is None:
            raise RuntimeError("Accelerator has not been initialized")
        device = ACCELERATOR.ultralytics_device
        total_epochs = int(batch.get("epochs", 1))
        allocated_memory_mb = int(batch.get("estimated_memory_mb", 0))
        send_heartbeat(
            active_batches=1,
            allocated_memory_mb=allocated_memory_mb,
            training_epoch=0,
            training_total_epochs=total_epochs,
        )

        progress = {"epoch": 0}
        heartbeat_stop = threading.Event()

        def keep_training_alive():
            """YOLO train blocks, so its worker needs an independent heartbeat."""
            while not heartbeat_stop.wait(max(1.0, HEARTBEAT_SECONDS / 2)):
                try:
                    send_heartbeat(
                        active_batches=1,
                        allocated_memory_mb=allocated_memory_mb,
                        training_epoch=progress["epoch"],
                        training_total_epochs=total_epochs,
                    )
                except Exception as exc:
                    print("[worker] heartbeat during training failed: %s" % exc)

        def report_epoch_end(trainer):
            progress["epoch"] = int(getattr(trainer, "epoch", 0)) + 1
            try:
                send_heartbeat(
                    active_batches=1,
                    allocated_memory_mb=allocated_memory_mb,
                    training_epoch=progress["epoch"],
                    training_total_epochs=total_epochs,
                )
            except Exception:
                pass

        model.add_callback("on_fit_epoch_end", report_epoch_end)
        heartbeat_thread = threading.Thread(target=keep_training_alive, daemon=True)
        heartbeat_thread.start()

        train_seconds_start = time.time()
        try:
            train_options = {
                "data": str(data_yaml),
                "epochs": total_epochs,
                "imgsz": int(batch.get("imgsz", 640)),
                "batch": int(batch.get("batch_size", 8)),
                "project": str(temp_root / "runs"),
                "name": "round_%s" % batch.get("round_index", 0),
                "workers": 0,
                "verbose": False,
            }
            # The coordinator sets this per trial so repeated runs of the same
            # configuration are not bit-identical. Absent, Ultralytics uses 0.
            if batch.get("seed") is not None:
                train_options["seed"] = int(batch["seed"])
            if ACCELERATOR.backend == "xpu":
                from ultralytics_xpu import xpu_train

                xpu_train(model, optimizer="Adam", val=False, plots=False, **train_options)
            else:
                # Preserve the v3 CUDA/CPU Ultralytics call unchanged.
                model.train(device=device, **train_options)
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=2)

        train_seconds = time.time() - train_seconds_start
        encoded_weights = encode_state_dict(model.model.state_dict())

        return {
            "node_id": NODE_ID,
            "batch_id": batch["batch_id"],
            "round_index": int(batch.get("round_index", 0)),
            "weights_b64": encoded_weights,
            "metrics": {
                "trained": True,
                "device": str(ACCELERATOR.torch_device),
                "backend": ACCELERATOR.backend,
                "base_model": batch.get("base_model", "yolov8n.pt"),
                "samples": int(batch.get("samples") or 0),
                "train_seconds": round(train_seconds, 2),
                "total_seconds": round(time.time() - started, 2),
                "imgsz": batch.get("imgsz"),
                "batch_size": batch.get("batch_size"),
            },
        }


def submit_round_result(payload: dict) -> None:
    http_post("/submit_training_round_result", payload, timeout=300)


def submit_failure(batch_id: str, exc: Exception) -> None:
    http_post(
        "/submit_training_batch_failure",
        {
            "node_id": NODE_ID,
            "batch_id": batch_id,
            "error": ("%s: %s" % (type(exc).__name__, exc))[:4000],
        },
        timeout=30,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GradMesh worker agent")
    parser.add_argument("--server-url", default=SERVER_URL, help="Coordinator base URL")
    parser.add_argument("--token", default=MESH_TOKEN, help="Mesh join token")
    parser.add_argument("--name", default=DISPLAY_NAME, help="Display name shown in the dashboard")
    parser.add_argument("--owner", default=OWNER, help="Who is contributing this machine")
    parser.add_argument("--node-id", default=NODE_ID, help="Stable node id")
    parser.add_argument("--backend", choices=["auto", "cuda", "xpu", "cpu"], default=BACKEND)
    parser.add_argument("--max-batch-size", type=int, default=MAX_BATCH_SIZE)
    parser.add_argument("--gpu-memory-mb", type=int, default=GPU_MEMORY_MB)
    parser.add_argument("--poll-seconds", type=float, default=POLL_SECONDS)
    parser.add_argument("--heartbeat-seconds", type=float, default=HEARTBEAT_SECONDS)
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Scan the local subnet for a coordinator if the given URL is unreachable",
    )
    return parser.parse_args()


def coordinator_reachable() -> bool:
    try:
        req = request.Request(SERVER_URL + "/health", method="GET")
        with request.urlopen(req, timeout=3) as response:
            return json.loads(response.read().decode("utf-8")).get("status") == "ok"
    except Exception:
        return False


def main() -> None:
    global SERVER_URL, MESH_TOKEN, POLL_SECONDS, HEARTBEAT_SECONDS, NODE_ID
    global MAX_BATCH_SIZE, GPU_MEMORY_MB, DISPLAY_NAME, OWNER, BACKEND, ACCELERATOR, torch

    args = parse_args()
    SERVER_URL = args.server_url.rstrip("/")
    MESH_TOKEN = args.token
    POLL_SECONDS = args.poll_seconds
    HEARTBEAT_SECONDS = args.heartbeat_seconds
    NODE_ID = args.node_id
    MAX_BATCH_SIZE = args.max_batch_size
    GPU_MEMORY_MB = args.gpu_memory_mb
    DISPLAY_NAME = args.name
    OWNER = args.owner
    BACKEND = args.backend
    NODE_ID = stable_node_id()

    if not MESH_TOKEN:
        raise SystemExit(
            "A mesh token is required. Copy the join command from the dashboard's Contribute page."
        )

    if not coordinator_reachable():
        if args.discover:
            print("[worker] %s is unreachable, scanning the local network..." % SERVER_URL)
            found = discover_coordinator()
            if not found:
                raise SystemExit("No GradMesh coordinator was found on this network.")
            print("[worker] found coordinator at %s" % found)
            SERVER_URL = found
        else:
            raise SystemExit(
                "Cannot reach %s. Check that the host is on this network and the port is open."
                % SERVER_URL
            )

    import importlib

    torch = importlib.import_module("torch")
    ACCELERATOR = detect_accelerator(BACKEND, advertised_memory_mb=GPU_MEMORY_MB)

    from probe import probe

    print("[worker] measuring device capability...")
    capability = probe(ACCELERATOR).as_dict()
    print(
        "[worker] %s via %s, %d MB, %.0f GFLOP/s"
        % (
            capability["device_name"],
            capability["backend"],
            capability["total_memory_mb"],
            capability["gflops"],
        )
    )

    response = register(capability)
    admission = response.get("admission") or {}
    print(
        "[worker] joined %s as %s (%s: %s)"
        % (SERVER_URL, NODE_ID, admission.get("tier", "unknown"), admission.get("reason", ""))
    )
    if not ACCELERATOR.supports_training:
        print("[worker] no supported accelerator was found, so this node will not receive shards")

    atexit.register(deregister)

    def rejoin() -> None:
        """Re-register after the coordinator forgot us.

        The coordinator holds its node registry in memory, so a restart on the
        host drops every worker. Without this a contributor's terminal sits
        there printing 404s forever and they have to notice and restart it by
        hand, which on someone else's laptop means the machine is simply gone.
        """
        print("[worker] coordinator does not know this node, re-registering")
        register(capability)

    last_heartbeat = 0.0
    active_batches = 0
    allocated_memory_mb = 0
    current_batch: dict | None = None

    while True:
        try:
            now = time.time()
            if now - last_heartbeat >= HEARTBEAT_SECONDS:
                send_heartbeat(active_batches=active_batches, allocated_memory_mb=allocated_memory_mb)
                last_heartbeat = now

            payload = http_get("/get_batch/%s" % NODE_ID, timeout=20)
            batch = payload.get("batch")
            if not batch or not batch.get("batch_id"):
                time.sleep(POLL_SECONDS)
                continue

            current_batch = batch
            active_batches = 1
            allocated_memory_mb = int(batch.get("estimated_memory_mb") or 0)
            print(
                "[worker] round %s shard %s, %s images"
                % (batch.get("round_index"), batch.get("shard_index"), batch.get("samples"))
            )

            # The coordinator's liveness window is shorter than a cold start, so
            # heartbeat through the download and model-load phase too.
            stop = threading.Event()

            def keep_alive():
                while not stop.wait(max(1.0, HEARTBEAT_SECONDS / 2)):
                    try:
                        send_heartbeat(active_batches=1, allocated_memory_mb=allocated_memory_mb)
                    except Exception:
                        pass

            thread = threading.Thread(target=keep_alive, daemon=True)
            thread.start()
            try:
                result = train_batch(batch)
                submit_round_result(result)
                print("[worker] finished shard %s" % batch.get("shard_index"))
            finally:
                stop.set()
                thread.join(timeout=2)

            current_batch = None
            active_batches = 0
            allocated_memory_mb = 0
            send_heartbeat()
            last_heartbeat = time.time()

        except KeyboardInterrupt:
            print("\n[worker] leaving the mesh")
            deregister()
            break
        except Exception as exc:
            if isinstance(exc, error.HTTPError) and exc.code == 401:
                raise SystemExit("The mesh token was rejected. Copy a fresh join command.")
            if isinstance(exc, error.HTTPError) and exc.code == 404 and current_batch is None:
                try:
                    rejoin()
                    last_heartbeat = time.time()
                    continue
                except Exception as rejoin_error:
                    print("[worker] could not re-register: %s" % rejoin_error)
            if current_batch is not None:
                try:
                    submit_failure(current_batch["batch_id"], exc)
                except Exception:
                    pass
                current_batch = None
            active_batches = 0
            allocated_memory_mb = 0
            try:
                send_heartbeat()
            except Exception:
                pass
            print("[worker] %s: %s" % (type(exc).__name__, exc))
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
