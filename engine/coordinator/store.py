"""Durable coordinator state.

Everything that has to survive a coordinator restart lives under .gradmesh/ at
the repository root: the dataset registry, finished run records, the mesh join
token and the scheduling policy. In-flight round state stays in memory on
purpose, because a half-finished barrier is not resumable and pretending
otherwise would produce silently wrong aggregation.

A JSON file rather than a database is a deliberate choice. The whole point of
v4 is that npm run dev works on a laptop with nothing preinstalled, and a native
SQLite build is the most common way that promise breaks.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ENGINE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = ENGINE_DIR.parent
STATE_DIR = Path(os.getenv("GRADMESH_STATE_DIR", str(REPO_ROOT / ".gradmesh")))
DATASETS_DIR = STATE_DIR / "datasets"
RUNS_DIR = STATE_DIR / "runs"
MODELS_DIR = STATE_DIR / "models"
STATE_FILE = STATE_DIR / "coordinator.json"

_lock = threading.RLock()

_DEFAULT_STATE: Dict[str, Any] = {
    "version": 4,
    "created_at": None,
    "mesh_token": None,
    "mesh_name": "GradMesh",
    "policy": {},
    "datasets": {},
    "default_dataset_id": None,
    "runs": {},
}


def _ensure_dirs() -> None:
    for directory in (STATE_DIR, DATASETS_DIR, RUNS_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _atomic_write(path: Path, payload: str) -> None:
    """Write through a temp file so a crash mid-write cannot truncate state."""
    handle, temp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    except Exception:
        Path(temp_path).unlink(missing_ok=True)
        raise


def load() -> Dict[str, Any]:
    with _lock:
        _ensure_dirs()
        if not STATE_FILE.exists():
            state = dict(_DEFAULT_STATE)
            state["created_at"] = time.time()
            state["mesh_token"] = secrets.token_urlsafe(24)
            _atomic_write(STATE_FILE, json.dumps(state, indent=2))
            return state
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            state = dict(_DEFAULT_STATE)
        for key, value in _DEFAULT_STATE.items():
            state.setdefault(key, value)
        if not state.get("mesh_token"):
            state["mesh_token"] = secrets.token_urlsafe(24)
            save(state)
        return state


def save(state: Dict[str, Any]) -> None:
    with _lock:
        _ensure_dirs()
        _atomic_write(STATE_FILE, json.dumps(state, indent=2))


def update(mutator) -> Dict[str, Any]:
    """Read, mutate and persist under one lock."""
    with _lock:
        state = load()
        mutator(state)
        save(state)
        return state


# ---------------------------------------------------------------------------
# Mesh identity
# ---------------------------------------------------------------------------


def mesh_token() -> str:
    return load()["mesh_token"]


def rotate_mesh_token() -> str:
    new_token = secrets.token_urlsafe(24)

    def mutate(state: Dict[str, Any]) -> None:
        state["mesh_token"] = new_token

    update(mutate)
    return new_token


def policy() -> Dict[str, Any]:
    return load().get("policy") or {}


def set_policy(values: Dict[str, Any]) -> Dict[str, Any]:
    def mutate(state: Dict[str, Any]) -> None:
        merged = dict(state.get("policy") or {})
        merged.update(values)
        state["policy"] = merged

    return update(mutate).get("policy") or {}


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def dataset_dir(dataset_id: str) -> Path:
    return DATASETS_DIR / dataset_id


def list_datasets() -> List[dict]:
    state = load()
    default_id = state.get("default_dataset_id")
    datasets = []
    for record in state.get("datasets", {}).values():
        item = dict(record)
        item["is_default"] = record.get("id") == default_id
        datasets.append(item)
    datasets.sort(key=lambda item: item.get("created_at") or 0, reverse=True)
    return datasets


def get_dataset(dataset_id: str) -> Optional[dict]:
    return (load().get("datasets") or {}).get(dataset_id)


def default_dataset() -> Optional[dict]:
    state = load()
    default_id = state.get("default_dataset_id")
    if default_id:
        return (state.get("datasets") or {}).get(default_id)
    datasets = list_datasets()
    return datasets[0] if datasets else None


def put_dataset(record: dict, make_default: bool = False) -> dict:
    def mutate(state: Dict[str, Any]) -> None:
        state.setdefault("datasets", {})[record["id"]] = record
        if make_default or not state.get("default_dataset_id"):
            state["default_dataset_id"] = record["id"]

    update(mutate)
    return record


def set_default_dataset(dataset_id: str) -> None:
    def mutate(state: Dict[str, Any]) -> None:
        if dataset_id in (state.get("datasets") or {}):
            state["default_dataset_id"] = dataset_id

    update(mutate)


def delete_dataset(dataset_id: str) -> bool:
    removed = {"value": False}

    def mutate(state: Dict[str, Any]) -> None:
        datasets = state.get("datasets") or {}
        if dataset_id in datasets:
            datasets.pop(dataset_id)
            removed["value"] = True
        if state.get("default_dataset_id") == dataset_id:
            state["default_dataset_id"] = next(iter(datasets), None)

    update(mutate)
    shutil.rmtree(dataset_dir(dataset_id), ignore_errors=True)
    return removed["value"]


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def run_dir(run_id: str) -> Path:
    path = RUNS_DIR / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_runs(limit: int = 50) -> List[dict]:
    runs = list((load().get("runs") or {}).values())
    runs.sort(key=lambda item: item.get("created_at") or 0, reverse=True)
    return runs[:limit]


def get_run(run_id: str) -> Optional[dict]:
    return (load().get("runs") or {}).get(run_id)


def put_run(record: dict) -> dict:
    def mutate(state: Dict[str, Any]) -> None:
        state.setdefault("runs", {})[record["id"]] = record

    update(mutate)
    return record


def delete_run(run_id: str) -> bool:
    removed = {"value": False}

    def mutate(state: Dict[str, Any]) -> None:
        runs = state.get("runs") or {}
        if runs.pop(run_id, None) is not None:
            removed["value"] = True

    update(mutate)
    shutil.rmtree(RUNS_DIR / run_id, ignore_errors=True)
    return removed["value"]
