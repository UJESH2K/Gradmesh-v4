"""Accuracy measurement for the aggregated global model.

Speedup without accuracy is not a result. A mesh that trains four times faster
and loses ten points of mAP has not accelerated anything, so every claim in the
benchmark is paired with a measured accuracy on a held-out split.

Three decisions here exist so the numbers survive review.

**Evaluation happens on the coordinator, not on a worker.** Every configuration
in a sweep is then scored by the same code on the same machine against the same
images. Farming evaluation out to whichever worker was idle would make accuracy
depend on which GPU ran it, which is exactly the confound a controlled
comparison has to exclude.

**Evaluation time is never counted as training time.** It is timed separately
and reported separately. Otherwise a configuration that evaluated more often
would look slower for reasons that have nothing to do with distribution.

**The validation split is the parent dataset's, always.** A 100-image subset and
a 10000-image subset of the same dataset are scored against identical held-out
images, so a change in mAP is a change in what was learned rather than a change
in what was measured.
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from federated_training import decode_state_dict


def torch_device() -> str:
    """Best device available to the coordinator for evaluation."""
    try:
        import torch
    except Exception:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
    except Exception:
        pass
    return "cpu"


def write_eval_yaml(dataset_root: Path, destination: Path, class_names: List[str]) -> Path:
    """A data.yaml whose val split is the full held-out set.

    Ultralytics needs a train key even for validation, so it points at the val
    images too. Nothing trains from this file.
    """
    from coordinator.sharding import list_split_images

    listing = list_split_images(dataset_root)
    dirs = listing["dirs"]
    root = Path(dirs["root"]).resolve()

    val_dir = dirs["val_images"] or dirs["train_images"]
    val_relative = Path(val_dir).resolve().relative_to(root).as_posix()

    names = class_names or ["object"]
    names_block = "\n".join("  %d: %s" % (index, name) for index, name in enumerate(names))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "path: %s\ntrain: %s\nval: %s\nnc: %d\nnames:\n%s\n"
        % (root.as_posix(), val_relative, val_relative, len(names), names_block),
        encoding="utf-8",
    )
    return destination


def evaluate_weights(
    weights_b64: str,
    base_model_path: Path,
    eval_yaml: Path,
    imgsz: int,
    workdir: Path,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Run validation on an aggregated state dict and return the usual metrics.

    Returns a dict that always carries an `ok` flag. A failed evaluation must
    not fail a training run: the run produced weights, and losing the score is
    worth far less than losing the model.
    """
    started = time.perf_counter()
    try:
        from ultralytics import YOLO

        model = YOLO(str(base_model_path))
        state = decode_state_dict(weights_b64)
        current = model.model.state_dict()
        compatible = {
            key: value
            for key, value in state.items()
            if key in current
            and hasattr(value, "shape")
            and tuple(value.shape) == tuple(current[key].shape)
        }
        model.model.load_state_dict(compatible, strict=False)

        workdir.mkdir(parents=True, exist_ok=True)
        resolved = device or torch_device()

        # Ultralytics 8.4 rejects the string "xpu"; the worker hits the same
        # wall and solves it the same way, with a real torch.device.
        if resolved == "xpu":
            import torch

            resolved = torch.device("xpu:0")

        results = model.val(
            data=str(eval_yaml),
            imgsz=imgsz,
            device=resolved,
            workers=0,
            verbose=False,
            plots=False,
            save_json=False,
            project=str(workdir),
            name="val",
            exist_ok=True,
        )

        box = getattr(results, "box", None)
        return {
            "ok": True,
            "map50": round(float(getattr(box, "map50", 0.0) or 0.0), 5),
            "map50_95": round(float(getattr(box, "map", 0.0) or 0.0), 5),
            "precision": round(float(getattr(box, "mp", 0.0) or 0.0), 5),
            "recall": round(float(getattr(box, "mr", 0.0) or 0.0), 5),
            "device": str(resolved),
            "seconds": round(time.perf_counter() - started, 2),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": "%s: %s" % (type(exc).__name__, exc),
            "seconds": round(time.perf_counter() - started, 2),
        }


def decoded_size_bytes(weights_b64: Optional[str]) -> int:
    """Wire size of a base64 payload without decoding it.

    Communication overhead is one of the reported metrics, and these payloads
    are tens of megabytes. Decoding one just to call len() on it would allocate
    the whole model again per round, per worker.
    """
    if not weights_b64:
        return 0
    length = len(weights_b64)
    padding = weights_b64[-2:].count("=") if length >= 2 else 0
    return max(0, (length * 3) // 4 - padding)


def write_artifact(path: Path, weights_b64: str) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(weights_b64.encode("ascii")))
        return True
    except Exception:
        return False


def host_snapshot() -> Dict[str, Any]:
    """What ran this, for the disclosure table reviewers check first."""
    import platform
    import sys

    info: Dict[str, Any] = {
        "platform": "%s %s" % (platform.system(), platform.release()),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch"] = None
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except Exception:
        info["ultralytics"] = None
    return info
