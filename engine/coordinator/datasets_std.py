"""One-click import of standard object-detection datasets.

Uploading a zip is fine for your own data. For a paper you want a dataset a
reader already knows, with published numbers to compare against, and you want it
without hunting for a download link.

Ultralytics ships resolved configs for the usual sets and will fetch them on
demand, so this module wraps that and registers the result in GradMesh's own
dataset registry. Nothing is copied: the registry points at where Ultralytics
put the files, and the benchmark harness takes subsets by manifest.

**On ImageNet, since it comes up.** ImageNet is a classification dataset. Its
1.28M-image ILSVRC subset has one label per image and no bounding boxes, so
there is nothing for a detection model to learn from and nothing to convert. The
ILSVRC detection subset does have boxes, but it is roughly 150 GB, ships VOC-style
XML that needs converting, and sits behind a login and a signed agreement, so it
cannot be fetched by a script. COCO is the dataset that plays the role people
usually want ImageNet for here: standard, citable, boxes included, and published
YOLO baselines to compare against.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Curated because the point is a good default, not a catalogue. Sizes are the
# download, and image counts are the training split.
CATALOGUE: List[Dict[str, Any]] = [
    {
        "key": "coco8",
        "name": "COCO8",
        "images": 8,
        "classes": 80,
        "download_mb": 1,
        "blurb": "Eight images. Proves the pipeline runs. Useless as a result.",
        "good_for": "smoke test",
    },
    {
        "key": "coco128",
        "name": "COCO128",
        "images": 128,
        "classes": 80,
        "download_mb": 7,
        "blurb": "A 128-image slice of COCO. Fast enough to iterate on the harness.",
        "good_for": "smoke test",
    },
    {
        "key": "african-wildlife",
        "name": "African Wildlife",
        "images": 1052,
        "classes": 4,
        "download_mb": 100,
        "blurb": "Four animal classes, clean labels, small download. A sensible first real sweep.",
        "good_for": "100 and 1000 image sweeps",
    },
    {
        "key": "GlobalWheat2020",
        "name": "Global Wheat 2020",
        "images": 3422,
        "classes": 1,
        "download_mb": 700,
        "blurb": "Single class, dense small objects. Low label noise, quick epochs.",
        "good_for": "up to 3000 images",
    },
    {
        "key": "VisDrone",
        "name": "VisDrone2019-DET",
        "images": 6471,
        "classes": 10,
        "download_mb": 2300,
        "blurb": "Drone imagery, ten classes, many small objects. The best fit for a 100 to 6000 ladder.",
        "good_for": "the full scaling ladder",
    },
    {
        "key": "coco",
        "name": "COCO 2017",
        "images": 118287,
        "classes": 80,
        "download_mb": 20000,
        "blurb": "The citable standard, with published YOLO baselines. A 20 GB download.",
        "good_for": "a headline result, if you have the disk and the time",
    },
    {
        "key": "DOTAv1",
        "name": "DOTA v1",
        "images": 1411,
        "classes": 15,
        "download_mb": 2000,
        "blurb": "Oriented bounding boxes. Only useful with a -obb base model.",
        "good_for": "oriented box work",
    },
]

CATALOGUE_BY_KEY = {entry["key"]: entry for entry in CATALOGUE}


def _label_dir_for(image_dir: Path) -> Optional[Path]:
    """YOLO's convention: the labels path is the images path with one segment swapped."""
    parts = list(image_dir.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            candidate = Path(*parts[:index], "labels", *parts[index + 1 :])
            return candidate if candidate.is_dir() else None
    sibling = image_dir.parent / "labels"
    return sibling if sibling.is_dir() else None


def _first_existing(value: Any, root: Path) -> Optional[Path]:
    """A split in a dataset YAML may be a string or a list of paths."""
    candidates = value if isinstance(value, (list, tuple)) else [value]
    for item in candidates:
        if not item:
            continue
        path = Path(str(item))
        resolved = path if path.is_absolute() else (root / path)
        if resolved.is_dir():
            return resolved.resolve()
        # Some configs point at a .txt listing rather than a directory.
        if resolved.is_file() and resolved.suffix == ".txt":
            parent = resolved.parent / "images"
            if parent.is_dir():
                return parent.resolve()
    return None


def download(key: str, progress=None) -> Dict[str, Any]:
    """Fetch a catalogue dataset and return the resolved split layout.

    Blocking and potentially very slow, so callers run it on a worker thread and
    report progress through the event stream.
    """
    entry = CATALOGUE_BY_KEY.get(key)
    if entry is None:
        raise ValueError("Unknown dataset %r" % key)

    if progress:
        progress("Resolving %s" % entry["name"])

    from ultralytics.data.utils import check_det_dataset

    # Downloads if absent, verifies if present, and returns absolute paths.
    resolved = check_det_dataset("%s.yaml" % key, autodownload=True)

    root = Path(resolved.get("path") or ".").resolve()
    train_images = _first_existing(resolved.get("train"), root)
    val_images = _first_existing(resolved.get("val"), root) or train_images
    if train_images is None:
        raise RuntimeError("Could not locate the training images for %s" % entry["name"])

    names = resolved.get("names") or {}
    if isinstance(names, dict):
        class_names = [str(names[k]) for k in sorted(names, key=lambda v: int(v))]
    else:
        class_names = [str(n) for n in names]

    return {
        "key": key,
        "name": entry["name"],
        "root": str(root),
        "train_images": str(train_images),
        "train_labels": str(_label_dir_for(train_images) or ""),
        "val_images": str(val_images) if val_images else "",
        "val_labels": str(_label_dir_for(val_images) or "") if val_images else "",
        "class_names": class_names or ["object"],
        "yaml": str(resolved.get("yaml_file") or ""),
        "imported_at": time.time(),
    }


def catalogue() -> List[Dict[str, Any]]:
    """The list the dashboard renders, with a note on what is already local."""
    try:
        from ultralytics.utils import SETTINGS

        base = Path(SETTINGS.get("datasets_dir") or ".")
    except Exception:
        base = Path(".")

    entries = []
    for entry in CATALOGUE:
        local = base / entry["key"]
        entries.append({**entry, "already_downloaded": local.is_dir()})
    return entries
