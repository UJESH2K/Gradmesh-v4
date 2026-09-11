"""Proportional YOLO dataset sharding.

federated_training.build_yolo_shards splits a dataset into equal slices. The v4
scheduler produces a *sample vector* instead, one entry per admitted worker, so
this module materialises shards of arbitrary unequal sizes.

The v3 training pipeline is untouched: shards produced here have exactly the
layout worker.train_batch already expects, namely images/train, labels/train,
optional images/val and labels/val, and a data.yaml at the shard root.
"""

from __future__ import annotations

import random
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from federated_training import (
    IMAGE_SUFFIXES,
    discover_yolo_split_dirs,
    write_data_yaml,
    _copy_split_pairs,
)


def list_split_images(
    dataset_root: Path,
    manifest: Optional[Path] = None,
    splits: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the split directories plus the sorted image lists.

    A manifest restricts the training list to named files. That is how dataset
    subsets work: a 1000-image subset of a 10000-image parent is a list of
    filenames, not a second copy of the images on disk. Copying would make a
    scaling sweep across 100, 1000 and 10000 images cost several times the
    dataset in disk and minutes in setup, for no scientific gain.

    `splits` supplies the four directories directly, skipping discovery. Uploaded
    zips are found by directory name, but a standard dataset fetched through
    Ultralytics puts its images somewhere like VisDrone2019-DET-train/images,
    which no naming convention would guess. Its own config already names the
    paths, so when the caller has them it passes them in rather than searching.
    """
    if splits and splits.get("train_images"):
        split_dirs = {
            "root": Path(splits.get("root") or dataset_root),
            "train_images": Path(splits["train_images"]),
            "train_labels": Path(splits["train_labels"]) if splits.get("train_labels") else None,
            "val_images": Path(splits["val_images"]) if splits.get("val_images") else None,
            "val_labels": Path(splits["val_labels"]) if splits.get("val_labels") else None,
        }
        if split_dirs["train_labels"] is None:
            raise ValueError("The training labels directory could not be resolved")
    else:
        split_dirs = discover_yolo_split_dirs(dataset_root)
    train_images = sorted(
        path
        for path in split_dirs["train_images"].rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )

    if manifest is not None and manifest.is_file():
        wanted = {
            line.strip()
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        base = split_dirs["train_images"]
        train_images = [
            path for path in train_images if path.relative_to(base).as_posix() in wanted
        ]

    val_images: List[Path] = []
    if split_dirs["val_images"] is not None:
        val_images = sorted(
            path
            for path in split_dirs["val_images"].rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    return {"dirs": split_dirs, "train": train_images, "val": val_images}


def write_subset_manifest(
    dataset_root: Path,
    destination: Path,
    sample_count: int,
    seed: int = 0,
    splits: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Choose a reproducible subset of the training split and record it.

    The validation split is deliberately untouched. Every subset in a scaling
    sweep is evaluated against exactly the same held-out images, otherwise a
    change in measured accuracy could be a change in the test set rather than a
    change in the training set, and the sweep would answer nothing.
    """
    listing = list_split_images(dataset_root, splits=splits)
    base = listing["dirs"]["train_images"]
    relative = [path.relative_to(base).as_posix() for path in listing["train"]]

    if sample_count >= len(relative):
        chosen = relative
    else:
        chosen = random.Random(seed).sample(relative, sample_count)

    chosen.sort()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(chosen), encoding="utf-8")
    return {
        "manifest_path": str(destination),
        "train_count": len(chosen),
        "val_count": len(listing["val"]),
        "parent_train_count": len(relative),
        "seed": seed,
    }


def count_training_samples(
    dataset_root: Path,
    manifest: Optional[Path] = None,
    splits: Optional[Dict[str, Any]] = None,
) -> int:
    return len(list_split_images(dataset_root, manifest=manifest, splits=splits)["train"])


def partition(images: Sequence[Path], sizes: Sequence[int], seed: int) -> List[List[Path]]:
    """Split images into contiguous groups of the requested sizes.

    The pool is shuffled with a per-round seed first. Sharding by sorted order
    would hand every worker a class-correlated slice, because most YOLO exports
    are ordered by capture session, and that biases the local gradients before
    aggregation ever sees them. A deterministic shuffle keeps each shard an
    unbiased sample while staying reproducible for the paper.
    """
    pool = list(images)
    random.Random(seed).shuffle(pool)

    groups: List[List[Path]] = []
    cursor = 0
    for size in sizes:
        groups.append(pool[cursor : cursor + size])
        cursor += size

    # Rounding can leave a tail. Give it to the largest shard, which by
    # construction belongs to the fastest worker.
    if cursor < len(pool) and groups:
        largest = max(range(len(groups)), key=lambda index: len(groups[index]))
        groups[largest].extend(pool[cursor:])
    return groups


def build_proportional_shards(
    dataset_root: Path,
    shard_output_dir: Path,
    sizes: Sequence[int],
    class_names: List[str],
    seed: int = 0,
    replicate_val: bool = True,
    manifest: Optional[Path] = None,
    splits: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Materialise one zipped shard per entry in sizes.

    Every shard carries the full validation split when replicate_val is set, so
    each worker reports mAP against the same held-out data and the per-round
    numbers are comparable across heterogeneous shards.
    """
    if not sizes:
        raise ValueError("At least one shard size is required")

    listing = list_split_images(dataset_root, manifest=manifest, splits=splits)
    split_dirs = listing["dirs"]
    train_images = listing["train"]
    val_images = listing["val"]

    if not train_images:
        raise ValueError("No training images were found in the dataset")

    groups = partition(train_images, sizes, seed)
    shard_output_dir.mkdir(parents=True, exist_ok=True)
    shards: List[Dict[str, Any]] = []

    for shard_index, shard_images in enumerate(groups):
        shard_root = shard_output_dir / ("shard_%d" % shard_index)
        if shard_root.exists():
            shutil.rmtree(shard_root)
        shard_root.mkdir(parents=True, exist_ok=True)

        train_count = _copy_split_pairs(
            shard_images,
            split_dirs["train_images"],
            split_dirs["train_labels"],
            shard_root,
            "train",
        )

        val_count = 0
        if replicate_val and split_dirs["val_images"] is not None and split_dirs["val_labels"] is not None:
            val_count = _copy_split_pairs(
                val_images,
                split_dirs["val_images"],
                split_dirs["val_labels"],
                shard_root,
                "val",
            )

        val_relative = "images/val" if val_count > 0 else "images/train"
        data_yaml = write_data_yaml(shard_root, class_names, val_relative=val_relative)

        shard_zip = shard_output_dir / ("shard_%d.zip" % shard_index)
        if shard_zip.exists():
            shard_zip.unlink()
        with zipfile.ZipFile(shard_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file_path in shard_root.rglob("*"):
                if file_path.is_file():
                    archive.write(file_path, file_path.relative_to(shard_root))

        shards.append(
            {
                "shard_index": shard_index,
                "root": str(shard_root),
                "zip_path": str(shard_zip),
                "data_yaml_path": str(data_yaml),
                "train_count": train_count,
                "val_count": val_count,
                "bytes": shard_zip.stat().st_size,
            }
        )

    return shards


def infer_class_names(dataset_root: Path) -> List[str]:
    """Read class names from a data.yaml shipped inside the uploaded dataset."""
    import yaml

    for candidate in sorted(dataset_root.rglob("*.yaml")) + sorted(dataset_root.rglob("*.yml")):
        try:
            config = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        names = config.get("names")
        if isinstance(names, dict):
            return [str(names[key]) for key in sorted(names, key=lambda k: int(k))]
        if isinstance(names, list) and names:
            return [str(name) for name in names]
    return []
