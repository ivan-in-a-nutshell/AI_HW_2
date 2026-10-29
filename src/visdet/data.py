"""Dataset tables, class mapping, clip-level folds and frame subsampling.

All annotations are held in two flat DataFrames:
  images: one row per frame  (image_id, split, clip, frame, file_name, width, height, scale, out_w, out_h)
  boxes:  one row per box    (image_id, cls, x, y, w, h, iscrowd, track_id, occlusion, truncation, orig_cls)
``cls`` is the 0-based submission id; ignore regions have ``cls = -1`` and ``iscrowd = 1``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]
NUM_CLASSES = len(CLASSES)
MAX_SIDE = 1920  # no test frame is wider than this
COMPETITION = "multiple-object-video-detection-hw-2-ucu-ai-2026"

KAGGLE_NOTEBOOKS = Path("/kaggle/input/notebooks/ivanzarytskyics")

def find_raw_root() -> Path:
    """Locate the competition data: $VISDET_RAW, Kaggle input mount, or the local copy."""
    candidates = [
        os.environ.get("VISDET_RAW"),
        f"/kaggle/input/competitions/{COMPETITION}",
        Path(__file__).resolve().parents[2] / COMPETITION,
    ]
    for c in candidates:
        if c and (Path(c) / "train").is_dir():
            return Path(c)
    raise FileNotFoundError(f"competition data not found in {candidates}")


def find_prepared(name: str = "prepared") -> Path:
    """Locate a prepared dataset: $VISDET_PREPARED (default name only), a Kaggle input, or ./<name>."""
    env = os.environ.get("VISDET_PREPARED") if name == "prepared" else None
    candidates = [env, *sorted(KAGGLE_NOTEBOOKS.glob(f"*/{name}")),
                  Path(__file__).resolve().parents[2] / name]
    for c in candidates:
        if c and (Path(c) / "images.csv").is_file():
            return Path(c)
    raise FileNotFoundError(f"prepared dataset '{name}' not found in {candidates}")


def test_scale(w: int, h: int, max_side: int = MAX_SIDE) -> float:
    """Halve the frame until it fits ``max_side``.

    Test frames look like train resolutions halved when they exceed 1920 px
    (2688x1512 -> 1344x756, 2720x1530 -> 1360x764, 3840x2160 -> 1920x1080),
    so this reproduces the test resolution groups.
    """
    s = 1.0
    while max(w, h) * s > max_side:
        s /= 2
    return s


# ----------------------------------------------------------------------------- loading

def load_split(raw_root: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read ``<split>/annotations_coco.json`` into (images, boxes) tables in original pixels."""
    with open(Path(raw_root) / split / "annotations_coco.json") as f:
        coco = json.load(f)

    images = pd.DataFrame(coco["images"]).rename(
        columns={"id": "coco_id", "video": "clip", "frame_id": "frame"})
    images["split"] = split
    images["file_name"] = split + "/" + images["file_name"]
    images = images[["coco_id", "split", "clip", "frame", "file_name", "width", "height"]]

    anns = pd.DataFrame(coco["annotations"])
    xywh = np.array(anns["bbox"].tolist(), dtype=np.float32)
    boxes = pd.DataFrame({
        "coco_id": anns["image_id"].values,
        "cls": anns["category_id"].values - 1,
        "x": xywh[:, 0], "y": xywh[:, 1], "w": xywh[:, 2], "h": xywh[:, 3],
        "iscrowd": anns["iscrowd"].values.astype(np.int8),
        "track_id": anns.get("track_id", pd.Series(-1, index=anns.index)).fillna(-1).astype(int).values,
        "occlusion": anns.get("occlusion", pd.Series(-1, index=anns.index)).fillna(-1).astype(int).values,
        "truncation": anns.get("truncation", pd.Series(-1, index=anns.index)).fillna(-1).astype(int).values,
        "orig_cls": anns["orig_category_id"].values,
    })
    # Ignore regions are stored once per category (8 copies); keep a single class-agnostic copy.
    crowd = boxes["iscrowd"] == 1
    boxes.loc[crowd, ["cls", "track_id", "occlusion", "truncation"]] = -1
    boxes = pd.concat([boxes[~crowd], boxes[crowd].drop_duplicates(["coco_id", "x", "y", "w", "h"])])
    return images, boxes


def load_tables(raw_root: Path, splits=("train", "val")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge the given splits, assign global ``image_id`` and the test-scale output size."""
    all_images, all_boxes = [], []
    offset = 0
    for split in splits:
        images, boxes = load_split(raw_root, split)
        images["image_id"] = images["coco_id"] + offset
        boxes["image_id"] = boxes["coco_id"] + offset
        offset = int(images["image_id"].max()) + 1
        all_images.append(images.drop(columns="coco_id"))
        all_boxes.append(boxes.drop(columns="coco_id"))
    images = pd.concat(all_images, ignore_index=True)
    boxes = pd.concat(all_boxes, ignore_index=True)

    images["scale"] = [test_scale(w, h) for w, h in zip(images["width"], images["height"])]
    images["out_w"] = (images["width"] * images["scale"]).round().astype(int)
    images["out_h"] = (images["height"] * images["scale"]).round().astype(int)
    images = images.sort_values(["clip", "frame"]).reset_index(drop=True)
    cols = ["image_id", "cls", "x", "y", "w", "h", "iscrowd", "track_id", "occlusion", "truncation", "orig_cls"]
    return images, boxes.sort_values(["image_id", "iscrowd"]).reset_index(drop=True)[cols]


def to_test_scale(images: pd.DataFrame, boxes: pd.DataFrame) -> pd.DataFrame:
    """Rescale boxes to the test-scale frame size and clip them to the image."""
    meta = images.set_index("image_id").loc[boxes["image_id"], ["scale", "out_w", "out_h"]].to_numpy()
    s, W, H = meta[:, 0], meta[:, 1], meta[:, 2]
    x1 = np.clip(boxes["x"].to_numpy() * s, 0, W)
    y1 = np.clip(boxes["y"].to_numpy() * s, 0, H)
    x2 = np.clip((boxes["x"].to_numpy() + boxes["w"].to_numpy()) * s, 0, W)
    y2 = np.clip((boxes["y"].to_numpy() + boxes["h"].to_numpy()) * s, 0, H)
    out = boxes.copy()
    out["x"], out["y"], out["w"], out["h"] = x1, y1, x2 - x1, y2 - y1
    return out[(out["w"] >= 1) & (out["h"] >= 1)].reset_index(drop=True)


def read_prepared(prepared_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the tables written by :mod:`visdet.prepare` (test-scale coordinates)."""
    root = Path(prepared_root)
    images = pd.read_csv(root / "images.csv")
    boxes = pd.read_csv(root / "boxes.csv.gz")
    folds = pd.read_csv(root / "folds.csv")
    return images, boxes, folds


# ----------------------------------------------------------------------------- splits

def clip_stats(images: pd.DataFrame, boxes: pd.DataFrame) -> pd.DataFrame:
    """Per-clip summary used for stratification and EDA."""
    obj = boxes[boxes["iscrowd"] == 0]
    per_frame = obj.groupby("image_id").size().reindex(images["image_id"], fill_value=0)
    df = images.assign(n_obj=per_frame.values)
    stats = df.groupby("clip").agg(
        split=("split", "first"), n_frames=("frame", "size"),
        width=("out_w", "first"), height=("out_h", "first"),
        obj_per_frame=("n_obj", "mean"))
    cls_share = (obj.merge(images[["image_id", "clip"]], on="image_id")
                 .groupby("clip")["cls"].value_counts(normalize=True).unstack(fill_value=0))
    stats["person_share"] = cls_share.get(0, 0).reindex(stats.index, fill_value=0)
    return stats


def make_folds(images: pd.DataFrame, boxes: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Assign whole clips to folds, stratified by scene density x person share.

    Frames of one video are near-duplicates, so splitting by frame would leak.
    """
    from sklearn.model_selection import StratifiedKFold

    stats = clip_stats(images, boxes)
    density_bin = pd.qcut(stats["obj_per_frame"], 3, labels=False)
    person_heavy = (stats["person_share"] > stats["person_share"].median()).astype(int)
    strata = density_bin * 2 + person_heavy

    folds = pd.Series(-1, index=stats.index, name="fold")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, val_idx) in enumerate(skf.split(stats.index, strata)):
        folds.iloc[val_idx] = k
    return folds.reset_index()


def subsample(images: pd.DataFrame, step: int, offset: int = 0) -> pd.DataFrame:
    """Keep every ``step``-th frame of each clip (by frame order)."""
    rank = images.groupby("clip")["frame"].rank(method="first").astype(int) - 1
    return images[(rank % step) == offset % step]


def fold_split(images: pd.DataFrame, folds: pd.DataFrame, fold: int,
               train_step: int = 10, val_step: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training/validation frames of one fold. ``fold=-1`` trains on every clip (no validation)."""
    val_clips = set(folds.loc[folds["fold"] == fold, "clip"])
    is_val = images["clip"].isin(val_clips)
    return subsample(images[~is_val], train_step), subsample(images[is_val], val_step)
