"""Build the training dataset: test-scale frames, masked ignore regions, YOLO labels, tables.

Usage:
    python -m visdet.prepare --out prepared [--raw <data root>] [--step 1] [--workers 8]

Output layout (``--out``):
    images/<clip>/<frame>.jpg   resized to test scale, ignore regions filled with gray
    labels/<clip>/<frame>.txt   YOLO labels of the boxes kept for training
    images.csv, boxes.csv.gz    full tables in test-scale pixels (``ign_frac`` per box)
    folds.csv                   clip -> fold
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .data import find_raw_root, load_tables, make_folds, subsample, to_test_scale

FILL = 114            # same gray Ultralytics uses for letterbox padding
MAX_IGNORE_FRAC = 0.5  # drop a training box if more than half of it lies in ignore regions


def ignore_mask(shape: tuple[int, int], regions: np.ndarray) -> np.ndarray:
    """Rasterise xywh ignore regions into a boolean mask."""
    mask = np.zeros(shape, dtype=bool)
    for x, y, w, h in regions:
        x1, y1 = int(np.floor(x)), int(np.floor(y))
        x2, y2 = int(np.ceil(x + w)), int(np.ceil(y + h))
        mask[y1:y2, x1:x2] = True
    return mask


def ignore_fraction(mask: np.ndarray, objs: np.ndarray) -> np.ndarray:
    """Share of each xywh box covered by the ignore mask."""
    frac = np.zeros(len(objs), dtype=np.float32)
    for i, (x, y, w, h) in enumerate(objs):
        x1, y1 = int(np.floor(x)), int(np.floor(y))
        x2, y2 = max(int(np.ceil(x + w)), x1 + 1), max(int(np.ceil(y + h)), y1 + 1)
        frac[i] = mask[y1:y2, x1:x2].mean()
    return frac


def process_frame(task: dict) -> np.ndarray:
    """Resize one frame, mask its ignore regions, write image + YOLO label. Returns ign_frac."""
    W, H = task["out_w"], task["out_h"]
    objs, cls, regions = task["objs"], task["cls"], task["regions"]

    mask = ignore_mask((H, W), regions)
    frac = ignore_fraction(mask, objs) if len(objs) else np.zeros(0, np.float32)
    keep = frac <= MAX_IGNORE_FRAC

    img = cv2.imread(task["src"])
    if img.shape[1] != W or img.shape[0] != H:
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    if task["mask"] and mask.any():
        # never paint over the boxes we keep as targets
        for x, y, w, h in objs[keep]:
            mask[int(y):int(np.ceil(y + h)), int(x):int(np.ceil(x + w))] = False
        img[mask] = FILL

    Path(task["dst_img"]).parent.mkdir(parents=True, exist_ok=True)
    Path(task["dst_lbl"]).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(task["dst_img"], img, [cv2.IMWRITE_JPEG_QUALITY, task["quality"]])

    k = objs[keep]
    lines = [f"{c} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}"
             for c, (x, y, w, h) in zip(cls[keep], k)]
    Path(task["dst_lbl"]).write_text("\n".join(lines) + ("\n" if lines else ""))
    return frac


def build_tasks(images: pd.DataFrame, boxes: pd.DataFrame, raw: Path, out: Path,
                mask: bool, quality: int) -> list[dict]:
    groups = dict(tuple(boxes.groupby("image_id")))
    empty = boxes.iloc[:0]
    tasks = []
    for r in images.itertuples():
        b = groups.get(r.image_id, empty)
        obj = b[b["iscrowd"] == 0]
        crowd = b[b["iscrowd"] == 1]
        rel = Path("images") / r.clip / Path(r.file_name).name
        tasks.append({
            "image_id": r.image_id,
            "box_index": obj.index.to_numpy(),
            "src": str(raw / r.file_name),
            "dst_img": str(out / rel),
            "dst_lbl": str(out / "labels" / r.clip / (Path(r.file_name).stem + ".txt")),
            "out_w": int(r.out_w), "out_h": int(r.out_h),
            "objs": obj[["x", "y", "w", "h"]].to_numpy(np.float32),
            "cls": obj["cls"].to_numpy(),
            "regions": crowd[["x", "y", "w", "h"]].to_numpy(np.float32),
            "mask": mask, "quality": quality,
        })
    return tasks


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, default=None, help="competition data root")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--step", type=int, default=1, help="write only every N-th frame of each clip")
    p.add_argument("--clips", type=int, default=0, help="debug: only the first N clips")
    p.add_argument("--no-mask", action="store_true", help="keep ignore regions visible (labels still dropped)")
    p.add_argument("--quality", type=int, default=90, help="JPEG quality")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    args = p.parse_args(argv)

    raw = args.raw or find_raw_root()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"reading annotations from {raw}")
    images, boxes = load_tables(raw)
    boxes = to_test_scale(images, boxes)
    folds = make_folds(images, boxes, args.folds, args.seed)  # always on all clips
    if args.clips:
        keep = sorted(images["clip"].unique())[: args.clips]
        images = images[images["clip"].isin(keep)]
        boxes = boxes[boxes["image_id"].isin(images["image_id"])]

    todo = subsample(images, args.step)
    tasks = build_tasks(todo, boxes, raw, out, mask=not args.no_mask, quality=args.quality)
    boxes["ign_frac"] = np.nan
    with ProcessPoolExecutor(args.workers) as ex:
        results = ex.map(process_frame, tasks, chunksize=8)
        for task, frac in tqdm(zip(tasks, results), total=len(tasks), desc="frames"):
            boxes.loc[task["box_index"], "ign_frac"] = frac

    images = images.merge(folds, on="clip")
    images["prepared"] = images["image_id"].isin(todo["image_id"])
    images["path"] = "images/" + images["clip"] + "/" + images["file_name"].str.rsplit("/", n=1).str[-1]
    # keep = box is a training label (prepared frame, not mostly inside an ignore region)
    boxes["keep"] = (boxes["iscrowd"] == 0) & (boxes["ign_frac"] <= MAX_IGNORE_FRAC)
    boxes[["x", "y", "w", "h"]] = boxes[["x", "y", "w", "h"]].round(2)
    boxes["ign_frac"] = boxes["ign_frac"].round(3)
    images.to_csv(out / "images.csv", index=False)
    boxes.to_csv(out / "boxes.csv.gz", index=False)
    folds.to_csv(out / "folds.csv", index=False)

    dropped = (boxes["iscrowd"] == 0) & ~boxes["keep"] & boxes["ign_frac"].notna()
    print(f"{len(todo)} frames written, {boxes['keep'].sum()} training boxes, {dropped.sum()} dropped inside ignore regions")


if __name__ == "__main__":
    main()
