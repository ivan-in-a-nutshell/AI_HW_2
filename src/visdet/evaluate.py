"""COCO-style evaluation shared by every detector.

Ignore regions are added to the ground truth as ``iscrowd=1`` boxes for every class, so
pycocotools does not count detections inside them as false positives. Up to 500 detections
per image are scored, as on the leaderboard. Detections are a DataFrame with columns
``image_id, cls (0-based), score, x, y, w, h`` in the same pixel space as the ground truth.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .data import CLASSES, NUM_CLASSES

MAX_DETS = 500
EXPERIMENTS = Path(os.environ.get("VISDET_EXPERIMENTS", "experiments"))  # experiment log folder


def build_gt(images: pd.DataFrame, boxes: pd.DataFrame) -> COCO:
    """COCO ground truth for ``images`` (test-scale), ignore regions replicated per class."""
    boxes = boxes[boxes["image_id"].isin(images["image_id"])]
    obj = boxes[boxes["iscrowd"] == 0]
    crowd = boxes[boxes["iscrowd"] == 1]

    anns = [obj.assign(category_id=obj["cls"] + 1)]
    anns += [crowd.assign(category_id=c + 1) for c in range(NUM_CLASSES)]
    anns = pd.concat(anns, ignore_index=True)

    xywh = anns[["x", "y", "w", "h"]].to_numpy(float)
    dataset = {
        "images": [{"id": int(i), "width": int(w), "height": int(h)}
                   for i, w, h in images[["image_id", "out_w", "out_h"]].itertuples(index=False)],
        "categories": [{"id": c + 1, "name": n} for c, n in enumerate(CLASSES)],
        "annotations": [
            {"id": k + 1, "image_id": int(img), "category_id": int(cat), "bbox": list(b),
             "area": float(b[2] * b[3]), "iscrowd": int(crowd_flag)}
            for k, (img, cat, crowd_flag, b) in enumerate(
                zip(anns["image_id"], anns["category_id"], anns["iscrowd"], xywh))
        ],
    }
    gt = COCO()
    gt.dataset = dataset
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
    return gt


def top_k(dets: pd.DataFrame, k: int = MAX_DETS) -> pd.DataFrame:
    """Keep the ``k`` highest-scoring detections of each image."""
    dets = dets.sort_values(["image_id", "score"], ascending=[True, False])
    return dets[dets.groupby("image_id").cumcount() < k]


def evaluate(gt: COCO, dets: pd.DataFrame, per_class: bool = True) -> dict:
    """mAP@[.5:.95], AP50, AP75, AP_small/medium/large and AR at 500 detections per image."""
    dets = top_k(dets[dets["image_id"].isin(gt.getImgIds())])
    stats = {"n_images": len(gt.getImgIds()), "n_dets": len(dets)}
    if dets.empty:
        return stats | {k: 0.0 for k in ("mAP", "AP50", "AP75", "AP_s", "AP_m", "AP_l", "AR")}

    results = np.column_stack([
        dets["image_id"], dets["x"], dets["y"], dets["w"], dets["h"], dets["score"], dets["cls"] + 1,
    ]).astype(np.float64)
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(results)
        E = COCOeval(gt, dt, "bbox")
        E.params.maxDets = [1, 10, 100, MAX_DETS]
        E.evaluate()
        E.accumulate()

    prec = E.eval["precision"][..., -1]  # [iou, recall, class, area]
    rec = E.eval["recall"][..., -1]      # [iou, class, area]

    def mean(x):
        x = x[x > -1]
        return float(x.mean()) if x.size else float("nan")

    stats |= {
        "mAP": mean(prec[..., 0]), "AP50": mean(prec[0, ..., 0]), "AP75": mean(prec[5, ..., 0]),
        "AP_s": mean(prec[..., 1]), "AP_m": mean(prec[..., 2]), "AP_l": mean(prec[..., 3]),
        "AR": mean(rec[..., 0]),
    }
    if per_class:
        for k, name in enumerate(CLASSES):
            stats[f"AP_{name}"] = mean(prec[:, :, k, 0])
    return stats


def density_breakdown(images: pd.DataFrame, boxes: pd.DataFrame, dets: pd.DataFrame,
                      bins=(0, 30, 80, 10_000)) -> dict:
    """mAP on frames grouped by the number of objects (scene density)."""
    n = boxes[boxes["iscrowd"] == 0].groupby("image_id").size().reindex(images["image_id"], fill_value=0)
    out = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = images[(n.values >= lo) & (n.values < hi)]
        if len(sel):
            name = f"AP_dens<{hi}" if lo == 0 else (f"AP_dens>={lo}" if hi == bins[-1] else f"AP_dens{lo}-{hi}")
            out[name] = evaluate(build_gt(sel, boxes), dets, per_class=False)["mAP"]
    return out


def format_stats(stats: dict, keys=("mAP", "AP50", "AP75", "AP_s", "AP_m", "AP_l", "AR")) -> str:
    return "  ".join(f"{k} {stats[k]:.4f}" for k in keys if k in stats)


def log_result(run_id: str, stats: dict, results_csv: str | Path | None = None,
               history: pd.DataFrame | None = None, config: dict | None = None, **extra) -> pd.DataFrame:
    """Append (or replace) one run in the experiment log; keep its metrics + per-epoch history.

    Everything lands in ``experiments/`` so it can be committed and compared across notebooks.
    """
    results_csv = Path(results_csv or EXPERIMENTS / "results.csv")
    row = {"run_id": run_id, **extra, **{k: (round(v, 4) if isinstance(v, float) else v)
                                          for k, v in stats.items()}}
    run_dir = results_csv.parent / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(row, indent=2, default=str))
    if history is not None:
        history.to_csv(run_dir / "history.csv", index=False)
    if config is not None:
        import yaml
        (run_dir / "config.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(config, default=str)), sort_keys=False))

    log = pd.read_csv(results_csv) if results_csv.exists() else pd.DataFrame()
    log = pd.concat([log[log.get("run_id", pd.Series(dtype=str)) != run_id], pd.DataFrame([row])],
                    ignore_index=True)
    log.to_csv(results_csv, index=False)
    return log


def measure_fps(predict, frames: list[np.ndarray], warmup: int = 5) -> float:
    """Frames per second of a predictor at batch size 1, end to end (pre-processing + model + NMS)."""
    import torch

    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    warmup = min(warmup, len(frames) - 1)
    for f in frames[:warmup]:
        predict([f])
    sync()
    t0 = time.perf_counter()
    for f in frames[warmup:]:
        predict([f])
    sync()
    return (len(frames) - warmup) / (time.perf_counter() - t0)
