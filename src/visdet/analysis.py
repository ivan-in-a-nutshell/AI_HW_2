"""Failure analysis shared by all detectors.

Every confident detection is labelled with one error type (class-aware greedy matching, like COCO):
  tp          matches an unmatched object of the same class (IoU >= 0.5)
  duplicate   same class, IoU >= 0.5, but that object was already matched by a higher score
  wrong_cls   IoU >= 0.5 with an object of another class
  loc         same class, 0.1 <= IoU < 0.5 (poor localisation)
  ignored     mostly inside an ignore region (not penalised by the metric)
  background  none of the above
Every object is marked found / missed, so recall can be broken down by size and occlusion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CLASSES

ERROR_TYPES = ["tp", "duplicate", "wrong_cls", "loc", "ignored", "background"]


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of xywh boxes, plus intersection over the area of ``a`` (IoA)."""
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2) - np.maximum(a[:, None, 0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2) - np.maximum(a[:, None, 1], b[:, 1]), 0, None)
    inter = iw * ih
    area_a, area_b = a[:, 2] * a[:, 3], b[:, 2] * b[:, 3]
    return inter / (area_a[:, None] + area_b - inter + 1e-9), inter / (area_a[:, None] + 1e-9)


def classify_errors(boxes: pd.DataFrame, dets: pd.DataFrame, score_thr: float = 0.3,
                    iou_thr: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label detections (score >= ``score_thr``) by error type and objects by found / missed.

    ``boxes``: ground truth of the evaluated frames incl. ignore regions (``iscrowd``);
    ``dets``: image_id, cls, score, x, y, w, h.
    Returns (dets with ``type`` and ``gt_cls``, objects with ``found``).
    """
    dets = dets[dets["score"] >= score_thr].sort_values("score", ascending=False)
    det_groups = dict(tuple(dets.groupby("image_id")))
    out_d, out_g = [], []
    for img, g in boxes.groupby("image_id"):
        obj, ign = g[g["iscrowd"] == 0], g[g["iscrowd"] == 1]
        d = det_groups.get(img, dets.iloc[:0])
        D, G = d[["x", "y", "w", "h"]].to_numpy(float), obj[["x", "y", "w", "h"]].to_numpy(float)
        dc, gc = d["cls"].to_numpy(), obj["cls"].to_numpy()
        types, gt_cls = np.full(len(d), "background", dtype=object), np.full(len(d), -1)
        found = np.zeros(len(obj), bool)
        if len(d) and len(obj):
            iou, _ = _iou(D, G)
            for i in range(len(d)):
                same = gc == dc[i]
                cand = np.where(same & ~found, iou[i], -1)
                j = cand.argmax()
                if cand[j] >= iou_thr:
                    types[i], found[j], gt_cls[i] = "tp", True, gc[j]
                    continue
                if np.where(same, iou[i], -1).max() >= iou_thr:
                    types[i] = "duplicate"
                elif np.where(~same, iou[i], -1).max() >= iou_thr:
                    k = np.where(~same, iou[i], -1).argmax()
                    types[i], gt_cls[i] = "wrong_cls", gc[k]
                elif np.where(same, iou[i], -1).max() >= 0.1:
                    types[i] = "loc"
        if len(d) and len(ign):
            ioa = _iou(D, ign[["x", "y", "w", "h"]].to_numpy(float))[1].max(1)
            types[(types == "background") & (ioa >= 0.5)] = "ignored"
        out_d.append(d.assign(type=types, gt_cls=gt_cls))
        out_g.append(obj.assign(found=found))
    det_t = pd.concat(out_d, ignore_index=True) if out_d else dets.assign(type=[], gt_cls=[])
    gt_t = pd.concat(out_g, ignore_index=True)
    gt_t["size_bucket"] = pd.cut(np.sqrt(gt_t["w"] * gt_t["h"]), [0, 16, 32, 96, np.inf],
                                 labels=["tiny <16", "small 16-32", "medium 32-96", "large >96"])
    return det_t, gt_t


def error_summary(det_t: pd.DataFrame, gt_t: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Share of confident detections per error type, and recall by object size x occlusion."""
    shares = det_t["type"].value_counts(normalize=True).reindex(ERROR_TYPES, fill_value=0)
    occl = gt_t["occlusion"].map({0: "none", 1: "partial", 2: "heavy"})
    recall = gt_t.assign(occl=occl).pivot_table(index="size_bucket", columns="occl", values="found",
                                                aggfunc="mean", observed=False)
    recall["all"] = gt_t.groupby("size_bucket", observed=False)["found"].mean()
    return shares, recall[[c for c in ("none", "partial", "heavy", "all") if c in recall]]


def confusion(det_t: pd.DataFrame) -> pd.DataFrame:
    """Rows: true class of the overlapped object, columns: predicted class (tp + wrong_cls only)."""
    m = det_t[det_t["type"].isin(["tp", "wrong_cls"])]
    tab = pd.crosstab(pd.Categorical(np.array(CLASSES)[m["gt_cls"]], CLASSES),
                      pd.Categorical(np.array(CLASSES)[m["cls"]], CLASSES), dropna=False)
    tab.index.name, tab.columns.name = "true", "predicted"
    return tab


def frame_errors(det_t: pd.DataFrame, gt_t: pd.DataFrame) -> pd.DataFrame:
    """Per-frame counts of misses and each detection error type (to pick failure examples)."""
    d = det_t.pivot_table(index="image_id", columns="type", values="score", aggfunc="size", fill_value=0)
    g = gt_t.groupby("image_id").agg(objects=("found", "size"), missed=("found", lambda f: (~f).sum()))
    return g.join(d, how="left").fillna(0).astype(int)


# BGR colours for drawing: found = green, missed = yellow, background FP = red, wrong class = orange
COLORS = {"found": (60, 160, 30), "missed": (0, 200, 255), "background": (72, 73, 227),
          "wrong_cls": (52, 104, 235), "loc": (214, 81, 213), "duplicate": (200, 200, 200)}


def draw_errors(img: np.ndarray, gt_t: pd.DataFrame, det_t: pd.DataFrame) -> np.ndarray:
    """Overlay objects (found / missed) and erroneous detections on a BGR frame."""
    import cv2

    img = img.copy()
    for r in gt_t.itertuples():
        c = COLORS["found" if r.found else "missed"]
        cv2.rectangle(img, (int(r.x), int(r.y)), (int(r.x + r.w), int(r.y + r.h)), c, 1 if r.found else 2)
    for r in det_t[det_t["type"].isin(["background", "wrong_cls", "loc", "duplicate"])].itertuples():
        cv2.rectangle(img, (int(r.x), int(r.y)), (int(r.x + r.w), int(r.y + r.h)), COLORS[r.type], 2)
        if r.type == "wrong_cls":
            cv2.putText(img, f"{CLASSES[r.cls]}<{CLASSES[r.gt_cls]}", (int(r.x), int(r.y) - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[r.type], 1, cv2.LINE_AA)
    return img
