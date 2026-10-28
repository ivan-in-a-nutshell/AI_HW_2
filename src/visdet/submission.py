"""Kaggle submission writing, parsing and validation.

``PredictionString`` = space-separated ``class score x y width height`` groups, 0-based class ids,
top-left corner in pixels of the original test image, at most 500 groups per image.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import NUM_CLASSES
from .evaluate import MAX_DETS, top_k
from .infer import predict_dataset


def predict_test(predictor, raw_root: Path, limit: int | None = None, **kw) -> pd.DataFrame:
    """Detections on the original test frames, keyed by submission ``image_id`` (``limit``: debug)."""
    test = pd.read_csv(Path(raw_root) / "test_images.csv").head(limit)
    return predict_dataset(predictor, test["image_id"].tolist(),
                           [Path(raw_root) / f for f in test["file_name"]], desc="test", **kw)


def make_submission(dets: pd.DataFrame, test: pd.DataFrame, min_score: float = 0.0) -> pd.DataFrame:
    """One row per test image; empty string when there are no detections."""
    dets = top_k(dets[dets["score"] >= min_score], MAX_DETS)
    groups = {
        img: " ".join(f"{c} {s:.4f} {x:.1f} {y:.1f} {w:.1f} {h:.1f}"
                      for c, s, x, y, w, h in g[["cls", "score", "x", "y", "w", "h"]].itertuples(index=False))
        for img, g in dets.groupby("image_id")
    }
    return pd.DataFrame({"image_id": test["image_id"],
                         "PredictionString": [groups.get(i, "") for i in test["image_id"]]})


def parse_submission(sub: pd.DataFrame) -> pd.DataFrame:
    """Inverse of :func:`make_submission`."""
    rows = []
    for img, s in zip(sub["image_id"], sub["PredictionString"].fillna("")):
        v = np.array(s.split(), dtype=float).reshape(-1, 6)
        rows.append(pd.DataFrame(v, columns=["cls", "score", "x", "y", "w", "h"]).assign(image_id=img))
    df = pd.concat(rows, ignore_index=True)
    df["cls"] = df["cls"].astype(int)
    return df[["image_id", "cls", "score", "x", "y", "w", "h"]]


def validate_submission(sub: pd.DataFrame, test: pd.DataFrame) -> None:
    """Raise if the submission would be rejected or silently mis-scored."""
    assert list(sub.columns) == ["image_id", "PredictionString"], sub.columns
    assert len(sub) == len(test) and set(sub["image_id"]) == set(test["image_id"]), "image ids differ"
    dets = parse_submission(sub)
    if dets.empty:
        return
    counts = dets.groupby("image_id").size()
    assert counts.max() <= MAX_DETS, f"{counts.max()} detections in one image"
    assert dets["cls"].between(0, NUM_CLASSES - 1).all(), "class id out of range"
    assert dets["score"].between(0, 1).all(), "score out of [0, 1]"
    size = test.set_index("image_id").loc[dets["image_id"], ["width", "height"]].to_numpy()
    tol = 0.5  # rounding to one decimal
    assert (dets["x"] >= -tol).all() and (dets["y"] >= -tol).all(), "negative coordinates"
    assert (dets["x"] + dets["w"] <= size[:, 0] + tol).all(), "box exceeds image width"
    assert (dets["y"] + dets["h"] <= size[:, 1] + tol).all(), "box exceeds image height"
