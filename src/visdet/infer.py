"""Model-agnostic inference: predictor wrappers and dataset-level prediction.

A predictor maps a list of BGR frames to a list of ``[N, 6]`` arrays ``x1, y1, x2, y2, score, cls``
in the pixel space of the frame it received.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_frame(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    """Read a BGR frame and, if ``size=(w, h)`` differs, downscale it like the test frames."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    if size is not None and (img.shape[1], img.shape[0]) != tuple(size):
        img = cv2.resize(img, tuple(size), interpolation=cv2.INTER_AREA)
    return img


class UltralyticsPredictor:
    """YOLO / RT-DETR checkpoint wrapped as a predictor."""

    def __init__(self, weights, imgsz=1280, conf=0.001, iou=0.6, max_det=500,
                 augment=False, half=True, device=None, rtdetr: bool | None = None):
        from ultralytics import RTDETR, YOLO
        from ultralytics.nn.tasks import RTDETRDetectionModel

        if rtdetr is None:
            rtdetr = isinstance(YOLO(weights).model, RTDETRDetectionModel)
        self.model = RTDETR(weights) if rtdetr else YOLO(weights)
        self.kw = dict(imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, augment=augment,
                       quantize=16 if half else None, device=device, verbose=False)

    def __call__(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        out = []
        for r in self.model.predict(frames, **self.kw):
            b = r.boxes
            out.append(np.column_stack([b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()]))
        return out


class TiledPredictor:
    """Run ``base`` on overlapping tiles (+ optionally the full frame) and merge with class-wise NMS.

    Tile detections touching an inner tile border are dropped: with enough overlap, the object is
    seen whole by the neighbouring tile.
    """

    def __init__(self, base, grid=(2, 2), overlap: float = 0.2, full_frame: bool = True,
                 iou_thr: float = 0.6, max_det: int = 500, edge_margin: int = 2):
        self.base, self.grid, self.overlap, self.full = base, grid, overlap, full_frame
        self.iou_thr, self.max_det, self.margin = iou_thr, max_det, edge_margin

    def __call__(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        import torch
        from torchvision.ops import batched_nms

        from .tiles import tile_windows

        out = []
        for f in frames:
            H, W = f.shape[:2]
            wins = tile_windows(W, H, self.grid, self.overlap)
            preds = []
            for (x0, y0, x1, y1), p in zip(wins, self.base([f[y0:y1, x0:x1] for x0, y0, x1, y1 in wins])):
                p = p.copy()
                p[:, [0, 2]] += x0
                p[:, [1, 3]] += y0
                inner = ((x0 > 0) & (p[:, 0] <= x0 + self.margin)) | ((x1 < W) & (p[:, 2] >= x1 - self.margin)) | \
                        ((y0 > 0) & (p[:, 1] <= y0 + self.margin)) | ((y1 < H) & (p[:, 3] >= y1 - self.margin))
                preds.append(p[~inner])
            if self.full:
                preds += self.base([f])
            p = np.concatenate(preds) if preds else np.zeros((0, 6))
            t = torch.from_numpy(p).float()
            keep = batched_nms(t[:, :4], t[:, 4], t[:, 5].long(), self.iou_thr)[: self.max_det].numpy()
            out.append(p[keep])
        return out


class EnsemblePredictor:
    """Fuse several predictors (models, input sizes, horizontal flip) with weighted boxes fusion."""

    def __init__(self, members: list, weights: list[float] | None = None, hflip: bool = False,
                 iou_thr: float = 0.6, skip_thr: float = 0.01, max_det: int = 500, conf_type: str = "avg"):
        self.members, self.hflip = members, hflip
        n = len(members) * (2 if hflip else 1)
        self.weights = (weights or [1.0] * len(members)) * (2 if hflip else 1)
        assert len(self.weights) == n
        self.iou_thr, self.skip_thr, self.max_det, self.conf_type = iou_thr, skip_thr, max_det, conf_type

    def __call__(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        from ensemble_boxes import weighted_boxes_fusion

        runs = [m(frames) for m in self.members]
        if self.hflip:
            flipped = [f[:, ::-1].copy() for f in frames]
            for m in self.members:
                res = []
                for f, p in zip(frames, m(flipped)):
                    p = p.copy()
                    p[:, [0, 2]] = f.shape[1] - p[:, [2, 0]]
                    res.append(p)
                runs.append(res)
        out = []
        for i, f in enumerate(frames):
            H, W = f.shape[:2]
            scale = np.array([W, H, W, H], dtype=float)
            b = [np.clip(r[i][:, :4] / scale, 0, 1) for r in runs]
            s = [r[i][:, 4] for r in runs]
            c = [r[i][:, 5] for r in runs]
            fb, fs, fc = weighted_boxes_fusion(b, s, c, weights=self.weights, iou_thr=self.iou_thr,
                                               skip_box_thr=self.skip_thr, conf_type=self.conf_type)
            order = np.argsort(-fs)[: self.max_det]
            out.append(np.column_stack([fb[order] * scale, fs[order], fc[order]]))
        return out


def to_frame(pred: np.ndarray, w: int, h: int) -> pd.DataFrame:
    """``[N, 6]`` xyxy predictions -> clipped xywh DataFrame."""
    x1 = np.clip(pred[:, 0], 0, w)
    y1 = np.clip(pred[:, 1], 0, h)
    x2 = np.clip(pred[:, 2], 0, w)
    y2 = np.clip(pred[:, 3], 0, h)
    df = pd.DataFrame({"cls": pred[:, 5].astype(int), "score": pred[:, 4],
                       "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1})
    return df[(df["w"] > 0) & (df["h"] > 0)]


def predict_dataset(predictor, keys, paths, sizes=None, batch_size: int = 8, workers: int = 4,
                    desc: str = "predict") -> pd.DataFrame:
    """Run ``predictor`` over frames; ``sizes[i] = (w, h)`` resizes frame i before inference.

    Returns detections with an ``image_id`` column holding the matching key; ``desc=None`` hides progress.
    """
    sizes = sizes if sizes is not None else [None] * len(paths)
    items = list(zip(keys, paths, sizes))
    chunks = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    out = []
    with ThreadPoolExecutor(workers) as pool:
        loaded = pool.map(lambda ch: [load_frame(p, s) for _, p, s in ch], chunks)
        for chunk, frames in tqdm(zip(chunks, loaded), total=len(chunks), desc=desc, disable=desc is None):
            for (key, _, _), frame, pred in zip(chunk, frames, predictor(frames)):
                out.append(to_frame(pred, frame.shape[1], frame.shape[0]).assign(image_id=key))
    cols = ["image_id", "cls", "score", "x", "y", "w", "h"]
    return pd.concat(out, ignore_index=True)[cols] if out else pd.DataFrame(columns=cols)


def predict_eval_frames(predictor, raw_root: Path, images: pd.DataFrame, **kw) -> pd.DataFrame:
    """Predict on unmasked raw frames downscaled to test scale (validation protocol)."""
    return predict_dataset(
        predictor, images["image_id"].tolist(),
        [Path(raw_root) / f for f in images["file_name"]],
        list(zip(images["out_w"], images["out_h"])), **kw)
