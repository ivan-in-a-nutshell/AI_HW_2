"""Overlapping tiles: geometry shared by tiled training data and tiled (SAHI-style) inference.

Cutting a frame into a 2x2 grid of overlapping tiles and resizing each tile to the network input
enlarges every object by ~1.8x (1920-px frame -> ~1070-px tiles -> 1280 input), which is what the
many < 16 px objects need.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def tile_windows(W: int, H: int, grid=(2, 2), overlap: float = 0.2) -> list[tuple[int, int, int, int]]:
    """``x0, y0, x1, y1`` of a ``grid = (columns, rows)`` of tiles overlapping by ``overlap`` of a tile."""
    gx, gy = grid
    tw, th = W / (gx - (gx - 1) * overlap), H / (gy - (gy - 1) * overlap)
    xs = [round(i * tw * (1 - overlap)) for i in range(gx)]
    ys = [round(j * th * (1 - overlap)) for j in range(gy)]
    return [(x, y, min(W, round(x + tw)), min(H, round(y + th))) for y in ys for x in xs]


def _tile_frame(task: dict) -> list[str]:
    img = cv2.imread(task["src"])
    H, W = img.shape[:2]
    lbl = Path(task["lbl"])
    a = np.loadtxt(lbl, ndmin=2) if lbl.exists() and lbl.stat().st_size else np.zeros((0, 5))
    cls = a[:, 0].astype(int)
    xyxy = np.c_[(a[:, 1] - a[:, 3] / 2) * W, (a[:, 2] - a[:, 4] / 2) * H, (a[:, 1] + a[:, 3] / 2) * W, (a[:, 2] + a[:, 4] / 2) * H]
    out = []
    for k, (x0, y0, x1, y1) in enumerate(tile_windows(W, H, task["grid"], task["overlap"])):
        c = np.c_[np.clip(xyxy[:, 0], x0, x1), np.clip(xyxy[:, 1], y0, y1), np.clip(xyxy[:, 2], x0, x1), np.clip(xyxy[:, 3], y0, y1)]
        area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        keep = (c[:, 2] - c[:, 0]) * (c[:, 3] - c[:, 1]) >= task["min_visible"] * np.maximum(area, 1e-6)
        tw, th = x1 - x0, y1 - y0
        lines = [f"{k_} {((b[0] + b[2]) / 2 - x0) / tw:.6f} {((b[1] + b[3]) / 2 - y0) / th:.6f} "
                 f"{(b[2] - b[0]) / tw:.6f} {(b[3] - b[1]) / th:.6f}" for k_, b in zip(cls[keep], c[keep])]
        name = f"{Path(task['src']).stem}_t{k}"
        dst = Path(task["out"]) / "images" / task["clip"] / f"{name}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), img[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        dl = Path(task["out"]) / "labels" / task["clip"] / f"{name}.txt"
        dl.parent.mkdir(parents=True, exist_ok=True)
        dl.write_text("\n".join(lines) + ("\n" if lines else ""))
        out.append(str(dst))
    return out


def make_tiles(prepared_root: Path, rows: pd.DataFrame, out_dir: Path, grid=(2, 2), overlap: float = 0.2,
               min_visible: float = 0.5, workers: int | None = None) -> list[str]:
    """Cut prepared (masked) frames into tiles with YOLO labels; returns the tile image paths.

    A box is kept in a tile when at least ``min_visible`` of its area falls inside it (then clipped).
    """
    prepared_root, out_dir = Path(prepared_root), Path(out_dir)
    tasks = [{"src": str(prepared_root / p), "clip": c, "out": str(out_dir), "grid": grid, "overlap": overlap,
              "min_visible": min_visible,
              "lbl": str(prepared_root / "labels" / c / (Path(p).stem + ".txt"))}
             for p, c in rows[["path", "clip"]].itertuples(index=False)]
    paths = []
    with ProcessPoolExecutor(workers or os.cpu_count()) as ex:
        for out in tqdm(ex.map(_tile_frame, tasks, chunksize=8), total=len(tasks), desc="tiles"):
            paths += out
    return paths
