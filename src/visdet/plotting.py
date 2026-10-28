"""Matplotlib style and drawing helpers shared by notebooks and report figures."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# categorical slots, fixed order (validated palette); a series keeps its slot across charts
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def setup_style():
    if BLUES.name not in mpl.colormaps:
        mpl.colormaps.register(BLUES)
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=SERIES), "lines.linewidth": 2, "legend.frameon": False,
        "image.cmap": BLUES.name,
    })


def save(fig, name: str, folder: str | Path | None = None, photo: bool = False):
    """Save a report figure as ``<folder>/<name>.png`` (``.jpg`` for photo-like figures).

    The folder defaults to ``$VISDET_FIGURES`` or ``images/`` (the report's figure folder).
    """
    folder = Path(folder or os.environ.get("VISDET_FIGURES", "images"))
    folder.mkdir(parents=True, exist_ok=True)
    if photo:
        fig.savefig(Path(folder) / f"{name}.jpg", dpi=110, pil_kwargs={"quality": 82})
    else:
        fig.savefig(Path(folder) / f"{name}.png")


def draw_boxes(img: np.ndarray, xywh: np.ndarray, color=(42, 120, 214), thickness=1) -> np.ndarray:
    """Draw xywh pixel boxes on a BGR image (in place) and return it."""
    for x, y, w, h in np.asarray(xywh).reshape(-1, 4):
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)
    return img


def show_bgr(ax, img: np.ndarray, title: str = ""):
    ax.imshow(img[:, :, ::-1])
    ax.set_title(title, fontsize=8, fontweight="normal")
    ax.axis("off")
