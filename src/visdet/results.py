"""Experiment-log tables and plots used by the notebooks and the report."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluate import EXPERIMENTS
from .plotting import INK_2, SERIES

MAIN_COLS = ["mAP", "AP50", "AP75", "AP_s", "AP_m", "AP_l", "AR", "fps", "gpu_hours", "epochs"]


def load_log(path: str | Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or EXPERIMENTS / "results.csv").set_index("run_id")


def family_table(log: pd.DataFrame, runs: list[str], cols=MAIN_COLS) -> pd.DataFrame:
    """Metrics of ``runs`` (first = baseline) with the mAP change against the baseline."""
    runs = [r for r in runs if r in log.index]
    t = log.loc[runs, [c for c in cols if c in log]].copy()
    t.insert(1, "ΔmAP", t["mAP"] - t["mAP"].iloc[0])
    return t


def best_run(log: pd.DataFrame, runs: list[str]) -> str:
    runs = [r for r in runs if r in log.index]
    return log.loc[runs, "mAP"].idxmax()


def history(run_id: str, root: str | Path | None = None) -> pd.DataFrame:
    """Per-epoch history as (epoch, gpu_hours, mAP) for torchvision and Ultralytics runs alike.

    torchvision runs log our evaluator's mAP on unmasked frames; Ultralytics runs log their own
    validator on the masked prepared frames (comparable between YOLO and RT-DETR, not across frameworks).
    """
    import json

    run = Path(root or EXPERIMENTS / "runs") / run_id
    h = pd.read_csv(run / "history.csv")
    meta = json.loads((run / "metrics.json").read_text())
    gpus = meta.get("n_gpus") or 1
    if "metrics/mAP50-95(B)" in h:
        return pd.DataFrame({"epoch": h["epoch"], "gpu_hours": h["time"] / 3600 * gpus,
                             "mAP": h["metrics/mAP50-95(B)"]})
    h = h.dropna(subset=["mAP"])
    return pd.DataFrame({"epoch": h["epoch"], "gpu_hours": h["hours"] * gpus, "mAP": h["mAP"]})


def plot_histories(run_ids: list[str], x: str = "epoch", ax=None, title: str = "Validation mAP during training"):
    ax = ax or plt.subplots(figsize=(6, 3.2))[1]
    for k, r in enumerate(run_ids):
        h = history(r)
        ax.plot(h[x], h["mAP"], color=SERIES[k % len(SERIES)], marker="o", ms=3, label=r)
    ax.set_xlabel({"epoch": "epoch", "gpu_hours": "GPU hours (T4)"}[x])
    ax.set_ylabel("mAP@[.5:.95]")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return ax


def plot_breakdown(log: pd.DataFrame, runs: list[str], cols=("AP_s", "AP_m", "AP_l"),
                   labels=("small", "medium", "large"), ax=None, title: str = "AP by object size"):
    """Grouped bars: one group per bucket, one bar per run (runs keep their colour slot)."""
    runs = [r for r in runs if r in log.index]
    ax = ax or plt.subplots(figsize=(6.5, 3))[1]
    width = 0.8 / len(runs)
    x = np.arange(len(cols))
    for k, r in enumerate(runs):
        vals = log.loc[r, list(cols)].astype(float).values
        ax.bar(x + (k - (len(runs) - 1) / 2) * width, vals, width * 0.92, color=SERIES[k % len(SERIES)], label=r)
    ax.set_xticks(x, labels)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("AP@[.5:.95]")
    ax.set_title(title)
    ax.legend(fontsize=8, ncols=min(len(runs), 4), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return ax


def styled(t: pd.DataFrame):
    """Compact display: 3 decimals for metrics, 1 for speed / time."""
    fmt = {c: "{:.3f}" for c in t.columns}
    fmt |= {c: "{:.1f}" for c in ("fps", "gpu_hours") if c in t}
    fmt |= {c: "{:+.3f}" for c in ("ΔmAP",) if c in t}
    fmt |= {c: "{:.0f}" for c in ("epochs",) if c in t}
    return t.style.format(fmt, na_rep="–")
