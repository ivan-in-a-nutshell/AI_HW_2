"""Ultralytics (YOLO / RT-DETR) helpers: fold datasets, training with patches, evaluation.

Experiment configs (``configs/yolo*.yaml``, ``configs/rtdetr*.yaml``) hold Ultralytics train
arguments plus three of our own keys:
  model    checkpoint (``yolo11s.pt``) or model YAML relative to ``configs/``
  weights  pretrained checkpoint to transfer into a YAML-defined model
  patch    changes Ultralytics does not expose:
           tal_topk, matcher_cost ({class, bbox, giou}), backbone_lr_mult + backbone_layers,
           degrade_p + degrade_range (random down/up-scaling augmentation, YOLO only)

Patches travel to DDP workers through the ``VISDET_PATCH`` environment variable and are applied
when the worker imports this module to rebuild our trainer class.
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path

import pandas as pd
import yaml
from ultralytics import RTDETR, YOLO
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.yolo.detect import DetectionTrainer

from .config import load_config
from .data import CLASSES, fold_split
from .evaluate import build_gt, density_breakdown, evaluate, format_stats, log_result, measure_fps
from .infer import UltralyticsPredictor, load_frame, predict_eval_frames

OWN_KEYS = ("name", "base", "framework", "model", "weights", "patch", "spec")


# ----------------------------------------------------------------------------- data

def write_yolo_data(prepared_root: Path, images: pd.DataFrame, folds: pd.DataFrame, fold: int,
                    out_dir: Path, train_step: int = 10, val_step: int = 10, val_limit: int | None = None,
                    allow_missing: bool = False) -> Path:
    """Write image lists + ``data.yaml`` for one fold and return the yaml path.

    Validation frames are the masked prepared frames; they only drive Ultralytics' own
    per-epoch metrics. Reported numbers come from :mod:`visdet.evaluate` on unmasked frames.
    ``fold=-1`` trains on every clip and monitors on fold 0 (not held out). ``val_limit`` and
    ``allow_missing`` are for local dry runs on a partially prepared dataset.
    """
    prepared_root, out_dir = Path(prepared_root).resolve(), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train, val = fold_split(images, folds, fold, train_step, val_step)
    if fold < 0:
        val = fold_split(images, folds, 0, train_step, val_step)[1]
    val = val.head(val_limit)
    missing = (~train["prepared"]).sum() + (~val["prepared"]).sum()
    if missing and not allow_missing:
        raise ValueError(f"{missing} frames were not prepared; re-run prepare with a smaller --step")
    train, val = train[train["prepared"]], val[val["prepared"]]

    lists = {}
    for name, df in (("train", train), ("val", val)):
        lists[name] = out_dir / f"{name}.txt"
        lists[name].write_text("\n".join(str(prepared_root / p) for p in df["path"]) + "\n")
    data = {"path": str(out_dir), "train": str(lists["train"]), "val": str(lists["val"]),
            "nc": len(CLASSES), "names": CLASSES}
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"fold {fold}: {len(train)} train / {len(val)} val frames -> {yaml_path}")
    return yaml_path


def write_tile_data(prepared_root: Path, images: pd.DataFrame, folds: pd.DataFrame, fold: int, out_dir: Path,
                    train_step: int = 10, val_step: int = 10, val_limit: int | None = None,
                    grid=(2, 2), overlap: float = 0.2) -> Path:
    """Like :func:`write_yolo_data`, but training and monitoring frames are cut into overlapping tiles."""
    from .tiles import make_tiles

    out_dir = Path(out_dir)
    train, val = fold_split(images, folds, fold, train_step, val_step)
    if fold < 0:
        val = fold_split(images, folds, 0, train_step, val_step)[1]
    val = val.head(val_limit)
    lists = {}
    for name, df in (("train", train), ("val", val)):
        paths = make_tiles(prepared_root, df, out_dir / "tiles", grid, overlap)
        lists[name] = out_dir / f"{name}.txt"
        lists[name].write_text("\n".join(str(Path(q).resolve()) for q in paths) + "\n")
    data = {"path": str(out_dir.resolve()), "train": str(lists["train"].resolve()), "val": str(lists["val"].resolve()),
            "nc": len(CLASSES), "names": CLASSES}
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"fold {fold}: {len(train)} train / {len(val)} val frames cut into {grid[0]}x{grid[1]} tiles -> {yaml_path}")
    return yaml_path


# ----------------------------------------------------------------------------- patches

def _patch() -> dict:
    return json.loads(os.environ.get("VISDET_PATCH", "{}"))


class Degrade:
    """Downscale by a random factor and scale back: mimics the test clips stored at 1/4 resolution."""

    def __init__(self, p: float, low: float = 0.5, high: float = 0.8):
        self.p, self.low, self.high = p, low, high

    def __call__(self, labels: dict) -> dict:
        if random.random() < self.p:
            import cv2

            img = labels["img"]
            h, w = img.shape[:2]
            f = random.uniform(self.low, self.high)
            small = cv2.resize(img, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)
            labels["img"] = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        return labels


def _apply_patches():
    """Wrap Ultralytics constructors so the settings in VISDET_PATCH apply (also inside DDP workers)."""
    from ultralytics.data import dataset as ul_dataset
    from ultralytics.models.utils import loss as detr_loss
    from ultralytics.utils import loss as yolo_loss

    if not getattr(ul_dataset.YOLODataset, "_visdet", False):
        bt_orig = ul_dataset.YOLODataset.build_transforms

        def build_transforms(self, hyp):
            t = bt_orig(self, hyp)  # rebuilt by close_mosaic too, so the hook stays in place
            if self.augment and _patch().get("degrade_p", 0) > 0:
                t.insert(len(t.transforms) - 1, Degrade(_patch()["degrade_p"], *_patch().get("degrade_range", (0.5, 0.8))))
            return t

        ul_dataset.YOLODataset.build_transforms, ul_dataset.YOLODataset._visdet = build_transforms, True

    if not getattr(yolo_loss.v8DetectionLoss, "_visdet", False):
        v8_orig = yolo_loss.v8DetectionLoss.__init__

        def v8_init(self, model, *args, **kwargs):
            v8_orig(self, model, *args, **kwargs)
            if "tal_topk" in _patch():
                self.assigner.topk = _patch()["tal_topk"]

        yolo_loss.v8DetectionLoss.__init__, yolo_loss.v8DetectionLoss._visdet = v8_init, True

    if not getattr(detr_loss.DETRLoss, "_visdet", False):
        detr_orig = detr_loss.DETRLoss.__init__

        def detr_init(self, *args, **kwargs):
            detr_orig(self, *args, **kwargs)
            if "matcher_cost" in _patch():
                self.matcher.cost_gain = dict(_patch()["matcher_cost"])

        detr_loss.DETRLoss.__init__, detr_loss.DETRLoss._visdet = detr_init, True


_apply_patches()


def _scale_backbone_lr(optimizer, model):
    """Split every param group into backbone / rest and scale the backbone learning rate."""
    p = _patch()
    if "backbone_lr_mult" not in p:
        return optimizer
    layers = model.model[: p.get("backbone_layers", 10)]
    backbone = {id(q) for q in layers.parameters()}
    groups = []
    for g in optimizer.param_groups:
        bb = [q for q in g["params"] if id(q) in backbone]
        rest = [q for q in g["params"] if id(q) not in backbone]
        for params, mult in ((bb, p["backbone_lr_mult"]), (rest, 1.0)):
            if params:
                groups.append({**{k: v for k, v in g.items() if k != "params"},
                               "params": params, "lr": g["lr"] * mult})
    optimizer.param_groups.clear()
    for g in groups:
        optimizer.add_param_group(g)
    return optimizer


class VisdetDetectionTrainer(DetectionTrainer):
    def build_optimizer(self, model, *args, **kwargs):
        return _scale_backbone_lr(super().build_optimizer(model, *args, **kwargs), model)


class VisdetRTDETRTrainer(RTDETRTrainer):
    def build_optimizer(self, model, *args, **kwargs):
        return _scale_backbone_lr(super().build_optimizer(model, *args, **kwargs), model)


# ----------------------------------------------------------------------------- train / evaluate

def _print_epoch(trainer):
    """One compact line per epoch instead of Ultralytics' full log (rank 0 only)."""
    from ultralytics.utils import RANK

    if RANK in (-1, 0):
        m = trainer.metrics or {}
        hours = (time.time() - trainer.train_time_start) / 3600
        e, n = trainer.epoch + 1, trainer.epochs
        print(f"{f'epoch {e}/{n}' if e <= n else 'best.pt (final validation)'}  {hours:.2f} h  "
              f"val mAP50-95 {m.get('metrics/mAP50-95(B)', float('nan')):.4f}  mAP50 {m.get('metrics/mAP50(B)', float('nan')):.4f}",
              flush=True)


def is_rtdetr(cfg: dict) -> bool:
    return "rtdetr" in Path(cfg["model"]).stem.lower()


def train_ultralytics(cfg: str | Path | dict, data_yaml: Path, project: Path, config_dir: Path = Path("configs"),
                      **overrides) -> tuple[Path, dict]:
    """Train one experiment (config file or dict); returns (best checkpoint, config) and records GPU hours.

    A dict lets a notebook derive a variant from a tuned config, e.g. ``load_config(...) | {"name": ..., ...}``;
    model YAMLs are then resolved relative to ``config_dir``.
    """
    if not isinstance(cfg, dict):
        config_dir, cfg = Path(cfg).parent, load_config(cfg)
    cfg = dict(cfg)
    model_src = cfg["model"]
    if model_src.endswith(".yaml"):
        model_src = str(Path(config_dir) / model_src)
    os.environ["VISDET_PATCH"] = json.dumps(cfg.get("patch", {}))

    model = (RTDETR if is_rtdetr(cfg) else YOLO)(model_src)
    if cfg.get("weights"):
        model.load(cfg["weights"])
    args = {k: v for k, v in cfg.items() if k not in OWN_KEYS} | overrides
    trainer = VisdetRTDETRTrainer if is_rtdetr(cfg) else VisdetDetectionTrainer
    # quiet Ultralytics (model summary, arguments, progress bars); DDP workers read YOLO_VERBOSE
    from ultralytics.utils import LOGGER
    os.environ["YOLO_VERBOSE"] = "False"
    level = LOGGER.level
    LOGGER.setLevel(logging.WARNING)
    model.add_callback("on_fit_epoch_end", _print_epoch)
    print(f"{cfg['name']}: training ({', '.join(f'{k}={v}' for k, v in args.items() if k in ('imgsz', 'time', 'epochs', 'batch'))})")
    t0 = time.time()
    try:
        model.train(data=str(data_yaml), project=str(Path(project).resolve()), name=cfg["name"], exist_ok=True,
                    trainer=trainer, **args)
    finally:
        LOGGER.setLevel(level)
    run_dir = Path(project).resolve() / cfg["name"]
    device = str(args.get("device", "0"))
    n_gpu = len([d for d in device.split(",") if d.strip()]) if device not in ("cpu", "mps") else 1
    cfg["gpu_hours"], cfg["n_gpus"] = round((time.time() - t0) / 3600 * n_gpu, 2), n_gpu
    (run_dir / "visdet_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return run_dir / "weights" / "best.pt", cfg


def evaluate_ultralytics(weights: Path, cfg: dict, raw_root: Path, val_rows: pd.DataFrame, boxes: pd.DataFrame,
                         results_csv=None, fold: int = 0, log: bool = True, run_id: str | None = None,
                         wrap=None, config_extra: dict | None = None, **predict_kw) -> dict:
    """Shared-protocol metrics (unmasked test-scale frames, ignore-aware) + FPS of a checkpoint.

    ``wrap(predictor)`` adds an inference strategy (tiles, TTA); ``run_id`` names that variant in the log.
    """
    run_id = run_id or cfg["name"]
    kw = {"imgsz": cfg.get("imgsz", 1280), "max_det": cfg.get("max_det", 500), "rtdetr": is_rtdetr(cfg)} | predict_kw
    predictor = UltralyticsPredictor(weights, **kw)
    if wrap is not None:
        predictor = wrap(predictor)
    dets = predict_eval_frames(predictor, raw_root, val_rows, desc=f"eval {run_id}")
    stats = evaluate(build_gt(val_rows, boxes), dets) | density_breakdown(val_rows, boxes, dets)
    run_dir = Path(weights).parents[1]
    dets[dets["score"] >= 0.01].round(2).to_csv(run_dir / f"val_dets_{run_id}.csv.gz", index=False)
    frames = [load_frame(Path(raw_root) / f, (w, h)) for f, w, h in
              val_rows[["file_name", "out_w", "out_h"]].head(55).itertuples(index=False)]
    stats["fps"] = measure_fps(predictor, frames)
    print(run_id, format_stats(stats), f"fps {stats['fps']:.1f}")
    del predictor  # release the GPU memory before the next training run starts its DDP workers
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if log:
        hist = pd.read_csv(run_dir / "results.csv") if (run_dir / "results.csv").exists() else None
        if hist is not None:
            hist.columns = hist.columns.str.strip()
        log_result(run_id, stats, results_csv, history=hist, config=cfg | {"inference": predict_kw} | (config_extra or {}),
                   family="rtdetr" if is_rtdetr(cfg) else "yolo", fold=fold,
                   epochs=None if hist is None else len(hist), gpu_hours=cfg.get("gpu_hours"), n_gpus=cfg.get("n_gpus"))
    return stats
