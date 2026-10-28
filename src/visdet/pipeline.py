"""Final-pipeline experiments as declarative specs (notebook 07).

A :class:`Spec` is the best detector of notebooks 03-06 plus *one* change: training data (frame step,
ignore-region masking, tiles), augmentation/model settings (config overrides), or the inference
strategy (TTA, tiles, ensembles). :class:`Pipeline` trains and evaluates specs on any fold under the
benchmark protocol and writes the test submission of a spec.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from .data import fold_split
from .evaluate import build_gt, density_breakdown, evaluate, format_stats, log_result
from .infer import EnsemblePredictor, TiledPredictor, UltralyticsPredictor, predict_eval_frames
from .ul_utils import evaluate_ultralytics, is_rtdetr, train_ultralytics, write_tile_data, write_yolo_data


@dataclass
class Spec:
    name: str
    changes: dict = field(default_factory=dict)   # config overrides (model, time, patch, ...)
    masked: bool = True                            # ignore regions filled gray in training frames
    train_step: int = 10                           # every n-th frame of the training clips
    tiles: bool = False                            # train on 2x2 tiles (inference then uses tiles too)
    infer: str | None = None                       # None | "tta" | "tta_ms" | "tiles"
    weights_from: str | None = None                # reuse another run's checkpoint (inference-only change)
    ensemble: list[str] = field(default_factory=list)  # WBF over these runs' checkpoints

    def but(self, name: str, **kw) -> "Spec":
        """Copy with a new name and some fields changed (``changes`` are merged, not replaced)."""
        d = asdict(self) | {"name": name}
        for k, v in kw.items():
            d[k] = d[k] | v if k == "changes" else v
        return Spec(**d)


def find_weights(run: str, runs_dir: Path) -> Path:
    """Checkpoint of ``run`` in this session's runs or in an attached notebook output (Kaggle)."""
    for p in [Path(runs_dir) / run / "weights" / "best.pt", *Path("/kaggle/input").glob(f"*/runs/{run}/weights/best.pt"),
              *Path("/kaggle/input").glob(f"*/*/runs/{run}/weights/best.pt")]:
        if p.exists():
            return p
    raise FileNotFoundError(f"no checkpoint for {run}; train it or attach the notebook output that has it")


def run_config(run: str, runs_dir: Path) -> dict:
    """Effective config a run was trained with (saved next to its weights)."""
    return yaml.safe_load((find_weights(run, runs_dir).parents[1] / "visdet_config.yaml").read_text())


class Pipeline:
    def __init__(self, raw, prepared, runs, images, boxes, folds, base: str, train_kw: dict,
                 prepared_nomask=None, smoke_kw: dict | None = None):
        self.raw, self.prep, self.prep_nomask, self.runs = Path(raw), Path(prepared), prepared_nomask, Path(runs)
        self.images, self.boxes, self.folds = images, boxes, folds
        self.base_cfg = run_config(base, runs) | {"name": base}
        self.train_kw, self.smoke_kw = train_kw, smoke_kw or {}
        self._data = {}

    # ------------------------------------------------------------------ helpers
    def run_id(self, spec: Spec, fold: int) -> str:
        return {0: spec.name, -1: f"{spec.name}_all"}.get(fold, f"{spec.name}_f{fold}")

    def val_rows(self, fold: int) -> pd.DataFrame:
        rows = fold_split(self.images, self.folds, max(fold, 0), 10, 10)[1]
        return rows.head(8) if self.smoke_kw else rows

    def data_yaml(self, spec: Spec, fold: int) -> Path:
        key = (spec.masked, spec.train_step, spec.tiles, fold)
        if key not in self._data:
            root = self.prep if spec.masked else Path(self.prep_nomask)
            images = self.images if spec.masked else pd.read_csv(root / "images.csv")
            out = self.runs / f"data_f{fold}_s{spec.train_step}{'' if spec.masked else '_nomask'}{'_tiles' if spec.tiles else ''}"
            kw = {"val_limit": 16, "allow_missing": True} if self.smoke_kw else {}
            if spec.tiles:
                self._data[key] = write_tile_data(root, images, self.folds, fold, out, train_step=spec.train_step,
                                                  val_limit=kw.get("val_limit"))
            else:
                self._data[key] = write_yolo_data(root, images, self.folds, fold, out, train_step=spec.train_step, **kw)
        return self._data[key]

    def ensemble_predictor(self, spec: Spec) -> EnsemblePredictor:
        """WBF over the member runs, each predicted the way it was trained (e.g. on tiles)."""
        members = []
        for r in spec.ensemble:
            w, c = find_weights(r, self.runs), run_config(r, self.runs)
            members.append(self.predictor(Spec(**c["spec"]) if "spec" in c else Spec(r), w, c))
        return EnsemblePredictor(members)

    def predictor(self, spec: Spec, weights: Path, cfg: dict):
        kw = {"imgsz": self.smoke_kw.get("imgsz", cfg.get("imgsz", 1280)), "max_det": 500, "rtdetr": is_rtdetr(cfg)}
        p = UltralyticsPredictor(weights, **kw)
        if spec.tiles or spec.infer == "tiles":
            return TiledPredictor(p)
        if spec.infer == "tta":
            return EnsemblePredictor([p], hflip=True)
        if spec.infer == "tta_ms":
            big = UltralyticsPredictor(weights, **(kw | {"imgsz": int(kw["imgsz"] * 1.25) // 32 * 32}))
            return EnsemblePredictor([p, big], hflip=True)
        return p

    # ------------------------------------------------------------------ train / evaluate
    def train(self, spec: Spec, fold: int = 0) -> tuple[Path, dict]:
        """Train the spec on ``fold`` (skipped when the checkpoint exists or no training is needed)."""
        if spec.weights_from:
            return find_weights(spec.weights_from, self.runs), run_config(spec.weights_from, self.runs)
        run_id = self.run_id(spec, fold)
        try:
            return find_weights(run_id, self.runs), run_config(run_id, self.runs)
        except FileNotFoundError:
            pass
        cfg = self.base_cfg | spec.changes | {"name": run_id, "spec": asdict(spec)}
        for k in ("gpu_hours", "n_gpus"):
            cfg.pop(k, None)
        return train_ultralytics(cfg, self.data_yaml(spec, fold), self.runs, **(self.train_kw | self.smoke_kw))

    def evaluate(self, spec: Spec, fold: int = 0) -> dict:
        """Train if needed, then evaluate on the fold's validation clips and log the result."""
        run_id, val = self.run_id(spec, fold), self.val_rows(fold)
        if spec.ensemble:
            predictor = self.ensemble_predictor(spec)
            dets = predict_eval_frames(predictor, self.raw, val, desc=f"eval {run_id}")
            stats = evaluate(build_gt(val, self.boxes), dets) | density_breakdown(val, self.boxes, dets)
            print(run_id, format_stats(stats))
            log_result(run_id, stats, config=asdict(spec), family="ensemble", fold=fold)
            return stats
        weights, cfg = self.train(spec, fold)
        return evaluate_ultralytics(weights, cfg, self.raw, val, self.boxes, fold=fold, run_id=run_id,
                                    wrap=lambda p: self.predictor(spec, weights, cfg), config_extra={"spec": asdict(spec)},
                                    **({"imgsz": self.smoke_kw["imgsz"]} if self.smoke_kw else {}))

    def submission(self, spec: Spec, fold: int = -1, limit: int | None = None) -> pd.DataFrame:
        """Validated test submission of a spec trained on ``fold`` (-1 = all labelled clips)."""
        from .submission import make_submission, predict_test, validate_submission

        if spec.ensemble:
            predictor = self.ensemble_predictor(spec)
        else:
            weights, cfg = self.train(spec, fold)
            predictor = self.predictor(spec, weights, cfg)
        test = pd.read_csv(self.raw / "test_images.csv")
        sub = make_submission(predict_test(predictor, self.raw, limit=limit), test)
        validate_submission(sub, test)
        return sub


def oof_table(log: pd.DataFrame, names: list[str], folds=(0, 1, 2)) -> pd.DataFrame:
    """mAP of each spec on each fold (run ids ``name`` / ``name_f{k}``) and the mean over folds."""
    rows = {}
    for n in names:
        ids = {k: n if k == 0 else f"{n}_f{k}" for k in folds}
        rows[n] = {f"fold {k}": log.loc[i, "mAP"] if i in log.index else float("nan") for k, i in ids.items()}
    t = pd.DataFrame(rows).T
    t["mean"] = t.mean(axis=1)
    t["std"] = t.drop(columns="mean").std(axis=1)
    return t.sort_values("mean", ascending=False)
