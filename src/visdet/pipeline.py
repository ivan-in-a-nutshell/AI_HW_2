"""Final-pipeline experiments as declarative specs (notebook 07).

A :class:`Spec` is the best detector of notebooks 03-06 plus *one* change: training data (frame step,
ignore-region masking, tiles), augmentation/model settings (config overrides), or the inference
strategy (TTA, tiles, ensembles). :class:`Pipeline` trains and evaluates specs on any fold under the
benchmark protocol and writes the test submission of a spec.

The whole notebook needs more than one 12-hour Kaggle session. :meth:`Pipeline.run` therefore never
raises: a run that would not fit in the session stops all later work, so the notebook still finishes and
its output can be attached to the next session, where :func:`carry_over` restores everything done so far.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from .data import fold_split
from .evaluate import EXPERIMENTS, build_gt, density_breakdown, evaluate, format_stats, log_result
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


KAGGLE_INPUT = Path("/kaggle/input")
_SKIP = {"images", "labels", "train", "val", "test", "weights", ".git"}   # large or irrelevant folders
SESSION_HOURS = float(os.environ.get("VISDET_SESSION_HOURS", "12" if KAGGLE_INPUT.exists() else "inf"))
SAVE_MARGIN = 0.4   # hours left free at the end of a session for Kaggle to save the output
EVAL_HOURS = 0.3    # rough upper bound of one evaluation / test prediction


class OutOfTime(RuntimeError):
    pass


def attached(pattern: str, max_depth: int = 5) -> list[Path]:
    """Files matching ``pattern`` in the attached Kaggle inputs, whatever the mount layout
    (``<slug>/``, ``notebooks/<user>/<slug>/``, ``datasets/<user>/<slug>/<folder>/``)."""
    found = []
    for root, dirs, _ in os.walk(KAGGLE_INPUT):
        found += Path(root).glob(pattern)
        depth = len(Path(root).relative_to(KAGGLE_INPUT).parts)
        dirs[:] = [d for d in dirs if d not in _SKIP] if depth < max_depth else []
    return sorted(found)


def carry_over(runs_dir: Path, experiments: Path = EXPERIMENTS) -> list[str]:
    """Resume from attached Kaggle outputs (an earlier session of this notebook, notebooks 05/06, or a
    dataset holding a downloaded ``runs/`` folder): copy their Ultralytics runs (without ``last.pt``) into
    ``runs_dir`` and add the experiment-log rows this checkout does not have yet. Each session's output
    then holds everything done so far, so only the latest one needs to be attached."""
    runs_dir, experiments = Path(runs_dir), Path(experiments)
    restored = []
    for best in attached("runs/*/weights/best.pt"):
        src, dst = best.parents[1], Path(runs_dir) / best.parents[1].name
        if not (dst / "weights" / "best.pt").exists():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("last.pt"), dirs_exist_ok=True)
            restored.append(src.name)
    results = experiments / "results.csv"
    log = pd.read_csv(results) if results.exists() else pd.DataFrame({"run_id": []})
    n = len(log)
    for other in attached("experiments/results.csv"):
        new = pd.read_csv(other)
        new = new[~new["run_id"].isin(log["run_id"])]
        for run_id in new["run_id"]:
            if (other.parent / "runs" / run_id).is_dir():
                shutil.copytree(other.parent / "runs" / run_id, experiments / "runs" / run_id, dirs_exist_ok=True)
        log = pd.concat([log, new], ignore_index=True)
    if len(log) > n:
        log.to_csv(results, index=False)
    return restored


def _session_start() -> float:
    try:
        import psutil
        return psutil.Process().create_time()   # kernel start = start of the Kaggle session
    except Exception:
        return time.time()


def find_weights(run: str, runs_dir: Path) -> Path:
    """Checkpoint of ``run`` in this session's runs or in an attached Kaggle output."""
    for p in [Path(runs_dir) / run / "weights" / "best.pt", *attached(f"runs/{run}/weights/best.pt")]:
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
        self.deadline = _session_start() + (SESSION_HOURS - SAVE_MARGIN) * 3600
        self.stopped, self.failed = None, []

    # ------------------------------------------------------------------ session control
    def hours_left(self) -> float:
        return (self.deadline - time.time()) / 3600

    def _need(self, hours: float, what: str):
        if self.hours_left() < hours:
            raise OutOfTime(f"{what} needs ~{hours:.1f} h, {max(self.hours_left(), 0):.1f} h left in this session")

    def attempt(self, what: str, fn, *args, **kw):
        """``fn(*args, **kw)`` that never raises, so the notebook always finishes and its output can be
        attached to the next session. Work that does not fit in the session stops everything after it;
        an error is printed and recorded, and :meth:`end_stage` stops the stages that build on it."""
        if self.stopped:
            print(f"skipped {what}: {self.stopped}")
            return None
        try:
            self._need(EVAL_HOURS, what)
            return fn(*args, **kw)
        except OutOfTime as e:
            self.stopped = f"out of session time ({e})"
            print(f"stopped: {e}")
        except Exception:
            traceback.print_exc()
            self.failed.append(what)
        return None

    def run(self, spec: Spec, fold: int = 0) -> dict | None:
        """:meth:`evaluate` under :meth:`attempt`; runs already in the experiment log are not repeated."""
        run_id, results = self.run_id(spec, fold), EXPERIMENTS / "results.csv"
        if results.exists() and run_id in set(pd.read_csv(results)["run_id"]):
            print(f"{run_id}: already evaluated")
            return None
        return self.attempt(run_id, self.evaluate, spec, fold)

    def end_stage(self, stage: str):
        """Later stages build on this one: stop if any run so far failed or did not fit in the session."""
        if self.failed and not self.stopped:
            self.stopped = f"failed runs {self.failed}"
        if self.stopped:
            print(f"stage {stage} incomplete: {self.stopped}.\n"
                  "Fix the error (if any), attach this version's output and run the notebook again.")
        else:
            print(f"stage {stage} complete ({self.hours_left():.1f} h left in this session)")

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
        self._need(((cfg | self.smoke_kw).get("time", 1.0) or 0) + 2 * EVAL_HOURS, f"training {run_id}")
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
