"""Train and evaluate a torchvision detector under the fixed benchmark protocol.

Single GPU:  python -m visdet.tv_train --config configs/frcnn_base.yaml --prepared <dir> --out runs/frcnn_base
Two GPUs:    torchrun --nproc_per_node 2 -m visdet.tv_train ...  (DDP; evaluation runs on rank 0)

Each epoch logs losses and validation metrics to ``<out>/history.csv``; the best checkpoint by
validation mAP is kept, and the final metrics + FPS are appended to ``experiments/results.csv``.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .config import load_config
from .data import find_raw_root, fold_split, read_prepared
from .evaluate import build_gt, density_breakdown, evaluate, format_stats, log_result, measure_fps
from .infer import load_frame, predict_eval_frames
from .tv_models import TorchvisionPredictor, build_model, label_offset


class DetectionDataset(Dataset):
    """Prepared (masked, test-scale) frames with their YOLO labels as torchvision targets."""

    def __init__(self, root: Path, rows: pd.DataFrame, offset: int, train: bool):
        self.root, self.paths, self.offset, self.train = Path(root), rows["path"].tolist(), offset, train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.root / self.paths[i]
        img = cv2.imread(str(path))[:, :, ::-1]
        H, W = img.shape[:2]
        lbl = path.parents[2] / "labels" / path.parent.name / (path.stem + ".txt")
        a = np.loadtxt(lbl, ndmin=2, dtype=np.float32) if lbl.stat().st_size else np.zeros((0, 5), np.float32)
        cls, (cx, cy, w, h) = a[:, 0].astype(np.int64), a[:, 1:].T
        boxes = np.stack([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H], 1)
        if self.train and random.random() < 0.5:  # horizontal flip
            img = img[:, ::-1]
            boxes = boxes[:, [2, 1, 0, 3]] * [-1, 1, -1, 1] + [W, 0, W, 0]
        x = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255
        target = {"boxes": torch.from_numpy(boxes.astype(np.float32)).reshape(-1, 4),
                  "labels": torch.from_numpy(cls + self.offset)}
        return x, target


def collate(batch):
    return tuple(zip(*batch))


def lr_factor(it: int, iters_per_epoch: int, cfg: dict) -> float:
    """Linear warm-up, then step or cosine decay (schedule defined in epochs)."""
    warm = cfg.get("warmup_iters", 500)
    if it < warm:
        return 0.001 + (1 - 0.001) * it / warm
    epoch = it / iters_per_epoch
    if cfg.get("lr_schedule", "step") == "cosine":
        return 0.5 * (1 + math.cos(math.pi * epoch / cfg["epochs"]))
    return 0.1 ** sum(epoch >= s for s in cfg.get("lr_steps", [8, 11]))


def run_eval(model, cfg, raw_root, val_images, gt, device) -> tuple[dict, pd.DataFrame]:
    predictor = TorchvisionPredictor(model, cfg, device=device)
    dets = predict_eval_frames(predictor, raw_root, val_images, batch_size=cfg.get("eval_batch", 4), desc=None)
    model.train()
    return evaluate(gt, dets), dets


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--raw", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--train-step", type=int, default=10)
    p.add_argument("--val-step", type=int, default=10)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-hours", type=float, default=11.0, help="stop safely before the session limit")
    p.add_argument("--limit", type=int, default=0, help="debug: cap train/val frames")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results", default=None, help="experiment log (default: $VISDET_EXPERIMENTS/results.csv)")
    args = p.parse_args(argv)

    # ---- distributed setup
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if torch.cuda.is_available() else torch.device("cpu")
    main_proc = rank == 0
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    cfg = load_config(args.config)
    if args.epochs:
        cfg["epochs"] = args.epochs
    raw_root = args.raw or find_raw_root()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    # ---- data
    images, boxes, folds = read_prepared(args.prepared)
    train_rows, val_rows = fold_split(images, folds, args.fold, args.train_step, args.val_step)
    if args.limit:
        train_rows, val_rows = train_rows.sample(min(args.limit, len(train_rows)), random_state=0), val_rows.head(args.limit)
    ds = DetectionDataset(args.prepared, train_rows, label_offset(cfg), train=True)
    sampler = DistributedSampler(ds, shuffle=True, seed=args.seed) if world > 1 else None
    loader = DataLoader(ds, batch_size=cfg.get("batch_size", 4), shuffle=sampler is None, sampler=sampler,
                        num_workers=args.workers, collate_fn=collate, drop_last=True, persistent_workers=args.workers > 0)
    gt = build_gt(val_rows, boxes) if main_proc else None
    if main_proc:
        print(f"{cfg['name']}: fold {args.fold}, {len(train_rows)} train frames, {len(val_rows)} val frames, world {world}")

    # ---- model / optimiser
    model = build_model(cfg).to(device)
    params = [q for q in model.parameters() if q.requires_grad]
    # cfg["lr"] is defined for a global batch of cfg["lr_batch"] frames (linear scaling rule)
    base_lr = cfg["lr"] * cfg.get("batch_size", 4) * world / cfg.get("lr_batch", 8)
    opt = torch.optim.SGD(params, lr=base_lr, momentum=0.9, weight_decay=cfg.get("weight_decay", 1e-4))
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index]) if world > 1 else model
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    history, best, it, t0 = [], -1.0, 0, time.time()
    for epoch in range(cfg["epochs"]):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        sums, n = {}, 0
        for images_b, targets in loader:
            images_b = [x.to(device, non_blocking=True) for x in images_b]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            for g in opt.param_groups:
                g["lr"] = base_lr * lr_factor(it, len(loader), cfg)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                losses = ddp(images_b, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at iter {it}: {losses}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, cfg.get("clip_grad", 10.0))
            scaler.step(opt)
            scaler.update()
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + v.item()
            n, it = n + 1, it + 1

        # ---- per-epoch validation (rank 0) and checkpointing
        hours = (time.time() - t0) / 3600
        last = epoch == cfg["epochs"] - 1 or hours > args.max_hours
        if main_proc:
            row = {"epoch": epoch + 1, "hours": round(hours, 3), "lr": opt.param_groups[0]["lr"],
                   **{k: v / max(n, 1) for k, v in sums.items()}}
            if (epoch + 1) % cfg.get("eval_every", 1) == 0 or last:
                stats, _ = run_eval(model, cfg, raw_root, val_rows, gt, device)
                row |= {k: stats[k] for k in ("mAP", "AP50", "AP_s", "AP_m", "AP_l", "AR")}
                if stats["mAP"] > best:
                    best = stats["mAP"]
                    torch.save(model.state_dict(), args.out / "best.pt")
            history.append(row)
            pd.DataFrame(history).to_csv(args.out / "history.csv", index=False)
            torch.save(model.state_dict(), args.out / "last.pt")
            print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
        if world > 1:
            dist.barrier()
        if last:
            break

    # ---- final metrics of the best checkpoint
    if main_proc:
        wall = (time.time() - t0) / 3600
        model.load_state_dict(torch.load(args.out / "best.pt", map_location=device))
        stats, dets = run_eval(model, cfg, raw_root, val_rows, gt, device)
        stats |= density_breakdown(val_rows, boxes, dets)
        dets[dets["score"] >= 0.01].round(2).to_csv(args.out / "val_dets.csv.gz", index=False)
        predictor = TorchvisionPredictor(model, cfg, device=device)
        frames = [load_frame(Path(raw_root) / f, (w, h)) for f, w, h in
                  val_rows[["file_name", "out_w", "out_h"]].head(55).itertuples(index=False)]
        stats["fps"] = measure_fps(predictor, frames)
        print("best:", format_stats(stats), f"fps {stats['fps']:.1f}")
        log_result(cfg["name"], stats, args.results, history=pd.DataFrame(history), config=cfg, family=cfg["arch"],
                   fold=args.fold, epochs=len(history), gpu_hours=round(wall * world, 2), n_gpus=world,
                   train_frames=len(train_rows))
    if world > 1:
        dist.destroy_process_group()


def launch(config, prepared, out, fold: int = 0, gpus: int | None = None, **options):
    """Run :func:`main` in a subprocess (``torchrun`` on several GPUs) and stream its log.

    ``options`` become CLI flags, e.g. ``launch(..., train_step=5)`` -> ``--train-step 5``.
    """
    gpus = torch.cuda.device_count() if gpus is None else gpus
    head = ["torchrun", "--standalone", f"--nproc_per_node={gpus}", "-m"] if gpus > 1 else [sys.executable, "-m"]
    cmd = head + ["visdet.tv_train", "--config", str(config), "--prepared", str(prepared),
                  "--out", str(out), "--fold", str(fold)]
    for k, v in options.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
        for line in proc.stdout:
            if "warn" not in line.lower():
                print(line, end="", flush=True)
    if proc.returncode:
        raise RuntimeError(f"training failed ({proc.returncode}): {' '.join(cmd)}")


if __name__ == "__main__":
    main()
