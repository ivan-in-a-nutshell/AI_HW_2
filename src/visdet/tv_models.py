"""Config-driven torchvision detectors: Faster R-CNN (two-stage) and RetinaNet (one-stage, anchor-based).

Both start from COCO-pretrained ResNet50-FPN v2 weights. Anchor sizes, sampling, NMS and loss
settings come from the experiment config, so every tweak is one line in a YAML file.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights, RetinaNet_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2, retinanet_resnet50_fpn_v2)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead, _sum
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import generalized_box_iou_loss, sigmoid_focal_loss

from .data import NUM_CLASSES


def anchor_generator(cfg: dict) -> AnchorGenerator:
    """One anchor-size group per FPN level; ``scales_per_octave`` sizes per group (RetinaNet uses 3)."""
    n = cfg.get("scales_per_octave", 1)
    sizes = tuple(tuple(int(round(s * 2 ** (i / n))) for i in range(n)) for s in cfg["anchor_sizes"])
    ratios = (tuple(cfg["aspect_ratios"]),) * len(sizes)
    return AnchorGenerator(sizes, ratios)


# ----------------------------------------------------------------------------- Faster R-CNN

class GIoURoIHeads(RoIHeads):
    """RoI head whose box loss is GIoU on decoded boxes instead of smooth-L1 on deltas."""

    giou_weight = 10.0  # GIoU is O(1) per box; smooth-L1 on encoded deltas is ~10x smaller

    def forward(self, features, proposals, image_shapes, targets=None):
        if not self.training:
            return super().forward(features, proposals, image_shapes, targets)
        proposals, _, labels, regression_targets = self.select_training_samples(proposals, targets)
        box_features = self.box_head(self.box_roi_pool(features, proposals, image_shapes))
        class_logits, box_regression = self.box_predictor(box_features)

        labels = torch.cat(labels)
        loss_cls = F.cross_entropy(class_logits, labels)
        pos = torch.where(labels > 0)[0]
        deltas = box_regression.reshape(len(labels), -1, 4)[pos, labels[pos]]
        props = torch.cat(proposals)[pos]
        pred = self.box_coder.decode_single(deltas.float(), props)
        target = self.box_coder.decode_single(torch.cat(regression_targets)[pos], props)
        loss_box = generalized_box_iou_loss(pred, target, reduction="sum") / labels.numel()
        return [], {"loss_classifier": loss_cls, "loss_box_reg": self.giou_weight * loss_box}


def build_faster_rcnn(cfg: dict) -> nn.Module:
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        min_size=cfg["min_size"], max_size=cfg["max_size"],
        rpn_pre_nms_top_n_train=cfg.get("rpn_pre_nms_top_n_train", 2000),
        rpn_pre_nms_top_n_test=cfg.get("rpn_pre_nms_top_n_test", 1000),
        rpn_post_nms_top_n_train=cfg.get("rpn_post_nms_top_n_train", 2000),
        rpn_post_nms_top_n_test=cfg.get("rpn_post_nms_top_n_test", 1000),
        rpn_batch_size_per_image=cfg.get("rpn_batch_size_per_image", 256),
        box_batch_size_per_image=cfg.get("box_batch_size_per_image", 512),
        box_positive_fraction=cfg.get("box_positive_fraction", 0.25),
        box_detections_per_img=cfg.get("detections_per_img", 500),
        box_score_thresh=cfg.get("score_thresh", 0.01),
        box_nms_thresh=cfg.get("nms_thresh", 0.5),
    )
    anchors = anchor_generator(cfg)
    if anchors.num_anchors_per_location() != model.rpn.anchor_generator.num_anchors_per_location():
        raise ValueError("keep one size and three ratios per level so the pretrained RPN head fits")
    model.rpn.anchor_generator = anchors
    model.roi_heads.box_predictor = FastRCNNPredictor(1024, NUM_CLASSES + 1)  # + background
    if cfg.get("box_loss", "smooth_l1") == "giou":
        model.roi_heads.__class__ = GIoURoIHeads
        model.roi_heads.giou_weight = cfg.get("giou_weight", 10.0)
    return model


# ----------------------------------------------------------------------------- RetinaNet

class FocalClassificationHead(RetinaNetClassificationHead):
    """RetinaNet classification head with configurable focal-loss alpha / gamma."""

    alpha, gamma = 0.25, 2.0

    def compute_loss(self, targets, head_outputs, matched_idxs):
        losses = []
        for tgt, logits, matched in zip(targets, head_outputs["cls_logits"], matched_idxs):
            fg = matched >= 0
            gt = torch.zeros_like(logits)
            gt[fg, tgt["labels"][matched[fg]]] = 1.0
            valid = matched != self.BETWEEN_THRESHOLDS
            loss = sigmoid_focal_loss(logits[valid], gt[valid], alpha=self.alpha, gamma=self.gamma,
                                      reduction="sum")
            losses.append(loss / max(1, fg.sum()))
        return _sum(losses) / len(targets)


def build_retinanet(cfg: dict) -> nn.Module:
    anchors = anchor_generator(cfg)
    model = retinanet_resnet50_fpn_v2(
        weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT,
        min_size=cfg["min_size"], max_size=cfg["max_size"],
        detections_per_img=cfg.get("detections_per_img", 500),
        topk_candidates=cfg.get("topk_candidates", 1000),
        score_thresh=cfg.get("score_thresh", 0.01),
        nms_thresh=cfg.get("nms_thresh", 0.5),
        fg_iou_thresh=cfg.get("fg_iou_thresh", 0.5),
        bg_iou_thresh=cfg.get("bg_iou_thresh", 0.4),
    )
    num_anchors = anchors.num_anchors_per_location()[0]
    pretrained_anchors = model.anchor_generator.num_anchors_per_location()[0]
    model.anchor_generator = anchors
    if num_anchors != pretrained_anchors:  # regression head shape depends on the anchor count
        reg_loss = model.head.regression_head._loss_type
        model.head.regression_head = type(model.head.regression_head)(
            256, num_anchors, norm_layer=partial(nn.GroupNorm, 32))
        model.head.regression_head._loss_type = reg_loss
    head = FocalClassificationHead(256, num_anchors, NUM_CLASSES, norm_layer=partial(nn.GroupNorm, 32))
    head.alpha, head.gamma = cfg.get("focal_alpha", 0.25), cfg.get("focal_gamma", 2.0)
    model.head.classification_head = head
    if "box_loss" in cfg:
        model.head.regression_head._loss_type = cfg["box_loss"]
    return model


def build_model(cfg: dict) -> nn.Module:
    return {"faster_rcnn": build_faster_rcnn, "retinanet": build_retinanet}[cfg["arch"]](cfg)


def label_offset(cfg: dict) -> int:
    """Faster R-CNN reserves label 0 for background; RetinaNet uses 0-based labels."""
    return 1 if cfg["arch"] == "faster_rcnn" else 0


class TorchvisionPredictor:
    """Wrap a torchvision detector with the common predictor interface (BGR frames in, xyxy+score+cls out)."""

    def __init__(self, model: nn.Module, cfg: dict, device=None, half: bool = True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.offset = label_offset(cfg)
        self.half = half and self.device == "cuda"

    @torch.no_grad()
    def __call__(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        x = [torch.from_numpy(f[:, :, ::-1].copy()).permute(2, 0, 1).to(self.device).float() / 255 for f in frames]
        with torch.autocast("cuda", enabled=self.half):
            outs = self.model(x)
        return [np.column_stack([o["boxes"].float().cpu().numpy(), o["scores"].float().cpu().numpy(),
                                 o["labels"].cpu().numpy() - self.offset]) for o in outs]


@torch.no_grad()
def proposal_recall(model: nn.Module, frames: list[np.ndarray], objects: list[np.ndarray],
                    top_n=(100, 300, 1000, 2000), iou_thr: float = 0.5, device=None) -> pd.DataFrame:
    """Recall of the RPN proposals (top-N by objectness) and of the final detections, by object size.

    ``objects[i]`` holds the xywh boxes of frame i in frame pixels. Shows what the first stage finds
    versus what survives the second stage (classification + refinement + NMS).
    """
    from torchvision.ops import box_iou

    device = device or next(model.parameters()).device
    model.eval()
    rows = []
    for frame, obj in zip(frames, objects):
        if not len(obj):
            continue
        x = [torch.from_numpy(frame[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255).to(device)]
        images, _ = model.transform(x)
        proposals, _ = model.rpn(images, model.backbone(images.tensors))
        (h, w), (H, W) = images.image_sizes[0], frame.shape[:2]
        props = proposals[0] * torch.tensor([W / w, H / h, W / w, H / h], device=device)
        dets = model(x)[0]["boxes"]
        gt = torch.as_tensor(np.c_[obj[:, :2], obj[:, :2] + obj[:, 2:]], dtype=torch.float32, device=device)
        hit = {f"rpn top {n}": box_iou(gt, props[:n]).max(1).values >= iou_thr for n in top_n}
        hit["final detections"] = (box_iou(gt, dets).max(1).values >= iou_thr) if len(dets) else torch.zeros(len(gt), dtype=torch.bool)
        size = np.sqrt(obj[:, 2] * obj[:, 3])
        rows.append(pd.DataFrame({"size": size, **{k: v.cpu().numpy() for k, v in hit.items()}}))
    df = pd.concat(rows, ignore_index=True)
    df["size_bucket"] = pd.cut(df.pop("size"), [0, 16, 32, 96, np.inf],
                               labels=["tiny <16", "small 16-32", "medium 32-96", "large >96"])
    out = df.groupby("size_bucket", observed=False).mean()
    out.loc["all"] = df.drop(columns="size_bucket").mean()
    return out
