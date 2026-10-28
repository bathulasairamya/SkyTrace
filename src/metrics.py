"""Segmentation Metrics: IoU, Dice/F1, Precision, Recall, Pixel Accuracy, and Losses."""

import torch
import numpy as np


def compute_metrics(pred_prob: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7):
    """Compute binary segmentation metrics given predicted probabilities and target masks.
    
    Returns both normalized rates and integer counts for global micro-aggregation.

    Convention:
        If both prediction and ground truth contain zero building pixels (TP=0, FP=0, FN=0),
        IoU and Dice are defined as 1.0 (perfect true negative prediction).
        If ground truth has buildings but prediction is empty, or vice-versa, IoU is 0.0.
    """
    pred_bin = (pred_prob > threshold).float()
    target_bin = (target > 0.5).float()

    # Flatten spatial dimensions
    p = pred_bin.view(-1)
    t = target_bin.view(-1)

    tp = int((p * t).sum().item())
    fp = int((p * (1.0 - t)).sum().item())
    fn = int(((1.0 - p) * t).sum().item())
    tn = int(((1.0 - p) * (1.0 - t)).sum().item())

    # Empty tile convention
    if tp == 0 and fp == 0 and fn == 0:
        iou = 1.0
        dice = 1.0
        precision = 1.0
        recall = 1.0
    else:
        iou = tp / (tp + fp + fn + eps)
        dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
        precision = tp / (tp + fp + eps) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn + eps) if (tp + fn) > 0 else 0.0

    total_pixels = tp + fp + fn + tn
    accuracy = (tp + tn) / (total_pixels + eps) if total_pixels > 0 else 1.0

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


class DiceLoss(torch.nn.Module):
    """Soft Dice Loss computed per batch sample and averaged."""
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred_prob: torch.Tensor, target: torch.Tensor):
        # pred_prob: (B, 1, H, W)
        intersection = (pred_prob * target).sum(dim=(2, 3))
        cardinality = pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return 1.0 - dice.mean()


class BCEDiceLoss(torch.nn.Module):
    """Combined Binary Cross Entropy + Dice Loss (matches author training setup)."""
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        bce_loss = self.bce(logits, target)
        pred_prob = torch.sigmoid(logits)
        dice_loss = self.dice(pred_prob, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

