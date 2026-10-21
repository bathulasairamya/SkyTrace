"""Segmentation Metrics: IoU, Dice/F1, Precision, Recall, and Pixel Accuracy."""

import torch
import numpy as np


def compute_metrics(pred_prob: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7):
    """Compute binary segmentation metrics given predicted probabilities and target masks.
    
    Args:
        pred_prob: Tensor of shape (B, 1, H, W) with values in [0, 1]
        target: Tensor of shape (B, 1, H, W) with binary values {0, 1}
        threshold: Decision threshold for classification
    """
    pred_bin = (pred_prob > threshold).float()
    target_bin = (target > 0.5).float()

    # Flatten spatial dimensions
    p = pred_bin.view(-1)
    t = target_bin.view(-1)

    tp = (p * t).sum().item()
    fp = (p * (1 - t)).sum().item()
    fn = ((1 - p) * t).sum().item()
    tn = ((1 - p) * (1 - t)).sum().item()

    iou = (tp + eps) / (tp + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
    }


class DiceLoss(torch.nn.Module):
    """Soft Dice Loss for backpropagation."""
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred_prob: torch.Tensor, target: torch.Tensor):
        intersection = (pred_prob * target).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.eps) / (union + self.eps)
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
