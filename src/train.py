"""Modular, GPU-Accelerated / CPU-Compatible Fine-Tuning Pipeline.

Features:
- Auto-detects NVIDIA CUDA GPU or runs on CPU.
- Supports mixed-precision (FP16/AMP) for fast GPU execution.
- Implements BCE + Dice loss, Adam optimizer (lr=1e-4), and ReduceLROnPlateau.
- Monitors validation IoU and saves checkpoints/best_finetuned_model.pth.
- Generates loss/IoU convergence curves in logs/.
"""

import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import time
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.models import build_model
from src.dataset import get_dataloaders
from src.metrics import BCEDiceLoss, compute_metrics


def train_model(
    model_type: str = "deeplabv3",
    checkpoint_path: str = "checkpoints/best_deeplab.pth",
    output_checkpoint: str = "checkpoints/best_finetuned_model.pth",
    epochs: int = 5,
    batch_size: int = 2,
    lr: float = 1e-4,
    patch_size: int = 512,
    device_name: str = "auto",
    use_amp: bool = True,
):
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # 1. Device Setup
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print("=" * 65)
    print(f"       STARTING TRAINING PIPELINE ON: {device.type.upper()}       ")
    print("=" * 65)

    if device.type == "cuda":
        print(f"  GPU Name:          {torch.cuda.get_device_name(0)}")
        print(f"  VRAM Available:    {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        print(f"  Mixed Precision:   {'Enabled (AMP FP16)' if use_amp else 'Disabled'}")
    else:
        scaler = None
        print("  Running on CPU. Batch size 2, patch size 512.")

    # 2. Build Model & Load Checkpoint
    model = build_model(model_name=model_type, checkpoint_path=checkpoint_path, device=device)

    # 3. DataLoaders
    train_loader, val_loader = get_dataloaders(
        train_json="data/splits/train.json",
        val_json="data/splits/val.json",
        batch_size=batch_size,
        patch_size=patch_size,
        num_workers=2 if device.type == "cuda" else 0,
    )
    print(f"  Train Batches:     {len(train_loader)} (total {len(train_loader.dataset)} samples)")
    print(f"  Val Batches:       {len(val_loader)} (total {len(val_loader.dataset)} samples)")

    # 4. Optimizer, Loss & Scheduler
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, verbose=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_iou": [],
        "val_dice": [],
        "lr": [],
        "epoch_times": [],
    }

    best_val_iou = 0.0

    # 5. Training Loop
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]")
        for imgs, masks in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            if device.type == "cuda" and use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

            running_train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_train_loss / len(train_loader)

        # Validation Loop
        model.eval()
        running_val_loss = 0.0
        val_metrics_list = []

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)

                outputs = model(imgs)
                loss = criterion(outputs, masks)
                running_val_loss += loss.item()

                probs = torch.sigmoid(outputs)
                batch_metrics = compute_metrics(probs, masks, threshold=0.5)
                val_metrics_list.append(batch_metrics)

        avg_val_loss = running_val_loss / len(val_loader)
        avg_val_iou = float(np.mean([m["iou"] for m in val_metrics_list]))
        avg_val_dice = float(np.mean([m["dice"] for m in val_metrics_list]))
        epoch_sec = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(avg_val_iou)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_iou"].append(avg_val_iou)
        history["val_dice"].append(avg_val_dice)
        history["lr"].append(current_lr)
        history["epoch_times"].append(round(epoch_sec, 2))

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val IoU: {avg_val_iou:.4f} | "
            f"Val Dice: {avg_val_dice:.4f} | "
            f"Time: {epoch_sec:.1f}s"
        )

        # Save Best Checkpoint
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            torch.save(model.state_dict(), output_checkpoint)
            print(f"  [*] Saved new best model checkpoint to {output_checkpoint} (Val IoU: {best_val_iou:.4f})")

    # 6. Save Training Log & Plot
    with open("logs/training_log.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Plot Convergence Curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, epochs + 1)

    axes[0].plot(epochs_range, history["train_loss"], label="Train Loss", marker="o", color="#2563eb")
    axes[0].plot(epochs_range, history["val_loss"], label="Val Loss", marker="s", color="#dc2626")
    axes[0].set_title("Loss Curves (BCE + Dice)", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(epochs_range, history["val_iou"], label="Validation IoU", marker="^", color="#16a34a")
    axes[1].plot(epochs_range, history["val_dice"], label="Validation Dice", marker="d", color="#9333ea")
    axes[1].set_title("Validation Accuracy Metrics", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("logs/loss_curve.png", dpi=150)
    plt.close()

    print("\n" + "=" * 65)
    print(f"  TRAINING COMPLETE! Best Validation IoU: {best_val_iou:.4f}")
    print(f"  Checkpoint saved:  {output_checkpoint}")
    print(f"  Training logs:     logs/training_log.json")
    print(f"  Convergence curve: logs/loss_curve.png")
    print("=" * 65)

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Satellite Building Segmentation Fine-Tuning")
    parser.add_argument("--model_type", type=str, default="deeplabv3", choices=["deeplabv3", "author_unet", "paper_unet"])
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_deeplab.pth")
    parser.add_argument("--output", type=str, default="checkpoints/best_finetuned_model.pth")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    train_model(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
        output_checkpoint=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patch_size=args.patch_size,
        device_name=args.device,
    )
