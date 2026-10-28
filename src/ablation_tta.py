"""Test-Time Augmentation (TTA) Ablation Study.

Compares:
1. Standard Inference (no TTA - single view)
2. 2-View TTA (Horizontal Flip)
3. 4-View TTA (Horizontal Flip + Vertical Flip + 90-degree Rotation)

Evaluates on the 10 holdout test tiles measuring:
- Pixel IoU (Macro and Global Micro)
- Dice Score (F1)
- Precision & Recall
- Connected-Component Object-Level F1 (IoU >= 0.5)
- Average Inference Latency (seconds/tile)
- Relative Computational Overhead Factor (1.0x baseline)
Exports: outputs/tta_ablation.json
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import cv2
from tqdm import tqdm

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import build_model
from src.inference import load_inference_image, predict_sliding_window, predict_sliding_window_tta
from src.metrics import compute_metrics
from src.evaluate import evaluate_connected_component_object_f1


def run_tta_ablation(
    model_name: str = "deeplabv3",
    checkpoint_path: str = "checkpoints/best_deeplab.pth",
    test_json: str = "data/splits/test.json",
    output_json: str = "outputs/tta_ablation.json",
    device_name: str = "auto",
):
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else device_name
    )

    print("=" * 70)
    print("      STARTING TEST-TIME AUGMENTATION (TTA) ABLATION EXPERIMENT       ")
    print(f"  Device:     {device.type.upper()}")
    print(f"  Model:      {model_name} ({checkpoint_path})")
    print(f"  Test Tiles: {test_json}")
    print("=" * 70)

    model = build_model(model_name=model_name, checkpoint_path=checkpoint_path, device=device)
    model.eval()

    with open(test_json, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    modes = [
        {"id": "no_tta", "name": "Standard (No TTA)", "views": 1},
        {"id": "2_view", "name": "2-View TTA (H-Flip)", "views": 2},
        {"id": "4_view", "name": "4-View TTA (H+V+Rot90)", "views": 4},
    ]

    ablation_results = {}

    for mode_cfg in modes:
        mode_id = mode_cfg["id"]
        mode_name = mode_cfg["name"]
        print(f"\nEvaluating configuration: {mode_name} ...")

        tile_metrics = []
        latencies = []

        for record in tqdm(test_records, desc=f"  TTA [{mode_id}]"):
            tile_id = record["tile_id"]
            img_path = record["image_path"]
            mask_path = record["mask_path"]

            raw_rgb, norm_rgb, meta = load_inference_image(img_path)
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                raise FileNotFoundError(f"Missing or corrupt ground truth mask: {mask_path}")

            # Measure inference latency with high-resolution timer
            t_start = time.perf_counter()

            if mode_id == "no_tta":
                prob_map, pred_mask = predict_sliding_window(
                    model=model,
                    img_norm=norm_rgb,
                    patch_size=512,
                    stride=256,
                    device=device,
                    threshold=0.5,
                )
            elif mode_id == "2_view":
                prob_map, pred_mask = predict_sliding_window_tta(
                    model=model,
                    img_norm=norm_rgb,
                    patch_size=512,
                    stride=256,
                    device=device,
                    threshold=0.5,
                    mode="2-view",
                )
            elif mode_id == "4_view":
                prob_map, pred_mask = predict_sliding_window_tta(
                    model=model,
                    img_norm=norm_rgb,
                    patch_size=512,
                    stride=256,
                    device=device,
                    threshold=0.5,
                    mode="4-view",
                )
            else:
                raise ValueError(f"Unknown mode_id: {mode_id}")

            t_elapsed = time.perf_counter() - t_start
            latencies.append(t_elapsed)

            # Compute Pixel-Level Metrics
            prob_tensor = torch.tensor(prob_map).unsqueeze(0).unsqueeze(0)
            gt_tensor = torch.tensor(gt_mask / 255.0).unsqueeze(0).unsqueeze(0)
            pm = compute_metrics(prob_tensor, gt_tensor, threshold=0.5)

            # Compute Connected-Component Object Metrics
            om = evaluate_connected_component_object_f1(pred_mask, gt_mask, iou_thresh=0.5)

            tile_metrics.append({
                "tile_id": tile_id,
                "latency_sec": round(t_elapsed, 4),
                "pixel_iou": pm["iou"],
                "pixel_dice": pm["dice"],
                "precision": pm["precision"],
                "recall": pm["recall"],
                "accuracy": pm["accuracy"],
                "tp": pm["tp"],
                "fp": pm["fp"],
                "fn": pm["fn"],
                "tn": pm["tn"],
                "cc_f1_iou50": om["f1"],
                "object_tp": om["tp"],
                "object_fp": om["fp"],
                "object_fn": om["fn"],
            })

        # Global micro metrics aggregation
        tot_tp = sum(m["tp"] for m in tile_metrics)
        tot_fp = sum(m["fp"] for m in tile_metrics)
        tot_fn = sum(m["fn"] for m in tile_metrics)
        tot_tn = sum(m["tn"] for m in tile_metrics)
        tot_pix = tot_tp + tot_fp + tot_fn + tot_tn

        global_iou = tot_tp / (tot_tp + tot_fp + tot_fn + 1e-7)
        global_dice = (2 * tot_tp) / (2 * tot_tp + tot_fp + tot_fn + 1e-7)
        global_prec = tot_tp / (tot_tp + tot_fp + 1e-7)
        global_rec = tot_tp / (tot_tp + tot_fn + 1e-7)

        mean_latency = float(np.mean(latencies))

        ablation_results[mode_id] = {
            "name": mode_name,
            "views": mode_cfg["views"],
            "avg_latency_sec": round(mean_latency, 3),
            "macro_metrics": {
                "mean_pixel_iou": float(np.mean([m["pixel_iou"] for m in tile_metrics])),
                "mean_pixel_dice": float(np.mean([m["pixel_dice"] for m in tile_metrics])),
                "mean_precision": float(np.mean([m["precision"] for m in tile_metrics])),
                "mean_recall": float(np.mean([m["recall"] for m in tile_metrics])),
                "mean_accuracy": float(np.mean([m["accuracy"] for m in tile_metrics])),
                "mean_cc_f1_iou50": float(np.mean([m["cc_f1_iou50"] for m in tile_metrics])),
            },
            "global_micro_metrics": {
                "global_iou": float(global_iou),
                "global_dice": float(global_dice),
                "global_precision": float(global_prec),
                "global_recall": float(global_rec),
            },
            "object_counts": {
                "total_tp": sum(m["object_tp"] for m in tile_metrics),
                "total_fp": sum(m["object_fp"] for m in tile_metrics),
                "total_fn": sum(m["object_fn"] for m in tile_metrics),
            },
            "per_tile": tile_metrics,
        }

    # Compute relative speedup / slowdown factor relative to base
    base_latency = ablation_results["no_tta"]["avg_latency_sec"]
    for mode_id, data in ablation_results.items():
        data["slowdown_factor"] = round(data["avg_latency_sec"] / max(base_latency, 1e-6), 2)

    # Export machine-readable artifact
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)

    print("\n" + "=" * 80)
    print("                    TTA ABLATION SUMMARY RESULTS                         ")
    print("=" * 80)
    print(f"{'Configuration':<25} | {'Macro IoU':<10} | {'Micro IoU':<10} | {'Dice (F1)':<10} | {'CC F1':<10} | {'Latency':<10} | {'Slowdown':<8}")
    print("-" * 80)
    for mode_id, res in ablation_results.items():
        m = res["macro_metrics"]
        g = res["global_micro_metrics"]
        print(
            f"{res['name']:<25} | "
            f"{m['mean_pixel_iou']:<10.4f} | "
            f"{g['global_iou']:<10.4f} | "
            f"{m['mean_pixel_dice']:<10.4f} | "
            f"{m['mean_cc_f1_iou50']:<10.4f} | "
            f"{res['avg_latency_sec']:<8.2f}s | "
            f"{res['slowdown_factor']:<6.1f}x"
        )
    print("=" * 80)
    print(f"Machine-readable results saved to: {output_json}")

    return ablation_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTA Ablation Study on Holdout Test Set")
    parser.add_argument("--model", type=str, default="deeplabv3")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_deeplab.pth")
    parser.add_argument("--test_json", type=str, default="data/splits/test.json")
    parser.add_argument("--output_json", type=str, default="outputs/tta_ablation.json")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_tta_ablation(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        test_json=args.test_json,
        output_json=args.output_json,
        device_name=args.device,
    )
