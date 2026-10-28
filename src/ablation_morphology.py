"""Morphological Post-Processing & Vector Simplification Ablation Study.

Evaluates how morphological filtering and Douglas-Peucker polygon simplification
impact segmentation fidelity, noise suppression, and vector file complexity.

Configurations tested on the 10 holdout test tiles:
1. `raw_thresh`: Raw binary thresholding (P > 0.5)
2. `closing_3x3`: Morphological Closing (3x3 kernel) to bridge rooftop gaps
3. `closing_opening_3x3`: Closing + Opening (3x3 kernel) for speckle noise suppression
4. `simplify_eps_0_5`: Closing + Opening + Douglas-Peucker (epsilon = 0.5 px)
5. `simplify_eps_1_5`: Closing + Opening + Douglas-Peucker (epsilon = 1.5 px - default)
6. `simplify_eps_3_0`: Closing + Opening + Douglas-Peucker (epsilon = 3.0 px - aggressive)

Outputs:
- outputs/morphology_ablation.json
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import cv2
import torch
from tqdm import tqdm

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import build_model
from src.inference import load_inference_image, predict_sliding_window
from src.metrics import compute_metrics
from src.evaluate import evaluate_connected_component_object_f1


def apply_pipeline(prob_map: np.ndarray, config: dict, min_area: int = 15):
    """Execute a specific post-processing pipeline on probability map.
    
    Returns:
        processed_mask (np.ndarray uint8): 0 or 255 binary raster mask
        polygons (list of np.ndarray): list of Nx1x2 or Nx2 contour coordinates
        timing_ms (float): duration of post-processing in milliseconds
    """
    t0 = time.perf_counter()
    thresh = config.get("threshold", 0.5)
    binary = (prob_map > thresh).astype(np.uint8) * 255

    # 1. Morphological Operations
    morph_mode = config.get("morph_mode", "none")
    k_size = config.get("kernel_size", 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))

    if morph_mode == "close":
        clean_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    elif morph_mode == "close_open":
        clean_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)
    else:
        clean_mask = binary.copy()

    # 2. Contour Extraction
    raw_contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 3. Filtering & Simplification
    simplify_eps = config.get("simplify_epsilon", 0.0)
    polygons = []

    for cnt in raw_contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        if simplify_eps > 0.0:
            simplified = cv2.approxPolyDP(cnt, simplify_eps, closed=True)
            if len(simplified) >= 3:
                polygons.append(simplified)
            else:
                polygons.append(cnt)
        else:
            polygons.append(cnt)

    # 4. Rasterize back to binary mask for pixel IoU calculation of simplified shapes
    rasterized_mask = np.zeros_like(clean_mask)
    if polygons:
        cv2.drawContours(rasterized_mask, polygons, -1, 255, thickness=cv2.FILLED)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return rasterized_mask, polygons, elapsed_ms


def run_morphology_ablation(
    model_name: str = "deeplabv3",
    checkpoint_path: str = "checkpoints/best_deeplab.pth",
    test_json: str = "data/splits/test.json",
    output_json: str = "outputs/morphology_ablation.json",
    device_name: str = "auto",
):
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else device_name
    )

    print("=" * 75)
    print("      STARTING MORPHOLOGICAL & VECTOR SIMPLIFICATION ABLATION        ")
    print(f"  Device:     {device.type.upper()}")
    print(f"  Model:      {model_name} ({checkpoint_path})")
    print(f"  Test Tiles: {test_json}")
    print("=" * 75)

    model = build_model(model_name=model_name, checkpoint_path=checkpoint_path, device=device)
    model.eval()

    with open(test_json, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    # Pre-compute probability maps for all 10 test tiles once to isolate post-processing
    print(f"\n[Step 1/2] Computing baseline probability maps for {len(test_records)} test tiles...")
    cached_prob_maps = {}
    ground_truths = {}

    for record in tqdm(test_records, desc="Inference"):
        tile_id = record["tile_id"]
        raw_rgb, norm_rgb, meta = load_inference_image(record["image_path"])
        gt_mask = cv2.imread(record["mask_path"], cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            raise FileNotFoundError(f"Missing mask: {record['mask_path']}")

        prob_map, _ = predict_sliding_window(
            model=model,
            img_norm=norm_rgb,
            patch_size=512,
            stride=256,
            device=device,
            threshold=0.5,
        )
        cached_prob_maps[tile_id] = prob_map
        ground_truths[tile_id] = gt_mask

    # Define ablation variants
    pipelines = [
        {
            "id": "raw_thresh",
            "name": "Raw Threshold (P > 0.5)",
            "morph_mode": "none",
            "simplify_epsilon": 0.0,
        },
        {
            "id": "closing_3x3",
            "name": "Morph Closing (3x3)",
            "morph_mode": "close",
            "kernel_size": 3,
            "simplify_epsilon": 0.0,
        },
        {
            "id": "closing_opening_3x3",
            "name": "Closing + Opening (3x3)",
            "morph_mode": "close_open",
            "kernel_size": 3,
            "simplify_epsilon": 0.0,
        },
        {
            "id": "simplify_eps_0_5",
            "name": "Douglas-Peucker (eps=0.5 px)",
            "morph_mode": "close_open",
            "kernel_size": 3,
            "simplify_epsilon": 0.5,
        },
        {
            "id": "simplify_eps_1_5",
            "name": "Douglas-Peucker (eps=1.5 px, default)",
            "morph_mode": "close_open",
            "kernel_size": 3,
            "simplify_epsilon": 1.5,
        },
        {
            "id": "simplify_eps_3_0",
            "name": "Douglas-Peucker (eps=3.0 px, aggressive)",
            "morph_mode": "close_open",
            "kernel_size": 3,
            "simplify_epsilon": 3.0,
        },
    ]

    print(f"\n[Step 2/2] Running post-processing across {len(pipelines)} variants...")
    results = {}

    for pipe in pipelines:
        pipe_id = pipe["id"]
        pipe_name = pipe["name"]

        tile_records = []
        timings = []
        total_poly_counts = []
        total_vertex_counts = []

        for record in test_records:
            tile_id = record["tile_id"]
            prob_map = cached_prob_maps[tile_id]
            gt_mask = ground_truths[tile_id]

            proc_mask, polygons, elapsed_ms = apply_pipeline(prob_map, pipe, min_area=15)
            timings.append(elapsed_ms)

            poly_count = len(polygons)
            vertex_count = sum(len(p) for p in polygons)
            total_poly_counts.append(poly_count)
            total_vertex_counts.append(vertex_count)

            # Metrics
            prob_tensor = torch.tensor(proc_mask / 255.0).unsqueeze(0).unsqueeze(0)
            gt_tensor = torch.tensor(gt_mask / 255.0).unsqueeze(0).unsqueeze(0)
            pm = compute_metrics(prob_tensor, gt_tensor, threshold=0.5)
            om = evaluate_connected_component_object_f1(proc_mask, gt_mask, iou_thresh=0.5)

            tile_records.append({
                "tile_id": tile_id,
                "postproc_ms": round(elapsed_ms, 2),
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
                "poly_count": poly_count,
                "vertex_count": vertex_count,
            })

        tot_tp = sum(t["tp"] for t in tile_records)
        tot_fp = sum(t["fp"] for t in tile_records)
        tot_fn = sum(t["fn"] for t in tile_records)
        global_iou = tot_tp / (tot_tp + tot_fp + tot_fn + 1e-7)
        global_dice = (2 * tot_tp) / (2 * tot_tp + tot_fp + tot_fn + 1e-7)

        sum_polys = sum(total_poly_counts)
        sum_verts = sum(total_vertex_counts)
        mean_verts_per_poly = (sum_verts / sum_polys) if sum_polys > 0 else 0.0

        results[pipe_id] = {
            "name": pipe_name,
            "config": {
                "morph_mode": pipe.get("morph_mode"),
                "simplify_epsilon": pipe.get("simplify_epsilon"),
            },
            "avg_postproc_ms": round(float(np.mean(timings)), 2),
            "macro_metrics": {
                "mean_pixel_iou": float(np.mean([t["pixel_iou"] for t in tile_records])),
                "mean_pixel_dice": float(np.mean([t["pixel_dice"] for t in tile_records])),
                "mean_precision": float(np.mean([t["precision"] for t in tile_records])),
                "mean_recall": float(np.mean([t["recall"] for t in tile_records])),
                "mean_cc_f1_iou50": float(np.mean([t["cc_f1_iou50"] for t in tile_records])),
            },
            "global_micro_metrics": {
                "global_iou": float(global_iou),
                "global_dice": float(global_dice),
            },
            "vector_geometry": {
                "total_polygons": sum_polys,
                "total_vertices": sum_verts,
                "mean_vertices_per_polygon": round(mean_verts_per_poly, 2),
            },
            "per_tile": tile_records,
        }

    # Compute vertex reduction percentage relative to raw_thresh
    base_verts = results["raw_thresh"]["vector_geometry"]["total_vertices"]
    for pid, data in results.items():
        v = data["vector_geometry"]["total_vertices"]
        reduction_pct = round((1.0 - (v / max(base_verts, 1))) * 100.0, 1)
        data["vector_geometry"]["vertex_reduction_pct"] = reduction_pct

    # Export machine-readable results
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 90)
    print("             POST-PROCESSING & SIMPLIFICATION ABLATION RESULTS                 ")
    print("=" * 90)
    print(f"{'Variant':<35} | {'Macro IoU':<10} | {'Dice':<8} | {'CC F1':<8} | {'Polys':<6} | {'Verts/Poly':<10} | {'Vert Reduct':<11} | {'Time (ms)':<9}")
    print("-" * 90)
    for pid, res in results.items():
        m = res["macro_metrics"]
        g = res["vector_geometry"]
        print(
            f"{res['name']:<35} | "
            f"{m['mean_pixel_iou']:<10.4f} | "
            f"{m['mean_pixel_dice']:<8.4f} | "
            f"{m['mean_cc_f1_iou50']:<8.4f} | "
            f"{g['total_polygons']:<6d} | "
            f"{g['mean_vertices_per_polygon']:<10.1f} | "
            f"{g['vertex_reduction_pct']:<10.1f}% | "
            f"{res['avg_postproc_ms']:<8.1f}ms"
        )
    print("=" * 90)
    print(f"Machine-readable results saved to: {output_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Morphology & Simplification Ablation Study")
    parser.add_argument("--model", type=str, default="deeplabv3")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_deeplab.pth")
    parser.add_argument("--test_json", type=str, default="data/splits/test.json")
    parser.add_argument("--output_json", type=str, default="outputs/morphology_ablation.json")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_morphology_ablation(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        test_json=args.test_json,
        output_json=args.output_json,
        device_name=args.device,
    )
