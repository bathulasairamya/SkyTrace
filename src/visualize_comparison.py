"""Generate Publication-Quality Visual Comparison Panel.

Compares:
1. SpaceNet High-Resolution Satellite RGB Image
2. Ground Truth Binary Mask (from GeoJSON)
3. Pretrained Baseline DeepLabV3 (ResNet-50) Prediction
4. Fine-Tuned DeepLabV3 Model Prediction
5. Detected Building Footprints Vector Contour Overlay
"""

import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.models import build_model
from src.inference import (
    load_inference_image,
    predict_sliding_window,
    predict_sliding_window_tta,
    extract_contours_and_overlay,
)


def generate_comparison_panel(
    sample_tile_id: str = "img100",
    baseline_ckpt: str = "checkpoints/best_deeplab.pth",
    finetuned_ckpt: str = "checkpoints/best_finetuned_model.pth",
    output_path: str = "outputs/comparison_panel.png",
    use_tta: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    img_filename = f"SN2_buildings_train_AOI_3_Paris_PS-RGB_{sample_tile_id}.tif"
    img_path = os.path.join("data/images", img_filename)
    mask_path = os.path.join("data/masks", f"{sample_tile_id}.png")

    if not os.path.exists(img_path):
        # Pick first available test tile
        available = [f for f in os.listdir("data/images") if f.endswith(".tif")]
        if not available:
            raise FileNotFoundError("No satellite images found in data/images/")
        img_path = os.path.join("data/images", available[0])
        sample_tile_id = available[0].replace("SN2_buildings_train_AOI_3_Paris_PS-RGB_", "").replace(".tif", "")
        mask_path = os.path.join("data/masks", f"{sample_tile_id}.png")

    print(f"Generating visual comparison for tile: {sample_tile_id}...")

    # 1. Load satellite image
    raw_rgb, norm_rgb, meta = load_inference_image(img_path)
    rgb_disp = (raw_rgb - raw_rgb.min()) / (raw_rgb.max() - raw_rgb.min() + 1e-8)
    rgb_disp = np.clip(rgb_disp, 0, 1)

    # 2. Load ground truth mask with strict verification
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Missing ground truth mask for tile {sample_tile_id}: '{mask_path}'")
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        raise FileNotFoundError(f"Corrupt ground truth mask for tile {sample_tile_id}: '{mask_path}'")

    # 3. Predict with Baseline
    print("Running baseline model inference...")
    base_model = build_model("deeplabv3", checkpoint_path=baseline_ckpt, device=device)
    if use_tta:
        _, base_binary = predict_sliding_window_tta(base_model, norm_rgb, device=device)
    else:
        _, base_binary = predict_sliding_window(base_model, norm_rgb, device=device)

    # 4. Predict with Fine-Tuned Model
    print("Running fine-tuned model inference...")
    ft_model = build_model("deeplabv3", checkpoint_path=finetuned_ckpt, device=device)
    if use_tta:
        ft_prob, ft_binary = predict_sliding_window_tta(ft_model, norm_rgb, device=device)
    else:
        ft_prob, ft_binary = predict_sliding_window(ft_model, norm_rgb, device=device)

    # 5. Extract contours with Douglas-Peucker polygon approximation
    overlay, contours = extract_contours_and_overlay(
        raw_rgb,
        ft_binary,
        contour_color=(0, 235, 255),
        simplify_epsilon=1.5,
    )

    # Create 5-panel figure
    fig, axes = plt.subplots(1, 5, figsize=(25, 5.5))

    axes[0].imshow(rgb_disp)
    axes[0].set_title("(a) Satellite RGB Tile", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(gt_mask, cmap="bone")
    axes[1].set_title("(b) Ground Truth Mask", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(base_binary, cmap="bone")
    axes[2].set_title("(c) Pretrained Baseline", fontsize=13, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(ft_binary, cmap="bone")
    axes[3].set_title("(d) Fine-Tuned Model", fontsize=13, fontweight="bold")
    axes[3].axis("off")

    axes[4].imshow(overlay)
    axes[4].set_title(f"(e) SkyTrace Vectors ({len(contours)} buildings)", fontsize=13, fontweight="bold")
    axes[4].axis("off")

    plt.suptitle(
        f"SkyTrace Building Extraction Benchmark — Tile: {sample_tile_id} (SpaceNet-2 Paris)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Successfully saved 5-panel comparison to: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_comparison_panel(
        sample_tile_id="img100",
        baseline_ckpt="checkpoints/best_deeplab.pth",
        finetuned_ckpt="checkpoints/best_finetuned_model.pth",
        output_path="outputs/comparison_panel.png",
        use_tta=True,
    )
