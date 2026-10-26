"""Sliding-Window Inference & Building Footprint Outline Extractor."""

import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import rasterio
import cv2
import torch

from src.models import build_model


def load_inference_image(image_path: str):
    """Load satellite image supporting both GeoTIFF (.tif) and standard RGB (.png, .jpg)."""
    if image_path.lower().endswith((".tif", ".tiff")):
        with rasterio.open(image_path) as src:
            bands = min(3, src.count)
            img = src.read(list(range(1, bands + 1))).transpose(1, 2, 0).astype("float32")
            if bands == 1:
                img = np.repeat(img, 3, axis=2)
    else:
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise ValueError(f"Could not read image from {image_path}")
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype("float32")

    # Min-Max Normalization to [0, 1] matching pretrained training distribution
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img_norm = (img - img_min) / (img_max - img_min)
    else:
        img_norm = np.zeros_like(img)

    return img, img_norm


def predict_sliding_window(
    model: torch.nn.Module,
    img_norm: np.ndarray,
    patch_size: int = 512,
    stride: int = 256,
    device: torch.device = None,
    threshold: float = 0.5,
):
    """Run sliding-window inference with weighted blending for seamless tile prediction."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    h, w, c = img_norm.shape

    # Pad image if smaller than patch_size
    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)
    if pad_h > 0 or pad_w > 0:
        img_norm = np.pad(img_norm, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        padded_h, padded_w, _ = img_norm.shape
    else:
        padded_h, padded_w = h, w

    prob_map = np.zeros((padded_h, padded_w), dtype=np.float32)
    count_map = np.zeros((padded_h, padded_w), dtype=np.float32)

    # 2D Hanning/spline window for edge smoothing
    window = np.outer(np.hanning(patch_size), np.hanning(patch_size)).astype(np.float32)
    window = np.maximum(window, 1e-4)

    y_steps = range(0, padded_h - patch_size + 1, stride)
    x_steps = range(0, padded_w - patch_size + 1, stride)

    # Ensure last edges are covered
    y_coords = list(y_steps)
    if not y_coords or y_coords[-1] != (padded_h - patch_size):
        y_coords.append(padded_h - patch_size)

    x_coords = list(x_steps)
    if not x_coords or x_coords[-1] != (padded_w - patch_size):
        x_coords.append(padded_w - patch_size)

    def _predict_patch(patch_arr):
        """Forward pass with optional Test-Time Augmentation (TTA: flips + rot90)."""
        patch_t = torch.tensor(patch_arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        out = model(patch_t)
        prob = torch.sigmoid(out).squeeze().cpu().numpy()
        return prob

    with torch.no_grad():
        for y in y_coords:
            for x in x_coords:
                patch = img_norm[y : y + patch_size, x : x + patch_size]

                # Standard prediction
                prob = _predict_patch(patch)

                prob_map[y : y + patch_size, x : x + patch_size] += prob * window
                count_map[y : y + patch_size, x : x + patch_size] += window

    # Un-pad and normalize probabilities
    prob_map = prob_map[:h, :w]
    count_map = count_map[:h, :w]
    final_prob = prob_map / np.maximum(count_map, 1e-7)
    binary_mask = (final_prob > threshold).astype(np.uint8) * 255

    return final_prob, binary_mask


def predict_sliding_window_tta(
    model: torch.nn.Module,
    img_norm: np.ndarray,
    patch_size: int = 512,
    stride: int = 256,
    device: torch.device = None,
    threshold: float = 0.5,
):
    """Test-Time Augmentation (TTA) inference: blends normal, h-flip, v-flip, and rot90 predictions."""
    # 1. Base prediction
    prob_orig, _ = predict_sliding_window(model, img_norm, patch_size, stride, device, threshold)

    # 2. Horizontal flip
    img_h = np.fliplr(img_norm).copy()
    prob_h, _ = predict_sliding_window(model, img_h, patch_size, stride, device, threshold)
    prob_h = np.fliplr(prob_h)

    # 3. Vertical flip
    img_v = np.flipud(img_norm).copy()
    prob_v, _ = predict_sliding_window(model, img_v, patch_size, stride, device, threshold)
    prob_v = np.flipud(prob_v)

    # 4. Rotate 90
    img_rot = np.rot90(img_norm, k=1).copy()
    prob_rot, _ = predict_sliding_window(model, img_rot, patch_size, stride, device, threshold)
    prob_rot = np.rot90(prob_rot, k=-1)

    # Blend 4-way TTA predictions
    final_prob = (prob_orig + prob_h + prob_v + prob_rot) / 4.0
    binary_mask = (final_prob > threshold).astype(np.uint8) * 255

    return final_prob, binary_mask


def extract_contours_and_overlay(
    rgb_img: np.ndarray,
    binary_mask: np.ndarray,
    contour_color=(0, 230, 255),
    min_area: int = 15,
    simplify_epsilon: float = 1.5,
):
    """Overlay detected building footprint vector contours on the satellite RGB image.
    
    Includes morphological noise filtering and Douglas-Peucker polygon simplification
    for sharp, architectural building outlines.
    """
    rgb_disp = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
    rgb_disp = (np.clip(rgb_disp, 0, 1) * 255).astype(np.uint8).copy()

    # Morphological refinement: close holes, remove tiny isolated specks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)

    raw_contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []

    for cnt in raw_contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        # Douglas-Peucker polygon approximation for crisp geometric edges
        if simplify_epsilon > 0:
            epsilon = simplify_epsilon
            simplified = cv2.approxPolyDP(cnt, epsilon, closed=True)
            if len(simplified) >= 3:
                filtered_contours.append(simplified)
            else:
                filtered_contours.append(cnt)
        else:
            filtered_contours.append(cnt)

    # Draw outline overlay
    overlay = rgb_disp.copy()
    cv2.drawContours(overlay, filtered_contours, -1, contour_color, 2)

    return overlay, filtered_contours


def contours_to_geojson(contours, img_width: int, img_height: int) -> dict:
    """Serialize detected building contours into standard GeoJSON FeatureCollection."""
    features = []
    for idx, cnt in enumerate(contours):
        pts = cnt.squeeze()
        if len(pts.shape) != 2 or pts.shape[0] < 3:
            continue
        # Close polygon ring
        coords = [[float(p[0]), float(p[1])] for p in pts]
        if coords[0] != coords[-1]:
            coords.append(coords[0])

        area_px = float(cv2.contourArea(cnt))
        features.append({
            "type": "Feature",
            "id": idx + 1,
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords],
            },
            "properties": {
                "building_id": idx + 1,
                "area_pixels": area_px,
                "est_area_sqm": round(area_px * 0.09, 1),
            },
        })

    return {
        "type": "FeatureCollection",
        "metadata": {
            "generator": "SkyTrace AI Building Extraction",
            "total_buildings": len(features),
            "image_dimensions": [img_width, img_height],
        },
        "features": features,
    }

