"""SkyTrace Web Interface: Satellite Image Upload, Building Mask & Outline Visualizer."""

import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
import base64
import numpy as np
import cv2
import torch
from flask import Flask, render_template, request, jsonify
from PIL import Image

from src.models import build_model
from src.inference import (
    load_inference_image,
    predict_sliding_window,
    predict_sliding_window_tta,
    extract_contours_and_overlay,
    contours_to_geojson,
)

app = Flask(__name__, template_folder="templates", static_folder="static")

# Auto-detect device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache models in memory
MODELS = {}

def get_loaded_model(model_choice: str = "pretrained"):
    """Load and cache model."""
    if model_choice in MODELS:
        return MODELS[model_choice]

    if model_choice == "finetuned" and os.path.exists("checkpoints/best_finetuned_model.pth"):
        ckpt = "checkpoints/best_finetuned_model.pth"
    else:
        ckpt = "checkpoints/best_deeplab.pth"

    model = build_model("deeplabv3", checkpoint_path=ckpt, device=DEVICE)
    model.eval()
    MODELS[model_choice] = model
    return model


def array_to_base64_png(arr: np.ndarray, is_rgb: bool = True) -> str:
    """Convert numpy array (uint8) to base64 encoded PNG string."""
    if not is_rgb and len(arr.shape) == 2:
        pil_img = Image.fromarray(arr, mode="L")
    else:
        pil_img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


@app.route("/")
def index():
    # Find sample test images available on disk
    sample_images = []
    if os.path.exists("data/images"):
        tifs = sorted([f for f in os.listdir("data/images") if f.endswith(".tif")])[:8]
        sample_images = [f.replace("SN2_buildings_train_AOI_3_Paris_PS-RGB_", "").replace(".tif", "") for f in tifs]

    return render_template(
        "index.html",
        device_name=str(DEVICE).upper(),
        cuda_available=torch.cuda.is_available(),
        sample_images=sample_images,
        has_finetuned=os.path.exists("checkpoints/best_finetuned_model.pth"),
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        model_choice = request.form.get("model_choice", "pretrained")
        threshold = float(request.form.get("threshold", 0.5))
        use_tta = request.form.get("use_tta", "false").lower() in ["true", "1", "yes"]
        sample_id = request.form.get("sample_id", "")

        temp_path = "temp_upload.tif"

        if sample_id:
            # Load local sample tile
            img_filename = f"SN2_buildings_train_AOI_3_Paris_PS-RGB_{sample_id}.tif"
            src_path = os.path.join("data/images", img_filename)
            if not os.path.exists(src_path):
                return jsonify({"error": f"Sample tile {sample_id} not found"}), 404
            img_path = src_path
        elif "file" in request.files and request.files["file"].filename:
            f = request.files["file"]
            f.save(temp_path)
            img_path = temp_path
        else:
            return jsonify({"error": "No image file provided"}), 400

        # Run inference
        raw_rgb, norm_rgb = load_inference_image(img_path)
        model = get_loaded_model(model_choice)
        
        if use_tta:
            prob_map, binary_mask = predict_sliding_window_tta(
                model=model,
                img_norm=norm_rgb,
                patch_size=512,
                stride=256,
                device=DEVICE,
                threshold=threshold,
            )
        else:
            prob_map, binary_mask = predict_sliding_window(
                model=model,
                img_norm=norm_rgb,
                patch_size=512,
                stride=256,
                device=DEVICE,
                threshold=threshold,
            )

        # Extract contours with morphological filtering & Douglas-Peucker simplification
        overlay, contours = extract_contours_and_overlay(
            rgb_img=raw_rgb,
            binary_mask=binary_mask,
            contour_color=(0, 230, 255),  # Cyan
            min_area=15,
            simplify_epsilon=1.5,
        )

        # Generate GIS GeoJSON
        geojson_data = contours_to_geojson(contours, raw_rgb.shape[1], raw_rgb.shape[0])

        # Compute telemetry statistics
        building_count = len(contours)
        building_pixels = int(np.sum(binary_mask == 255))
        total_pixels = binary_mask.size
        building_coverage_pct = round((building_pixels / total_pixels) * 100, 2)
        # Approximate GSD for SpaceNet WorldView-3 is 0.3m -> 0.09 m^2 per pixel
        est_building_area_sqm = round(building_pixels * 0.09, 1)

        # Convert visualizations to base64
        rgb_disp = (raw_rgb - raw_rgb.min()) / (raw_rgb.max() - raw_rgb.min() + 1e-8)
        rgb_disp_uint8 = (np.clip(rgb_disp, 0, 1) * 255).astype(np.uint8)

        # Clean up temporary upload
        if os.path.exists(temp_path) and img_path == temp_path:
            os.remove(temp_path)

        return jsonify({
            "status": "success",
            "telemetry": {
                "building_count": building_count,
                "building_pixels": building_pixels,
                "coverage_pct": building_coverage_pct,
                "estimated_area_sqm": est_building_area_sqm,
                "dimensions": f"{raw_rgb.shape[1]} x {raw_rgb.shape[0]} px",
                "device_used": str(DEVICE).upper(),
                "threshold_used": threshold,
                "tta_enabled": use_tta,
                "max_confidence": round(float(prob_map.max()) * 100, 1),
            },
            "images": {
                "satellite_rgb": array_to_base64_png(rgb_disp_uint8, is_rgb=True),
                "binary_mask": array_to_base64_png(binary_mask, is_rgb=False),
                "contour_overlay": array_to_base64_png(overlay, is_rgb=True),
            },
            "geojson": geojson_data,
        })

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
