import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
import uuid
import base64
import logging
import numpy as np
import cv2
import torch
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

from src.models import build_model, compute_file_sha256
from src.inference import (
    load_inference_image,
    predict_sliding_window,
    predict_sliding_window_tta,
    extract_contours_and_overlay,
    contours_to_geojson,
)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload limit
logging.basicConfig(level=logging.INFO)

# Auto-detect device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache models in memory
MODELS = {}
CHECKPOINT_HASHES = {}

UPLOAD_TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp_uploads")
os.makedirs(UPLOAD_TEMP_DIR, exist_ok=True)


def get_loaded_model(model_choice: str = "pretrained"):
    """Load and cache model with explicit verification (no silent fallback)."""
    if model_choice in MODELS:
        return MODELS[model_choice], CHECKPOINT_HASHES[model_choice]

    if model_choice == "finetuned":
        ckpt = "checkpoints/best_finetuned_model.pth"
        if not os.path.exists(ckpt):
            raise FileNotFoundError("Fine-tuned model checkpoint 'checkpoints/best_finetuned_model.pth' does not exist yet.")
    else:
        ckpt = "checkpoints/best_deeplab.pth"
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Baseline checkpoint '{ckpt}' not found on disk.")

    model = build_model("deeplabv3", checkpoint_path=ckpt, device=DEVICE)
    model.eval()
    sha256 = compute_file_sha256(ckpt)

    MODELS[model_choice] = model
    CHECKPOINT_HASHES[model_choice] = sha256
    return model, sha256


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
    created_temp_path = None
    try:
        model_choice = request.form.get("model_choice", "pretrained")
        threshold = float(request.form.get("threshold", 0.5))
        use_tta = request.form.get("use_tta", "false").lower() in ["true", "1", "yes"]
        sample_id = request.form.get("sample_id", "").strip()

        if sample_id:
            # Load local sample tile
            img_filename = f"SN2_buildings_train_AOI_3_Paris_PS-RGB_{sample_id}.tif"
            src_path = os.path.join("data/images", img_filename)
            if not os.path.exists(src_path):
                return jsonify({"status": "error", "message": f"Sample tile {sample_id} not found"}), 404
            img_path = src_path
        elif "file" in request.files and request.files["file"].filename:
            f = request.files["file"]
            filename = secure_filename(f.filename)
            ext = os.path.splitext(filename)[1].lower()
            if ext not in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
                return jsonify({"status": "error", "message": f"Unsupported file extension: {ext}"}), 400

            unique_id = uuid.uuid4().hex
            temp_path = os.path.join(UPLOAD_TEMP_DIR, f"upload_{unique_id}{ext}")
            f.save(temp_path)
            created_temp_path = temp_path
            img_path = temp_path
        else:
            return jsonify({"status": "error", "message": "No satellite image file provided"}), 400

        # Load image and model with geospatial metadata
        raw_rgb, norm_rgb, meta = load_inference_image(img_path)
        model, ckpt_sha = get_loaded_model(model_choice)

        # Run inference
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

        # Extract contours with courtyard hole topology & Douglas-Peucker simplification
        overlay, contours = extract_contours_and_overlay(
            rgb_img=raw_rgb,
            binary_mask=binary_mask,
            contour_color=(0, 230, 255),  # Cyan
            courtyard_color=(255, 140, 0),  # Amber
            min_area=15,
            min_hole_area=8,
            simplify_epsilon=1.5,
        )

        # Generate GIS GeoJSON with real geodetic coordinates and ellipsoidal area
        geojson_data = contours_to_geojson(contours, raw_rgb.shape[1], raw_rgb.shape[0], meta=meta)

        # Compute telemetry statistics
        building_count = len(contours)
        building_pixels = int(np.sum(binary_mask == 255))
        total_pixels = binary_mask.size
        building_coverage_pct = round((building_pixels / total_pixels) * 100, 2)
        
        # Rigorous area telemetry: geodetic ellipsoidal sqm if georeferenced, otherwise explicit pixel count
        est_building_area_sqm = geojson_data["metadata"]["total_building_area_sqm"]
        is_geo = geojson_data["metadata"]["is_georeferenced"]

        # Convert visualizations to base64
        rgb_disp = (raw_rgb - raw_rgb.min()) / (raw_rgb.max() - raw_rgb.min() + 1e-8)
        rgb_disp_uint8 = (np.clip(rgb_disp, 0, 1) * 255).astype(np.uint8)

        # Compute meaningful probability telemetry (mean confidence on predicted buildings)
        bldg_mask_bool = binary_mask > 127
        mean_bldg_prob = float(prob_map[bldg_mask_bool].mean()) if bldg_mask_bool.any() else 0.0
        max_prob = float(np.max(prob_map)) if prob_map.size > 0 else 0.0

        return jsonify({
            "status": "success",
            "telemetry": {
                "building_count": building_count,
                "building_pixels": building_pixels,
                "coverage_pct": building_coverage_pct,
                "estimated_area_sqm": est_building_area_sqm,
                "area_display": f"{est_building_area_sqm:,.1f} m²" if is_geo else f"{building_pixels:,} px (unreferenced)",
                "is_georeferenced": is_geo,
                "source_crs": meta.get("crs") if meta else None,
                "courtyards_detected": geojson_data["metadata"]["total_courtyards"],
                "dimensions": f"{raw_rgb.shape[1]} x {raw_rgb.shape[0]} px",
                "device_used": str(DEVICE).upper(),
                "threshold_used": threshold,
                "tta_enabled": use_tta,
                "model_name": "DeepLabV3-ResNet50",
                "checkpoint_sha256": ckpt_sha[:8],
                "mean_building_confidence": round(mean_bldg_prob * 100, 1),
                "max_confidence": round(max_prob * 100, 1),
            },
            "images": {
                "satellite_rgb": array_to_base64_png(rgb_disp_uint8, is_rgb=True),
                "binary_mask": array_to_base64_png(binary_mask, is_rgb=False),
                "contour_overlay": array_to_base64_png(overlay, is_rgb=True),
            },
            "geojson": geojson_data,
        })

    except Exception as err:
        app.logger.exception("Error processing inference request")
        return jsonify({
            "status": "error",
            "message": "Prediction failed during image processing. Verify the uploaded satellite tile format.",
        }), 500

    finally:
        # Guaranteed cleanup of uploaded temporary file
        if created_temp_path and os.path.exists(created_temp_path):
            try:
                os.remove(created_temp_path)
            except OSError:
                pass


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    app.run(host=host, port=port, debug=False)
