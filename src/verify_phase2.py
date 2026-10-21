"""Verification Script for Phase 2: Dataset Acquisition & Mask Generation."""

import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import rasterio
import cv2
import numpy as np

def run_phase2_verification():
    print("=" * 65)
    print("           PHASE 2 COMPREHENSIVE VERIFICATION REPORT           ")
    print("=" * 65)

    errors = []

    # 1. Check Model Checkpoints
    print("\n[1/4] Verifying Pretrained Model Checkpoints...")
    checkpoints = {
        "best_deeplab.pth": (155 * 1024 * 1024, 175 * 1024 * 1024),
        "best_unet.pth": (115 * 1024 * 1024, 130 * 1024 * 1024),
    }
    for ckpt_name, (min_sz, max_sz) in checkpoints.items():
        ckpt_path = os.path.join("checkpoints", ckpt_name)
        if not os.path.exists(ckpt_path):
            errors.append(f"Missing checkpoint: {ckpt_path}")
            print(f"  FAILED: {ckpt_name} not found!")
        else:
            sz = os.path.getsize(ckpt_path)
            if min_sz <= sz <= max_sz:
                print(f"  PASS: {ckpt_name} exists ({sz / (1024 * 1024):.1f} MB)")
            else:
                errors.append(f"Unexpected file size for {ckpt_name}: {sz} bytes")
                print(f"  WARNING: {ckpt_name} size ({sz} bytes) outside expected bounds")

    # 2. Check Satellite Images
    print("\n[2/4] Verifying Satellite Imagery (PS-RGB)...")
    img_files = sorted([f for f in os.listdir("data/images") if f.endswith(".tif")])
    print(f"  Found {len(img_files)} satellite .tif images.")
    if len(img_files) < 10:
        errors.append(f"Expected at least 10 images, found {len(img_files)}")
    else:
        sample_img = os.path.join("data/images", img_files[0])
        with rasterio.open(sample_img) as src:
            h, w = src.height, src.width
            bands = src.count
            print(f"  PASS: Sample image {img_files[0]}: {bands} bands, dimensions {h}x{w}")
            if bands != 3 or h != 650 or w != 650:
                errors.append(f"Unexpected image dimensions: {bands} bands, {h}x{w}")

    # 3. Check GeoJSON Annotations & Binary Masks
    print("\n[3/4] Verifying GeoJSON Annotations & Rasterized Binary Masks...")
    geo_files = sorted([f for f in os.listdir("data/geojson") if f.endswith(".geojson")])
    mask_files = sorted([f for f in os.listdir("data/masks") if f.endswith(".png")])
    print(f"  Found {len(geo_files)} GeoJSON annotation files.")
    print(f"  Found {len(mask_files)} rasterized binary masks.")

    if len(mask_files) != len(img_files):
        errors.append(f"Mask count ({len(mask_files)}) does not match image count ({len(img_files)})")
    else:
        print(f"  PASS: 1:1 match between images and generated masks ({len(mask_files)} pairs)")

    # Test sample mask values
    sample_mask_path = os.path.join("data/masks", mask_files[0])
    mask = cv2.imread(sample_mask_path, cv2.IMREAD_GRAYSCALE)
    unique_vals = np.unique(mask)
    building_pixels = np.sum(mask == 255)
    total_pixels = mask.size
    coverage_pct = (building_pixels / total_pixels) * 100

    print(f"  PASS: Sample mask {mask_files[0]}: Unique pixel values: {list(unique_vals)}")
    print(f"  PASS: Building pixels: {building_pixels:,} ({coverage_pct:.2f}% building density)")
    if not (unique_vals.tolist() == [0, 255] or unique_vals.tolist() == [0]):
        errors.append(f"Mask contains non-binary pixel values: {unique_vals}")

    # 4. Check QA Output Artifact
    print("\n[4/4] Verifying Visual Quality Assurance Plot...")
    qa_path = "outputs/mask_validation_sample.png"
    if os.path.exists(qa_path) and os.path.getsize(qa_path) > 100 * 1024:
        print(f"  PASS: Visual QA plot verified ({os.path.getsize(qa_path) / (1024*1024):.2f} MB) at {qa_path}")
    else:
        errors.append(f"Missing or incomplete visual QA plot: {qa_path}")
        print(f"  FAILED: QA plot missing or too small")

    print("\n" + "=" * 65)
    if not errors:
        print("   ALL VERIFICATION CHECKS PASSED: PHASE 2 IS RUNNING PERFECTLY!   ")
    else:
        print(f"   VERIFICATION FAILED WITH {len(errors)} ERROR(S):")
        for err in errors:
            print(f"     - {err}")
    print("=" * 65)
    return len(errors) == 0

if __name__ == "__main__":
    success = run_phase2_verification()
    sys.exit(0 if success else 1)
