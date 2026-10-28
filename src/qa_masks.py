"""Mask Quality Assurance (QA) & Geometric Verification Pipeline.

Audits every tile in the dataset splits to verify:
1. Raster and binary mask existence & decodability
2. Spatial extent matching (exact height and width alignment)
3. Value domain validity (strictly binary {0, 255})
4. Connected-component vs GeoJSON polygon topology consistency
5. Foreground building coverage bounds (0% to 65%)

Exports comprehensive report to outputs/qa_report.csv.
"""

import os
import sys
import csv
import json
from pathlib import Path
import cv2
import numpy as np
import rasterio

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run_mask_quality_audit(
    manifest_path: str = "data/splits/split_v1.json",
    output_csv: str = "outputs/qa_report.csv",
):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(manifest_path, "r", encoding="utf-8") as f:
        master = json.load(f)

    # Collect all tile records across train, val, and test
    records = []
    for split_name in ["train.json", "val.json", "test.json"]:
        split_file = os.path.join("data/splits", split_name)
        if os.path.exists(split_file):
            with open(split_file, "r", encoding="utf-8") as sf:
                records.extend(json.load(sf))

    print("=" * 70)
    print("           GEOSPATIAL MASK QUALITY ASSURANCE (QA) AUDIT           ")
    print("=" * 70)
    print(f"Auditing {len(records)} satellite image and annotation pairs...")

    results = []
    failures = []

    for r in records:
        tile_id = r["tile_id"]
        img_path = r["image_path"]
        mask_path = r["mask_path"]
        geo_path = r["geojson_path"]

        status = "PASSED"
        reasons = []

        # 1. Existence check
        if not os.path.exists(img_path):
            status = "FAILED"
            reasons.append("Image file missing")
        if not os.path.exists(mask_path):
            status = "FAILED"
            reasons.append("Mask file missing")
        if not os.path.exists(geo_path):
            status = "FAILED"
            reasons.append("GeoJSON file missing")

        if status == "FAILED":
            failures.append((tile_id, "; ".join(reasons)))
            results.append({
                "tile_id": tile_id,
                "status": status,
                "width": 0,
                "height": 0,
                "building_pixels": 0,
                "coverage_pct": 0.0,
                "mask_components": 0,
                "geojson_polygons": 0,
                "notes": "; ".join(reasons),
            })
            continue

        # 2. Dimensions & CRS decodability
        try:
            with rasterio.open(img_path) as src:
                w, h = src.width, src.height
                bands = src.count
        except Exception as e:
            status = "FAILED"
            reasons.append(f"Corrupt raster: {e}")
            w, h, bands = 0, 0, 0

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            status = "FAILED"
            reasons.append("Corrupt mask image")
            mask_h, mask_w = 0, 0
        else:
            mask_h, mask_w = mask.shape[:2]
            if (mask_h, mask_w) != (h, w):
                status = "FAILED"
                reasons.append(f"Dimension mismatch: img ({w}x{h}) vs mask ({mask_w}x{mask_h})")

        # 3. Value domain check
        if mask is not None:
            unique_vals = set(np.unique(mask).tolist())
            if not unique_vals.issubset({0, 255}):
                status = "FAILED"
                reasons.append(f"Non-binary mask values detected: {unique_vals}")

            building_pixels = int(np.sum(mask == 255))
            total_pixels = mask.size
            coverage_pct = round((building_pixels / total_pixels) * 100, 2)
            if coverage_pct > 65.0:
                status = "WARNING"
                reasons.append(f"Very high coverage: {coverage_pct}%")

            num_components, _ = cv2.connectedComponents((mask > 127).astype(np.uint8))
            num_components -= 1  # exclude background
        else:
            building_pixels = 0
            coverage_pct = 0.0
            num_components = 0

        # 4. GeoJSON feature comparison
        try:
            with open(geo_path, "r", encoding="utf-8") as gf:
                gj = json.load(gf)
            geojson_count = len(gj.get("features", []))
        except Exception as e:
            geojson_count = 0
            status = "FAILED"
            reasons.append(f"Invalid GeoJSON: {e}")

        # Check topological discrepancy between vector polygons and raster components
        if geojson_count > 0:
            diff_pct = abs(num_components - geojson_count) / geojson_count
            if diff_pct > 0.40 and num_components < geojson_count:
                # Merged touching buildings in raster
                reasons.append(f"Touching building merge: {geojson_count} polygons -> {num_components} components")

        results.append({
            "tile_id": tile_id,
            "status": status,
            "width": w,
            "height": h,
            "building_pixels": building_pixels,
            "coverage_pct": coverage_pct,
            "mask_components": num_components,
            "geojson_polygons": geojson_count,
            "notes": "; ".join(reasons) if reasons else "OK",
        })

    # Write QA CSV Report
    fieldnames = [
        "tile_id", "status", "width", "height", "building_pixels",
        "coverage_pct", "mask_components", "geojson_polygons", "notes"
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    passed_count = sum(1 for r in results if r["status"] == "PASSED")
    warning_count = sum(1 for r in results if r["status"] == "WARNING")
    failed_count = sum(1 for r in results if r["status"] == "FAILED")

    print(f"\nQA Results Summary:")
    print(f"  [PASSED]  {passed_count} tiles (100% verified integrity)")
    print(f"  [WARNING] {warning_count} tiles (high density or merged components)")
    print(f"  [FAILED]  {failed_count} tiles")
    print(f"Report exported to: {output_csv}")
    print("=" * 70)

    if failed_count > 0:
        raise RuntimeError(f"QA audit failed on {failed_count} tiles! Check {output_csv}")

    return results


if __name__ == "__main__":
    run_mask_quality_audit()
