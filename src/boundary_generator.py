"""Boundary & Contact-Zone Mask Generator for Separating Adjacent Buildings.

Generates boundary edge targets (data/masks_boundary/<tile_id>.png) using:
1. Rasterized polygon perimeters with cv2.polylines (thickness=2)
2. Morphological gradient isolation: (dilate(mask) - erode(mask))

These boundary masks allow multi-task learning or watershed marker generation
to resolve touching building instance merging in dense urban environments.
"""

import os
import sys
import json
from pathlib import Path
import cv2
import numpy as np
import rasterio
from shapely.geometry import shape

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def generate_boundary_masks(
    geojson_dir: str = "data/geojson",
    image_dir: str = "data/images",
    mask_dir: str = "data/masks",
    output_dir: str = "data/masks_boundary",
    thickness: int = 2,
):
    os.makedirs(output_dir, exist_ok=True)
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])

    print("=" * 70)
    print("        GENERATING BOUNDARY & CONTACT TARGET MASKS (PHASE B8)        ")
    print("=" * 70)
    print(f"Generating boundary targets for {len(mask_files)} tiles...")

    generated = 0
    for mask_name in mask_files:
        tile_id = mask_name.replace(".png", "")
        mask_path = os.path.join(mask_dir, mask_name)
        img_name = f"SN2_buildings_train_AOI_3_Paris_PS-RGB_{tile_id}.tif"
        img_path = os.path.join(image_dir, img_name)
        geo_name = f"SN2_buildings_train_AOI_3_Paris_geojson_buildings_{tile_id}.geojson"
        geo_path = os.path.join(geojson_dir, geo_name)

        out_path = os.path.join(output_dir, f"{tile_id}.png")

        # Load building mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        h, w = mask.shape
        boundary_canvas = np.zeros((h, w), dtype=np.uint8)

        # 1. Primary method: Rasterize GeoJSON polygon perimeters if available
        if os.path.exists(geo_path) and os.path.exists(img_path):
            with rasterio.open(img_path) as src:
                transform = src.transform

            with open(geo_path, "r", encoding="utf-8") as f:
                gj = json.load(f)

            for feat in gj.get("features", []):
                geom = feat.get("geometry")
                if not geom:
                    continue
                poly = shape(geom)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_empty:
                    continue

                polygons = list(poly.geoms) if hasattr(poly, "geoms") else [poly]

                for p in polygons:
                    # Map geographic coordinates of exterior ring to pixel coordinates
                    ext_coords = list(p.exterior.coords)
                    pts = []
                    for pt in ext_coords:
                        lon, lat = pt[0], pt[1]
                        # Invert affine transform: map (lon, lat) -> (col, row)
                        col, row = ~transform * (lon, lat)
                        pts.append([int(round(col)), int(round(row))])

                    pts_arr = np.array([pts], dtype=np.int32)
                    cv2.polylines(boundary_canvas, pts_arr, isClosed=True, color=255, thickness=thickness)

                    # Process interior rings (courtyards)
                    for interior in p.interiors:
                        int_pts = []
                        for pt in list(interior.coords):
                            lon, lat = pt[0], pt[1]
                            col, row = ~transform * (lon, lat)
                            int_pts.append([int(round(col)), int(round(row))])
                        int_arr = np.array([int_pts], dtype=np.int32)
                        cv2.polylines(boundary_canvas, int_arr, isClosed=True, color=255, thickness=thickness)

        # 2. Fallback / enhancement: morphological gradient along building edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(mask, kernel, iterations=1)
        eroded = cv2.erode(mask, kernel, iterations=1)
        morph_gradient = cv2.subtract(dilated, eroded)

        combined_boundary = cv2.bitwise_or(boundary_canvas, morph_gradient)
        # Only retain boundary pixels that overlap or touch building footprint
        combined_boundary = cv2.bitwise_and(combined_boundary, dilated)

        cv2.imwrite(out_path, combined_boundary)
        generated += 1

    print(f"Successfully generated {generated} boundary masks in '{output_dir}'.")
    print("=" * 70)
    return generated


if __name__ == "__main__":
    generate_boundary_masks()
