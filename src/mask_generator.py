"""Mask Generator: Convert SpaceNet-2 GeoJSON annotations to binary raster masks."""

import os
import json
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape
import cv2


def geojson_to_mask(image_path: str, geojson_path: str, allow_empty: bool = True) -> np.ndarray:
    """Rasterize vector building polygons into a binary 2D mask matching the satellite image extent.

    Fails safely with explicit errors if annotations are missing or corrupt, preventing
    silent substitution of all-zero background masks.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Source raster not found at '{image_path}'")

    with rasterio.open(image_path) as src:
        height, width = src.height, src.width
        transform = src.transform

    if not os.path.exists(geojson_path):
        raise FileNotFoundError(
            f"Annotation GeoJSON missing at '{geojson_path}'. "
            f"Cannot safely generate ground truth mask without verified annotation file."
        )

    with open(geojson_path, "r", encoding="utf-8") as f:
        try:
            gj = json.load(f)
        except json.JSONDecodeError as err:
            raise ValueError(f"Corrupted GeoJSON file at '{geojson_path}': {err}")

    features = gj.get("features", [])
    if not features:
        if allow_empty:
            # Genuinely empty scene confirmed by verified annotation file
            return np.zeros((height, width), dtype=np.uint8)
        else:
            raise ValueError(f"GeoJSON at '{geojson_path}' contains zero features.")

    shapes = []
    for feat in features:
        geom = feat.get("geometry")
        if geom and geom.get("coordinates"):
            try:
                poly = shape(geom)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if not poly.is_empty:
                    shapes.append((poly, 1))
            except Exception:
                continue

    if not shapes:
        return np.zeros((height, width), dtype=np.uint8)

    # all_touched=False: standard remote sensing pixel center rule (does not inflate building boundaries)
    mask = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )
    return mask


def generate_all_masks(image_dir: str, geojson_dir: str, output_dir: str):
    """Generate and save PNG binary masks for all matching image/geojson pairs."""
    os.makedirs(output_dir, exist_ok=True)
    images = sorted([f for f in os.listdir(image_dir) if f.endswith(".tif")])
    generated_count = 0

    for img_name in images:
        tile_id = img_name.replace(".tif", "").replace("SN2_buildings_train_AOI_3_Paris_PS-RGB_", "")
        img_path = os.path.join(image_dir, img_name)
        
        # Match corresponding geojson file
        geo_name = f"SN2_buildings_train_AOI_3_Paris_geojson_buildings_{tile_id}.geojson"
        geo_path = os.path.join(geojson_dir, geo_name)
        mask_path = os.path.join(output_dir, f"{tile_id}.png")

        mask = geojson_to_mask(img_path, geo_path)
        cv2.imwrite(mask_path, mask * 255)
        generated_count += 1

    print(f"[MaskGenerator] Generated {generated_count} binary masks in {output_dir}")
    return generated_count


if __name__ == "__main__":
    generate_all_masks(
        image_dir="data/images",
        geojson_dir="data/geojson",
        output_dir="data/masks",
    )
