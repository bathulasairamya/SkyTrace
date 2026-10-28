"""Dataset Splitting, Spatial Neighborhood Adjacency Audit & Data Leakage Prevention."""

import os
import sys
from pathlib import Path

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import json
import hashlib
import datetime
import cv2
import numpy as np
import rasterio
from shapely.geometry import box
from sklearn.model_selection import train_test_split


def load_master_author_split(solutions_csv: str = "data/solutions.csv"):
    """Reconstruct author's original 80/20 train/validation split across all 1148 Paris tiles."""
    with open(solutions_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_tiles = sorted(list(set(row["ImageId"] for row in reader)))

    author_train, author_holdout = train_test_split(all_tiles, test_size=0.2, random_state=42)
    return set(author_train), set(author_holdout)


def get_tile_stats_and_bounds(image_path: str, mask_path: str, geojson_path: str):
    """Compute building pixels, polygon count, density %, and geospatial bounding box."""
    # Read bounding box from GeoTIFF
    with rasterio.open(image_path) as src:
        bounds = src.bounds
        crs = str(src.crs)
        bbox = box(*bounds)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        building_pixels = 0
        density_pct = 0.0
    else:
        building_pixels = int(np.sum(mask == 255))
        density_pct = round((building_pixels / mask.size) * 100, 2)

    poly_count = 0
    if os.path.exists(geojson_path):
        try:
            with open(geojson_path, "r", encoding="utf-8") as f:
                gj = json.load(f)
                poly_count = len(gj.get("features", []))
        except Exception:
            poly_count = 0

    return building_pixels, poly_count, density_pct, bounds, bbox, crs


def compute_manifest_sha256(data_list) -> str:
    """Calculate deterministic SHA-256 hash of a partition list."""
    serialized = json.dumps(data_list, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def create_reproducible_splits(
    image_dir: str = "data/images",
    geojson_dir: str = "data/geojson",
    mask_dir: str = "data/masks",
    output_dir: str = "data/splits",
    val_ratio: float = 0.25,  # 5 or 6 val from 21 pool
    random_state: int = 42,
):
    """Partition dataset into verified train, val, and untouched test splits with spatial buffer audit."""
    os.makedirs(output_dir, exist_ok=True)
    author_train_set, author_holdout_set = load_master_author_split()

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".tif")])
    train_pool_tiles = []
    test_pool_tiles = []

    tile_boxes = {}

    for img_name in image_files:
        tile_suffix = img_name.replace(".tif", "").replace("SN2_buildings_train_AOI_3_Paris_PS-RGB_", "")
        full_id = f"AOI_3_Paris_{tile_suffix}"
        geo_name = f"SN2_buildings_train_AOI_3_Paris_geojson_buildings_{tile_suffix}.geojson"
        mask_name = f"{tile_suffix}.png"

        img_path = Path(image_dir, img_name).as_posix()
        geo_path = Path(geojson_dir, geo_name).as_posix()
        mask_path = Path(mask_dir, mask_name).as_posix()

        b_pix, b_count, density, bounds, bbox, crs = get_tile_stats_and_bounds(img_path, mask_path, geo_path)
        tile_boxes[tile_suffix] = bbox

        record = {
            "tile_id": tile_suffix,
            "full_id": full_id,
            "image_path": img_path,
            "geojson_path": geo_path,
            "mask_path": mask_path,
            "building_count": b_count,
            "building_pixels": b_pix,
            "density_pct": density,
            "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
            "crs": crs,
        }

        if full_id in author_holdout_set:
            record["leakage_audit"] = "VERIFIED_UNTOUCHED_HOLDOUT"
            test_pool_tiles.append(record)
        elif full_id in author_train_set:
            record["leakage_audit"] = "ORIGINAL_TRAIN_POOL"
            train_pool_tiles.append(record)
        else:
            raise ValueError(f"Unknown tile ID {full_id} not found in master solution set!")

    # Stratified Split of Train Pool by Building Density using pure NumPy
    densities = [r["density_pct"] for r in train_pool_tiles]
    q33, q67 = np.quantile(densities, [0.33, 0.67])
    density_bins = [0 if d <= q33 else (1 if d <= q67 else 2) for d in densities]

    train_split, val_split = train_test_split(
        train_pool_tiles,
        test_size=val_ratio,
        random_state=random_state,
        stratify=density_bins,
    )
    test_split = test_pool_tiles

    # --- Spatial Neighborhood Adjacency Audit (B3) ---
    adjacent_pairs = []
    for tr in train_split:
        tr_id = tr["tile_id"]
        tr_box = tile_boxes[tr_id]
        for te in test_split:
            te_id = te["tile_id"]
            te_box = tile_boxes[te_id]
            if tr_box.intersects(te_box) or tr_box.touches(te_box):
                adjacent_pairs.append({
                    "train_tile": tr_id,
                    "test_tile": te_id,
                    "relation": "touches_or_intersects",
                })

    # --- Strict Mathematical Disjointness Assertions ---
    train_ids = set(r["full_id"] for r in train_split)
    val_ids = set(r["full_id"] for r in val_split)
    test_ids = set(r["full_id"] for r in test_split)

    assert train_ids.isdisjoint(test_ids), "CRITICAL: Train and Test partitions overlap!"
    assert val_ids.isdisjoint(test_ids), "CRITICAL: Validation and Test partitions overlap!"
    assert train_ids.isdisjoint(val_ids), "CRITICAL: Train and Validation partitions overlap!"

    # Save individual split manifests with POSIX paths
    train_path = Path(output_dir, "train.json").as_posix()
    val_path = Path(output_dir, "val.json").as_posix()
    test_path = Path(output_dir, "test.json").as_posix()

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_split, f, indent=2)

    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_split, f, indent=2)

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_split, f, indent=2)

    # Master Consolidated Manifest: split_v1.json (B4)
    master_manifest = {
        "manifest_version": "v1.0",
        "created_at": datetime.datetime.now().isoformat(),
        "random_seed": random_state,
        "audit_status": "LOCAL_SPLITS_DISJOINT",
        "pretrained_checkpoint_independence": "unverified_limitation",
        "partition_counts": {
            "train": len(train_split),
            "val": len(val_split),
            "test": len(test_split),
            "total": len(image_files),
        },
        "density_stratification": {
            "train_mean_density_pct": round(float(np.mean([r["density_pct"] for r in train_split])), 2),
            "val_mean_density_pct": round(float(np.mean([r["density_pct"] for r in val_split])), 2),
            "test_mean_density_pct": round(float(np.mean([r["density_pct"] for r in test_split])), 2),
        },
        "spatial_adjacency_audit": {
            "adjacent_pairs_count": len(adjacent_pairs),
            "adjacent_pairs": adjacent_pairs,
            "methodology_note": (
                "Identified bounding-box adjacency between training tiles and holdout test tiles. "
                "Documented transparently to establish spatial awareness and edge-building leakage control."
            ),
        },
        "manifest_sha256": {
            "train": compute_manifest_sha256(train_split),
            "val": compute_manifest_sha256(val_split),
            "test": compute_manifest_sha256(test_split),
        },
        "train_tile_ids": sorted([r["tile_id"] for r in train_split]),
        "val_tile_ids": sorted([r["tile_id"] for r in val_split]),
        "test_tile_ids": sorted([r["tile_id"] for r in test_split]),
    }

    v1_path = Path(output_dir, "split_v1.json").as_posix()
    with open(v1_path, "w", encoding="utf-8") as f:
        json.dump(master_manifest, f, indent=2)

    # Update summary report
    summary = {
        "status": "LOCAL_SPLITS_DISJOINT",
        "local_splits_disjoint": True,
        "pretrained_checkpoint_independence": "unverified_limitation",
        "audit_notes": (
            "Local train, val, and test partitions share strictly 0 overlapping tile IDs. "
            "Pretrained Hugging Face checkpoint training data independence on Paris holdout tiles "
            "cannot be mathematically guaranteed and is recorded as a documented limitation."
        ),
        "random_state": random_state,
        "counts": master_manifest["partition_counts"],
        "building_counts": {
            "train": sum(r["building_count"] for r in train_split),
            "val": sum(r["building_count"] for r in val_split),
            "test": sum(r["building_count"] for r in test_split),
            "total": sum(r["building_count"] for r in train_split + val_split + test_split),
        },
        "building_pixels": {
            "train": sum(r["building_pixels"] for r in train_split),
            "val": sum(r["building_pixels"] for r in val_split),
            "test": sum(r["building_pixels"] for r in test_split),
            "total": sum(r["building_pixels"] for r in train_split + val_split + test_split),
        },
        "spatial_adjacency": {
            "adjacent_pairs": len(adjacent_pairs),
            "pairs": adjacent_pairs,
        },
        "tile_ids": {
            "train": sorted(list(train_ids)),
            "val": sorted(list(val_ids)),
            "test": sorted(list(test_ids)),
        },
    }

    with open(Path(output_dir, "leakage_audit_report.json").as_posix(), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print("           DATASET PARTITION & DATA LEAKAGE AUDIT REPORT           ")
    print("=" * 70)
    print(f"Train Split:      {len(train_split):2d} tiles | {summary['building_counts']['train']:4d} buildings | Mean Density: {master_manifest['density_stratification']['train_mean_density_pct']}%")
    print(f"Validation Split: {len(val_split):2d} tiles | {summary['building_counts']['val']:4d} buildings | Mean Density: {master_manifest['density_stratification']['val_mean_density_pct']}%")
    print(f"Test Split:       {len(test_split):2d} tiles | {summary['building_counts']['test']:4d} buildings | Mean Density: {master_manifest['density_stratification']['test_mean_density_pct']}%")
    print("-" * 70)
    print(f"Total:            {summary['counts']['total']:2d} tiles | {summary['building_counts']['total']:4d} buildings | {summary['building_pixels']['total']:,} building pixels")
    print("\nLEAKAGE & SPATIAL VERIFICATION AUDIT:")
    print("  [x] Train vs Test ID overlap:     0 (Strictly Disjoint)")
    print("  [x] Val vs Test ID overlap:       0 (Strictly Disjoint)")
    print("  [x] Train vs Val ID overlap:      0 (Strictly Disjoint)")
    print(f"  [x] Spatial Adjacency Pairs:      {len(adjacent_pairs)} documented boundary touches")
    print("  [x] Master Manifest:              data/splits/split_v1.json (SHA-256 hashed)")
    print("  [x] Audit Status:                 LOCAL_SPLITS_DISJOINT: VERIFIED")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    create_reproducible_splits()

