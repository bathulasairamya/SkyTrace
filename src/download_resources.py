import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import urllib.request
import urllib.error
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import numpy as np
import rasterio

from src.mask_generator import generate_all_masks, geojson_to_mask

S3_BASE = "https://spacenet-dataset.s3.amazonaws.com/spacenet/SN2_buildings/train/AOI_3_Paris"
HF_BASE = "https://huggingface.co/Abhatta7/building-footprint-segmentation/resolve/main"

CHECKPOINTS = {
    "best_deeplab.pth": f"{HF_BASE}/best_deeplab.pth",
    "best_unet.pth": f"{HF_BASE}/best_unet.pth",
}


def download_file(url: str, dest_path: str, desc: str = None):
    """Download file with progress bar and resume/skip support."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"[Skip] Already exists: {os.path.basename(dest_path)} ({os.path.getsize(dest_path)/(1024*1024):.1f} MB)")
        return True

    temp_path = dest_path + ".tmp"
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as f:
            total_size = int(resp.headers.get("content-length", 0))
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc or os.path.basename(dest_path)) as pbar:
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
        os.replace(temp_path, dest_path)
        return True
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"[Error] Failed downloading {url}: {e}")
        return False


def get_tile_selection(solutions_csv: str, n_train: int = 21, n_test: int = 10):
    """Select high-density building tiles strictly partitioned to prevent data leakage."""
    image_counts = Counter()
    with open(solutions_csv, mode="r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("BuildingId") and row["BuildingId"] != "-1":
                image_counts[row["ImageId"]] += 1
            else:
                image_counts[row["ImageId"]] += 0

    all_tiles = sorted(list(image_counts.keys()))
    train_pool, test_pool = train_test_split(all_tiles, test_size=0.2, random_state=42)

    # Sort candidates by building richness
    sorted_test = sorted([t for t in test_pool if image_counts[t] >= 15], key=lambda t: image_counts[t], reverse=True)
    sorted_train = sorted([t for t in train_pool if image_counts[t] >= 15], key=lambda t: image_counts[t], reverse=True)

    selected_test = sorted_test[:n_test]
    selected_train = sorted_train[:n_train]

    print(f"[SplitAudit] Selected {len(selected_train)} train tiles and {len(selected_test)} test tiles.")
    print(f"[SplitAudit] Zero-Leakage check: {set(selected_train).isdisjoint(set(selected_test))}")
    return selected_train, selected_test


def download_dataset(solutions_csv: str = "data/solutions.csv", n_train: int = 21, n_test: int = 10):
    """Download satellite tiles and geojson annotations for selected train and test sets."""
    os.makedirs("data/images", exist_ok=True)
    os.makedirs("data/geojson", exist_ok=True)

    train_tiles, test_tiles = get_tile_selection(solutions_csv, n_train=n_train, n_test=n_test)
    selected_tiles = train_tiles + test_tiles

    print(f"\n--- Downloading {len(selected_tiles)} SpaceNet-2 Paris Tiles ---")
    for tile_full_id in selected_tiles:
        tile_id = tile_full_id.replace("AOI_3_Paris_", "")
        img_filename = f"SN2_buildings_train_AOI_3_Paris_PS-RGB_{tile_id}.tif"
        geo_filename = f"SN2_buildings_train_AOI_3_Paris_geojson_buildings_{tile_id}.geojson"

        img_url = f"{S3_BASE}/PS-RGB/{img_filename}"
        geo_url = f"{S3_BASE}/geojson_buildings/{geo_filename}"

        img_dest = os.path.join("data/images", img_filename)
        geo_dest = os.path.join("data/geojson", geo_filename)

        download_file(img_url, img_dest, desc=f"Img {tile_id}")
        download_file(geo_url, geo_dest, desc=f"Geo {tile_id}")

    return train_tiles, test_tiles


def create_validation_plot(sample_tile_id: str = "img100"):
    """Render a visual quality-assurance plot overlaying masks and contours on true-color RGB."""
    os.makedirs("outputs", exist_ok=True)
    img_path = f"data/images/SN2_buildings_train_AOI_3_Paris_PS-RGB_{sample_tile_id}.tif"
    geo_path = f"data/geojson/SN2_buildings_train_AOI_3_Paris_geojson_buildings_{sample_tile_id}.geojson"
    mask_path = f"data/masks/{sample_tile_id}.png"

    if not os.path.exists(img_path):
        available = [f for f in os.listdir("data/images") if f.endswith(".tif")]
        if not available:
            return
        img_path = os.path.join("data/images", available[0])
        sample_tile_id = available[0].replace(".tif", "").replace("SN2_buildings_train_AOI_3_Paris_PS-RGB_", "")
        geo_path = f"data/geojson/SN2_buildings_train_AOI_3_Paris_geojson_buildings_{sample_tile_id}.geojson"
        mask_path = f"data/masks/{sample_tile_id}.png"

    with rasterio.open(img_path) as src:
        rgb = src.read([1, 2, 3]).transpose(1, 2, 0).astype("float32")

    # Min-Max normalize RGB to [0, 1] for display
    rgb_disp = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    rgb_disp = np.clip(rgb_disp, 0, 1)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        mask = geojson_to_mask(img_path, geo_path) * 255

    # Create contour boundary overlay
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = (rgb_disp * 255).astype(np.uint8).copy()
    cv2.drawContours(overlay, contours, -1, (255, 60, 60), 2)  # Red contours

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(rgb_disp)
    axes[0].set_title(f"SpaceNet Satellite RGB ({sample_tile_id})", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Rasterized Ground Truth Mask", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Building Footprint Outlines Overlay", fontsize=13, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    out_file = "outputs/mask_validation_sample.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"[QA] Mask validation visual saved to {out_file}")


def main():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # 1. Download Pretrained Checkpoint (DeepLabV3+)
    print("\n--- Step 1: Downloading Pretrained Checkpoint (Abhatta7/building-footprint-segmentation) ---")
    download_file(CHECKPOINTS["best_deeplab.pth"], "checkpoints/best_deeplab.pth", desc="best_deeplab.pth")

    # 2. Download Selected SpaceNet-2 Paris Tiles
    print("\n--- Step 2: Downloading Curated SpaceNet-2 Paris Dataset ---")
    download_dataset(solutions_csv="data/solutions.csv", n_train=21, n_test=10)

    # 3. Generate Binary Masks
    print("\n--- Step 3: Generating Binary Raster Masks from GeoJSON ---")
    generate_all_masks("data/images", "data/geojson", "data/masks")

    # 4. Render Validation QA Plot
    print("\n--- Step 4: Generating Validation Quality Plot ---")
    create_validation_plot("img100")
    print("\nPhase 2 Dataset Acquisition & Mask Generation Complete!")


if __name__ == "__main__":
    main()
