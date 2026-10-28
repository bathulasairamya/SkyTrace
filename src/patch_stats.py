"""Patch Sampling & Class Balance Statistics (Phase B9).

Audits 1,000 random patches drawn from the training partition to quantify:
1. Proportion of entirely empty patches (0% building pixels)
2. Mean and median building coverage fraction per patch
3. Class imbalance ratio (background pixels vs building foreground pixels)

Exports findings to outputs/patch_statistics.json.
"""

import os
import sys
import json
import random
import cv2
import numpy as np

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def compute_patch_balance_statistics(
    train_json: str = "data/splits/train.json",
    patch_size: int = 512,
    num_samples: int = 1000,
    seed: int = 42,
    output_json: str = "outputs/patch_statistics.json",
):
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    with open(train_json, "r", encoding="utf-8") as f:
        records = json.load(f)

    print("=" * 70)
    print("      COMPUTING CLASS BALANCE & PATCH SAMPLING STATISTICS (B9)      ")
    print("=" * 70)
    print(f"Sampling {num_samples} random {patch_size}x{patch_size} patches from {len(records)} train tiles...")

    # Load all training masks in memory
    masks = []
    for r in records:
        mask = cv2.imread(r["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            masks.append((mask > 127).astype(np.float32))

    if not masks:
        raise RuntimeError("No training masks found to sample from.")

    empty_count = 0
    fractions = []

    for _ in range(num_samples):
        # Pick random mask
        m = random.choice(masks)
        h, w = m.shape
        if h > patch_size and w > patch_size:
            y = random.randint(0, h - patch_size)
            x = random.randint(0, w - patch_size)
            patch = m[y : y + patch_size, x : x + patch_size]
        else:
            patch = cv2.resize(m, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)

        bldg_pixels = np.sum(patch > 0)
        frac = float(bldg_pixels / patch.size)
        fractions.append(frac)
        if bldg_pixels == 0:
            empty_count += 1

    empty_pct = round((empty_count / num_samples) * 100, 2)
    mean_frac = round(float(np.mean(fractions)) * 100, 2)
    median_frac = round(float(np.median(fractions)) * 100, 2)
    std_frac = round(float(np.std(fractions)) * 100, 2)

    stats = {
        "num_patches_sampled": num_samples,
        "patch_dimensions": f"{patch_size}x{patch_size}",
        "random_seed": seed,
        "empty_patches_count": empty_count,
        "empty_patches_pct": empty_pct,
        "mean_building_coverage_pct": mean_frac,
        "median_building_coverage_pct": median_frac,
        "std_building_coverage_pct": std_frac,
        "imbalance_ratio": f"{round((100 - mean_frac) / max(mean_frac, 0.01), 1)} : 1",
        "requires_foreground_sampling": empty_pct > 30.0,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nClass Balance Findings:")
    print(f"  Empty Patches:            {empty_count} / {num_samples} ({empty_pct}%)")
    print(f"  Mean Building Coverage:   {mean_frac}% (Median: {median_frac}%, Std: {std_frac}%)")
    print(f"  Background:Building Ratio: {stats['imbalance_ratio']}")
    print(f"  Foreground Sampling Req:  {stats['requires_foreground_sampling']} (Threshold: >30% empty)")
    print(f"Exported statistics to:     {output_json}")
    print("=" * 70)

    return stats


if __name__ == "__main__":
    compute_patch_balance_statistics()
