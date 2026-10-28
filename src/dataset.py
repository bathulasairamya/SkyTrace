"""PyTorch Dataset & Augmentations for Satellite Building Footprint Segmentation."""

import os
import json
import random
import numpy as np
import rasterio
import cv2
import torch
from torch.utils.data import Dataset, DataLoader


class SatelliteBuildingDataset(Dataset):
    """Satellite image dataset with patch extraction, min-max normalization, and spatial augmentations."""
    def __init__(
        self,
        split_json: str,
        patch_size: int = 512,
        is_train: bool = True,
        augment: bool = False,
        samples_per_tile: int = 4,
        seed: int = 42,
    ):
        with open(split_json, "r", encoding="utf-8") as f:
            self.records = json.load(f)

        self.patch_size = patch_size
        self.is_train = is_train
        self.augment = augment
        self.samples_per_tile = samples_per_tile
        self.seed = seed

    def __len__(self):
        return len(self.records) * self.samples_per_tile

    def _apply_augmentations(self, img: np.ndarray, mask: np.ndarray):
        """Random spatial transformations: horizontal flip, vertical flip, and 90/180/270 rotations."""
        # Random horizontal flip
        if random.random() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()

        # Random vertical flip
        if random.random() > 0.5:
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()

        # Random 90, 180, 270 rotation (Pasquali et al. 2019 Section 4.1)
        k = random.randint(0, 3)
        if k > 0:
            img = np.rot90(img, k=k).copy()
            mask = np.rot90(mask, k=k).copy()

        return img, mask

    def _get_deterministic_crop_origin(self, h: int, w: int, ps: int, sub_idx: int):
        """Compute reproducible, fixed crop coordinates for non-training evaluation."""
        if h <= ps or w <= ps:
            return 0, 0
        
        # Grid of fixed deterministic anchor points (corners and center)
        anchors = [
            (0, 0),                           # Top-Left
            (h - ps, w - ps),                 # Bottom-Right
            (0, w - ps),                     # Top-Right
            (h - ps, 0),                     # Bottom-Left
            ((h - ps) // 2, (w - ps) // 2),  # Center
        ]
        return anchors[sub_idx % len(anchors)]

    def __getitem__(self, idx):
        record_idx = idx // self.samples_per_tile
        sub_idx = idx % self.samples_per_tile
        record = self.records[record_idx]
        tile_id = record.get("tile_id", f"idx_{record_idx}")

        img_path = record["image_path"]
        mask_path = record["mask_path"]

        # 1. Verify existence of image and annotation mask
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Satellite image missing for tile {tile_id}: '{img_path}'")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(
                f"Annotation mask missing for tile {tile_id}: '{mask_path}'. "
                f"Missing annotations must raise explicit errors rather than fabricating all-zero background masks."
            )

        # 2. Read 3-band RGB satellite image
        with rasterio.open(img_path) as src:
            img = src.read([1, 2, 3]).transpose(1, 2, 0).astype("float32")

        # 3. Read binary mask with validation
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not decode binary mask image for tile {tile_id}: '{mask_path}'")

        # 4. Strict spatial dimension check
        if mask.shape[:2] != img.shape[:2]:
            raise ValueError(
                f"Spatial dimension mismatch for tile {tile_id}: image shape {img.shape[:2]} vs mask shape {mask.shape[:2]}"
            )

        # 5. Check mask content and binary validity
        mask_binary = (mask > 127).astype("float32")
        building_pixels = int(np.sum(mask_binary > 0))
        if building_pixels == 0:
            # Check whether scene is genuinely empty from record metadata
            if record.get("building_count", 0) > 0:
                raise ValueError(
                    f"Tile {tile_id} annotation has {record['building_count']} declared buildings but mask is completely empty!"
                )

        # 6. Min-Max Normalization: (img - min) / (max - min)
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)

        h, w, _ = img.shape
        ps = self.patch_size

        # 7. Extract patch: random during training, strictly deterministic during validation
        if h > ps and w > ps:
            if self.is_train:
                y = random.randint(0, h - ps)
                x = random.randint(0, w - ps)
            else:
                y, x = self._get_deterministic_crop_origin(h, w, ps, sub_idx)
            img = img[y : y + ps, x : x + ps]
            mask_binary = mask_binary[y : y + ps, x : x + ps]
        elif h != ps or w != ps:
            img = cv2.resize(img, (ps, ps), interpolation=cv2.INTER_LINEAR)
            mask_binary = cv2.resize(mask_binary, (ps, ps), interpolation=cv2.INTER_NEAREST)

        # 8. Apply augmentations if training
        if self.augment and self.is_train:
            img, mask_binary = self._apply_augmentations(img, mask_binary)

        # 9. Convert to PyTorch tensors: Image -> (3, H, W), Mask -> (1, H, W)
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        mask_tensor = torch.tensor(mask_binary, dtype=torch.float32).unsqueeze(0)

        return img_tensor, mask_tensor


def get_dataloaders(
    train_json: str = "data/splits/train.json",
    val_json: str = "data/splits/val.json",
    batch_size: int = 2,
    patch_size: int = 512,
    num_workers: int = 0,
    seed: int = 42,
):
    """Instantiate train and deterministic validation DataLoaders."""
    train_dataset = SatelliteBuildingDataset(
        split_json=train_json,
        patch_size=patch_size,
        is_train=True,
        augment=True,
        samples_per_tile=4,
        seed=seed,
    )
    val_dataset = SatelliteBuildingDataset(
        split_json=val_json,
        patch_size=patch_size,
        is_train=False,
        augment=False,
        samples_per_tile=2,
        seed=seed,
    )

    g = torch.Generator()
    g.manual_seed(seed)

    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(batch_size=2, patch_size=512)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    for imgs, masks in train_loader:
        print(f"Sample Batch -> Images: {imgs.shape}, Masks: {masks.shape}")
        print(f"Image value range: [{imgs.min():.2f}, {imgs.max():.2f}] | Mask unique: {torch.unique(masks).tolist()}")
        break
