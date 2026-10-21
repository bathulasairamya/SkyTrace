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
        augment: bool = False,
        samples_per_tile: int = 4,
    ):
        with open(split_json, "r", encoding="utf-8") as f:
            self.records = json.load(f)

        self.patch_size = patch_size
        self.augment = augment
        self.samples_per_tile = samples_per_tile

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

    def __getitem__(self, idx):
        record_idx = idx // self.samples_per_tile
        record = self.records[record_idx]

        img_path = record["image_path"]
        mask_path = record["mask_path"]

        # Read 3-band RGB satellite image
        with rasterio.open(img_path) as src:
            img = src.read([1, 2, 3]).transpose(1, 2, 0).astype("float32")

        # Read binary mask (values: 0 or 255 -> normalize to 0 or 1)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        else:
            mask = (mask > 127).astype("float32")

        # Min-Max Normalization: (img - min) / (max - min)
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)

        h, w, _ = img.shape
        ps = self.patch_size

        # Extract random patch
        if h > ps and w > ps:
            y = random.randint(0, h - ps)
            x = random.randint(0, w - ps)
            img = img[y : y + ps, x : x + ps]
            mask = mask[y : y + ps, x : x + ps]
        elif h != ps or w != ps:
            # Resize if smaller
            img = cv2.resize(img, (ps, ps), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (ps, ps), interpolation=cv2.INTER_NEAREST)

        # Apply augmentations if training
        if self.augment:
            img, mask = self._apply_augmentations(img, mask)

        # Convert to PyTorch tensors: Image -> (3, H, W), Mask -> (1, H, W)
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return img_tensor, mask_tensor


def get_dataloaders(
    train_json: str = "data/splits/train.json",
    val_json: str = "data/splits/val.json",
    batch_size: int = 2,
    patch_size: int = 512,
    num_workers: int = 0,
):
    """Instantiate train and validation DataLoaders."""
    train_dataset = SatelliteBuildingDataset(
        split_json=train_json,
        patch_size=patch_size,
        augment=True,
        samples_per_tile=4,
    )
    val_dataset = SatelliteBuildingDataset(
        split_json=val_json,
        patch_size=patch_size,
        augment=False,
        samples_per_tile=2,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(batch_size=2, patch_size=512)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    for imgs, masks in train_loader:
        print(f"Sample Batch -> Images: {imgs.shape}, Masks: {masks.shape}")
        print(f"Image value range: [{imgs.min():.2f}, {imgs.max():.2f}] | Mask unique: {torch.unique(masks).tolist()}")
        break
