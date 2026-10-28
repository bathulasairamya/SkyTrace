import os
import sys

try:
    import pytest
except ImportError:
    class _PyTestMock:
        class raises:
            def __init__(self, expected_exc):
                self.expected_exc = expected_exc
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is None:
                    raise AssertionError(f"Expected exception {self.expected_exc.__name__} was not raised!")
                return issubclass(exc_type, self.expected_exc)
    pytest = _PyTestMock()

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import build_model, MODEL_REGISTRY, DeepLabV3ResNet50
from src.dataset import SatelliteBuildingDataset, get_dataloaders
from src.metrics import compute_metrics


def test_model_instantiation_and_registry():
    """Verify models instantiate and produce (B, 1, H, W) tensors."""
    assert "deeplabv3" in MODEL_REGISTRY
    assert "deeplabv3_resnet50" in MODEL_REGISTRY
    assert "author_unet" in MODEL_REGISTRY
    assert "paper_unet" in MODEL_REGISTRY

    m = build_model("deeplabv3", device="cpu")
    assert isinstance(m, DeepLabV3ResNet50)
    m.eval()
    x = torch.zeros(1, 3, 256, 256)
    with torch.no_grad():
        out = m(x)
    assert out.shape == (1, 1, 256, 256), f"Expected (1, 1, 256, 256), got {out.shape}"


def test_strict_checkpoint_loading_failure():
    """Verify that loading an incompatible checkpoint raises RuntimeError under strict mode."""
    # Attempting to load AuthorUNet weights into DeepLabV3
    with pytest.raises(RuntimeError):
        build_model("deeplabv3", checkpoint_path="checkpoints/best_unet.pth", device="cpu", allow_partial=False)


def test_missing_mask_raises_explicit_error(tmp_path):
    """Verify that a missing annotation mask raises FileNotFoundError rather than silently fabricating zeros."""
    fake_split = tmp_path / "fake_split.json"
    fake_split.write_text(
        '[{"tile_id": "test_01", "image_path": "data/images/SN2_buildings_train_AOI_3_Paris_PS-RGB_img136.tif", "mask_path": "non_existent_mask.png", "building_count": 10}]',
        encoding="utf-8"
    )

    ds = SatelliteBuildingDataset(split_json=str(fake_split), is_train=False)
    with pytest.raises(FileNotFoundError):
        _ = ds[0]


def test_deterministic_validation_reproducibility():
    """Verify that validation DataLoaders yield identical pixel arrays on successive passes."""
    _, val_loader1 = get_dataloaders(batch_size=2, seed=42)
    _, val_loader2 = get_dataloaders(batch_size=2, seed=42)

    sums1 = [m.sum().item() for _, m in val_loader1]
    sums2 = [m.sum().item() for _, m in val_loader2]

    assert len(sums1) > 0, "Validation loader yielded zero batches"
    assert sums1 == sums2, f"Validation batches non-deterministic! {sums1} vs {sums2}"


def test_metrics_empty_tile_convention():
    """Verify empty tile edge-case convention: if GT and Pred are both 0, IoU and Dice must be 1.0."""
    pred = torch.zeros(1, 1, 64, 64)
    target = torch.zeros(1, 1, 64, 64)
    m = compute_metrics(pred, target, threshold=0.5)
    assert m["iou"] == 1.0
    assert m["dice"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0


import unittest

class TestPhaseA(unittest.TestCase):
    """Phase A regression and architectural correctness test suite."""

    def test_01_model_instantiation_and_registry(self):
        test_model_instantiation_and_registry()

    def test_02_strict_checkpoint_loading_failure(self):
        test_strict_checkpoint_loading_failure()

    def test_03_missing_mask_raises_explicit_error(self):
        import tempfile
        import pathlib
        with tempfile.TemporaryDirectory() as td:
            test_missing_mask_raises_explicit_error(pathlib.Path(td))

    def test_04_deterministic_validation_reproducibility(self):
        test_deterministic_validation_reproducibility()

    def test_05_metrics_empty_tile_convention(self):
        test_metrics_empty_tile_convention()


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING PHASE A REGRESSION & CORRECTNESS SUITE")
    print("=" * 60)
    
    print("[1/5] Testing model registry and tensor shapes...")
    test_model_instantiation_and_registry()
    print("  -> Passed.")

    print("[2/5] Testing strict checkpoint failure on incompatible architecture...")
    test_strict_checkpoint_loading_failure()
    print("  -> Passed.")

    print("[3/5] Testing missing mask explicit error handling...")
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as td:
        test_missing_mask_raises_explicit_error(pathlib.Path(td))
    print("  -> Passed.")

    print("[4/5] Testing deterministic validation reproducibility...")
    test_deterministic_validation_reproducibility()
    print("  -> Passed.")

    print("[5/5] Testing empty tile convention in compute_metrics...")
    test_metrics_empty_tile_convention()
    print("  -> Passed.")

    print("=" * 60)
    print("ALL 5 PHASE A TESTS PASSED PERFECTLY!")
    print("=" * 60)
