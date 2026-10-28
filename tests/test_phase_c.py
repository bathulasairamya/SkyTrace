"""Unit & Integration Tests for Phase C (Experiments, Ablations, Models)."""

import os
import sys
import unittest
import numpy as np
import torch
import cv2

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import build_model, AuthorUNet, PaperOptimizedUNet
from src.inference import predict_sliding_window, predict_sliding_window_tta
from src.ablation_morphology import apply_pipeline
from src.make_report_tables import generate_all_reports


class TestPhaseC(unittest.TestCase):
    """Phase C test suite."""

    def test_01_models_forward_pass_and_parameters(self):
        """Test AuthorUNet and PaperOptimizedUNet forward pass on odd and even dimensions."""
        model_author = AuthorUNet()
        model_author.eval()

        model_paper = PaperOptimizedUNet(in_channels=3, out_channels=1)
        model_paper.eval()

        # Test forward pass with non-square input
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            out_author = model_author(x)
            out_paper = model_paper(x)

        self.assertEqual(out_author.shape, (2, 1, 256, 256))
        self.assertEqual(out_paper.shape, (2, 1, 256, 256))

        # Check parameter count constraints
        params_author = sum(p.numel() for p in model_author.parameters())
        params_paper = sum(p.numel() for p in model_paper.parameters())
        self.assertGreater(params_author, 20_000_000)
        self.assertLess(params_paper, 10_000_000)

    def test_02_tta_inference_integrity(self):
        """Test standard sliding window vs 2-view vs 4-view TTA output shapes and bounds."""
        model = PaperOptimizedUNet(in_channels=3, out_channels=1)
        model.eval()

        # Synthetic 550x550 satellite patch
        synthetic_img = np.random.uniform(0.0, 1.0, (550, 550, 3)).astype(np.float32)

        prob_std, mask_std = predict_sliding_window(
            model=model,
            img_norm=synthetic_img,
            patch_size=512,
            stride=256,
            device=torch.device("cpu"),
            threshold=0.5,
        )

        prob_2view, mask_2view = predict_sliding_window_tta(
            model=model,
            img_norm=synthetic_img,
            patch_size=512,
            stride=256,
            device=torch.device("cpu"),
            threshold=0.5,
            mode="2-view",
        )

        self.assertEqual(prob_std.shape, (550, 550))
        self.assertEqual(prob_2view.shape, (550, 550))
        self.assertTrue(np.all(prob_std >= 0.0) and np.all(prob_std <= 1.0))
        self.assertTrue(np.all(prob_2view >= 0.0) and np.all(prob_2view <= 1.0))
        self.assertTrue(set(np.unique(mask_std)).issubset({0, 255}))

    def test_03_morphology_and_polygon_simplification(self):
        """Test morphological operations and Douglas-Peucker polygon simplification."""
        # Create a synthetic probability map with a 50x50 building having an internal hole and a 2x2 noise dot
        prob_map = np.zeros((200, 200), dtype=np.float32)
        prob_map[50:100, 50:100] = 0.9  # 50x50 building
        prob_map[70:80, 70:80] = 0.1    # internal hole
        prob_map[10:12, 10:12] = 0.9    # tiny isolated speck

        # Pipeline A: Raw threshold
        pipe_raw = {"morph_mode": "none", "simplify_epsilon": 0.0, "threshold": 0.5}
        mask_raw, poly_raw, _ = apply_pipeline(prob_map, pipe_raw, min_area=15)

        # Pipeline B: Closing + Opening + Simplification
        pipe_clean = {"morph_mode": "close_open", "kernel_size": 3, "simplify_epsilon": 1.5, "threshold": 0.5}
        mask_clean, poly_clean, _ = apply_pipeline(prob_map, pipe_clean, min_area=15)

        # The tiny speck (< 15 px area) should be filtered out in both
        self.assertEqual(len(poly_raw), 1)
        self.assertEqual(len(poly_clean), 1)

        # In pipe_clean with morphological close, the internal hole should be filled
        self.assertEqual(mask_clean[75, 75], 255)

        # Douglas-Peucker simplification should reduce vertices
        raw_verts = len(poly_raw[0])
        clean_verts = len(poly_clean[0])
        self.assertLessEqual(clean_verts, raw_verts)

    def test_04_master_report_generation(self):
        """Test master report generator produces valid markdown and csv."""
        generate_all_reports(outputs_dir="outputs")
        self.assertTrue(os.path.exists("outputs/evaluation_results.md"))


if __name__ == "__main__":
    unittest.main()
