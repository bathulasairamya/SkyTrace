"""Automated Verification & Regression Tests for Phase B Data, Splits & Auditing."""

import os
import sys
import json
import csv
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_split_manifest_and_disjointness():
    """Verify split_v1.json exists, contains all 31 tiles, and has zero ID overlap."""
    manifest_path = "data/splits/split_v1.json"
    assert os.path.exists(manifest_path), f"Missing {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    train_ids = set(m["train_tile_ids"])
    val_ids = set(m["val_tile_ids"])
    test_ids = set(m["test_tile_ids"])

    assert len(train_ids) == 15, f"Expected 15 train tiles, got {len(train_ids)}"
    assert len(val_ids) == 6, f"Expected 6 val tiles, got {len(val_ids)}"
    assert len(test_ids) == 10, f"Expected 10 test tiles, got {len(test_ids)}"

    assert train_ids.isdisjoint(test_ids), "Train and Test IDs overlap!"
    assert val_ids.isdisjoint(test_ids), "Val and Test IDs overlap!"
    assert train_ids.isdisjoint(val_ids), "Train and Val IDs overlap!"


def test_all_manifest_file_paths_exist():
    """Verify that every image, mask, and geojson path listed in manifests exists on disk."""
    for split_name in ["train.json", "val.json", "test.json"]:
        path = os.path.join("data/splits", split_name)
        assert os.path.exists(path), f"Missing split file {path}"

        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)

        for r in records:
            assert os.path.exists(r["image_path"]), f"Image not found: {r['image_path']}"
            assert os.path.exists(r["mask_path"]), f"Mask not found: {r['mask_path']}"
            assert os.path.exists(r["geojson_path"]), f"GeoJSON not found: {r['geojson_path']}"
            # Verify POSIX path format (forward slashes)
            assert "\\" not in r["image_path"], f"Non-POSIX backslash in path: {r['image_path']}"


def test_qa_report_and_boundary_masks():
    """Verify that QA report passed all 31 tiles and boundary masks were generated."""
    qa_path = "outputs/qa_report.csv"
    assert os.path.exists(qa_path), f"Missing QA report: {qa_path}"

    with open(qa_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 31, f"Expected 31 rows in QA report, got {len(rows)}"
    failed_rows = [r for r in rows if r["status"] == "FAILED"]
    assert len(failed_rows) == 0, f"QA report contains {len(failed_rows)} failed tiles: {failed_rows}"

    boundary_files = [f for f in os.listdir("data/masks_boundary") if f.endswith(".png")]
    assert len(boundary_files) == 31, f"Expected 31 boundary masks, found {len(boundary_files)}"


import unittest

class TestPhaseB(unittest.TestCase):
    """Phase B regression and data audit test suite."""

    def test_01_split_manifest_and_disjointness(self):
        test_split_manifest_and_disjointness()

    def test_02_all_manifest_file_paths_exist(self):
        test_all_manifest_file_paths_exist()

    def test_03_qa_report_and_boundary_masks(self):
        test_qa_report_and_boundary_masks()


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING PHASE B REGRESSION & AUDIT SUITE")
    print("=" * 60)

    print("[1/3] Testing split_v1.json and mathematical disjointness...")
    test_split_manifest_and_disjointness()
    print("  -> Passed.")

    print("[2/3] Testing manifest POSIX paths and file existence...")
    test_all_manifest_file_paths_exist()
    print("  -> Passed.")

    print("[3/3] Testing QA report integrity and boundary masks...")
    test_qa_report_and_boundary_masks()
    print("  -> Passed.")

    print("=" * 60)
    print("ALL 3 PHASE B TESTS PASSED PERFECTLY!")
    print("=" * 60)
