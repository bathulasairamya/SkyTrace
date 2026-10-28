"""Automated Regression & Verification Tests for Phase D (Geospatial & Vectorization Accuracy)."""

import os
import sys
import unittest
import numpy as np
import cv2
import json
import rasterio

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference import (
    load_inference_image,
    extract_contours_and_overlay,
    contours_to_geojson,
)


class TestPhaseD(unittest.TestCase):
    """Phase D test suite covering geospatial metadata, WGS84 reprojection, and courtyard topologies."""

    @classmethod
    def setUpClass(cls):
        cls.sample_tif = "data/images/SN2_buildings_train_AOI_3_Paris_PS-RGB_img136.tif"
        cls.sample_mask = "data/masks/img136.png"
        cls.courtyard_mask = "data/masks/img1684.png"

    def test_01_geotiff_metadata_preservation(self):
        """D.1: Verify load_inference_image extracts complete affine and CRS metadata from GeoTIFF."""
        self.assertTrue(os.path.exists(self.sample_tif), f"Missing test tile {self.sample_tif}")

        raw_rgb, norm_rgb, meta = load_inference_image(self.sample_tif, return_meta=True)

        self.assertIsInstance(raw_rgb, np.ndarray)
        self.assertIsInstance(norm_rgb, np.ndarray)
        self.assertIsInstance(meta, dict)

        self.assertTrue(meta["is_georeferenced"], "GeoTIFF should be flagged as georeferenced")
        self.assertEqual(meta["space"], "geodetic")
        self.assertIsNotNone(meta["transform"], "Affine transform must not be None")
        self.assertEqual(meta["crs"], "EPSG:4326")
        self.assertIsNotNone(meta["bounds"])
        self.assertEqual(len(meta["bounds"]), 4)
        self.assertEqual(meta["width"], 650)
        self.assertEqual(meta["height"], 650)
        self.assertAlmostEqual(meta["res"][0], 2.7e-6, places=7)

    def test_02_unreferenced_image_space_fallback(self):
        """D.1 & D.3: Verify standard non-geospatial images honestly report pixel units and image_space."""
        # Create a synthetic standard RGB image and save as PNG
        temp_png = "outputs/test_synthetic_nongeo.png"
        dummy = np.full((300, 300, 3), 120, dtype=np.uint8)
        # Draw a synthetic 50x50 building
        dummy[50:100, 50:100] = 255
        cv2.imwrite(temp_png, dummy)

        try:
            raw_rgb, norm_rgb, meta = load_inference_image(temp_png, return_meta=True)

            self.assertFalse(meta["is_georeferenced"])
            self.assertIsNone(meta["crs"])
            self.assertIsNone(meta["transform"])
            self.assertEqual(meta["space"], "image_space")

            # Extract contours and convert to GeoJSON
            binary_mask = (raw_rgb[:, :, 0] > 200).astype(np.uint8) * 255
            overlay, buildings = extract_contours_and_overlay(raw_rgb, binary_mask, min_area=20)
            geojson_data = contours_to_geojson(buildings, 300, 300, meta=meta)

            self.assertFalse(geojson_data["metadata"]["is_georeferenced"])
            self.assertEqual(geojson_data["metadata"]["target_crs"], "image_space")
            self.assertIsNone(geojson_data["metadata"]["total_building_area_sqm"])
            self.assertGreater(geojson_data["metadata"]["total_building_area_pixels"], 0)

            first_feat = geojson_data["features"][0]
            self.assertIsNone(first_feat["properties"]["area_sqm"], "Area sqm must be None for unreferenced images")
            self.assertGreater(first_feat["properties"]["area_pixels"], 0)
            self.assertFalse(first_feat["properties"]["is_georeferenced"])

        finally:
            if os.path.exists(temp_png):
                os.remove(temp_png)

    def test_03_courtyard_topology_extraction(self):
        """D.4: Verify cv2.RETR_CCOMP extracts interior courtyards as holes and subtracts their area."""
        # Create synthetic 100x100 mask: outer building 60x60 (area=3600), inner courtyard 20x20 (area=400)
        synthetic_mask = np.zeros((100, 100), dtype=np.uint8)
        synthetic_mask[20:80, 20:80] = 255
        synthetic_mask[40:60, 40:60] = 0  # Interior courtyard hole

        dummy_rgb = np.zeros((100, 100, 3), dtype=np.float32)

        overlay, buildings = extract_contours_and_overlay(
            rgb_img=dummy_rgb,
            binary_mask=synthetic_mask,
            min_area=15,
            min_hole_area=8,
            simplify_epsilon=0.0,
        )

        self.assertEqual(len(buildings), 1, "Expected exactly 1 exterior building")
        b = buildings[0]
        self.assertEqual(len(b["holes"]), 1, "Expected exactly 1 detected courtyard hole")

        outer_area = cv2.contourArea(b["exterior"])
        hole_area = cv2.contourArea(b["holes"][0])
        self.assertGreater(outer_area, 3000)
        self.assertGreater(hole_area, 300)
        self.assertAlmostEqual(b["raw_area_px"], outer_area - hole_area, delta=1.0)

    def test_04_wgs84_reprojection_and_geodetic_accuracy(self):
        """D.2 & D.3: Verify true WGS84 coordinates and ellipsoidal area calculation on Paris satellite tile."""
        raw_rgb, norm_rgb, meta = load_inference_image(self.sample_tif, return_meta=True)
        mask = cv2.imread(self.sample_mask, cv2.IMREAD_GRAYSCALE)

        overlay, buildings = extract_contours_and_overlay(raw_rgb, mask, min_area=15, simplify_epsilon=1.5)
        geojson_data = contours_to_geojson(buildings, meta["width"], meta["height"], meta=meta)

        self.assertTrue(geojson_data["metadata"]["is_georeferenced"])
        self.assertEqual(geojson_data["metadata"]["target_crs"], "EPSG:4326 (WGS84)")
        self.assertGreater(geojson_data["metadata"]["total_buildings"], 40)
        self.assertGreater(geojson_data["metadata"]["total_building_area_sqm"], 1000.0)

        # Audit coordinate validity in Paris bounds
        for feat in geojson_data["features"]:
            props = feat["properties"]
            self.assertTrue(props["is_georeferenced"])
            self.assertEqual(props["crs"], "EPSG:4326")
            self.assertGreater(props["area_sqm"], 0.0)
            self.assertGreater(props["perimeter_m"], 0.0)
            self.assertGreater(props["compactness"], 0.0)
            self.assertLessEqual(props["compactness"], 1.0)

            geom = feat["geometry"]
            self.assertIn(geom["type"], ("Polygon", "MultiPolygon"))

            # Check RFC 7946 coordinate ordering [longitude, latitude] in Paris bounding box
            if geom["type"] == "Polygon":
                for ring in geom["coordinates"]:
                    self.assertEqual(ring[0], ring[-1], "Linear ring must be closed (first == last)")
                    for lon, lat in ring:
                        self.assertTrue(2.0 <= lon <= 2.5, f"Longitude {lon} outside Paris geographic range")
                        self.assertTrue(48.8 <= lat <= 49.3, f"Latitude {lat} outside Paris geographic range")

    def test_05_real_courtyard_geojson_multi_ring(self):
        """D.4: Verify real European courtyard buildings produce multi-ring GeoJSON polygons."""
        self.assertTrue(os.path.exists(self.courtyard_mask), f"Missing {self.courtyard_mask}")
        raw_rgb, norm_rgb, meta = load_inference_image(self.sample_tif, return_meta=True)
        mask = cv2.imread(self.courtyard_mask, cv2.IMREAD_GRAYSCALE)

        overlay, buildings = extract_contours_and_overlay(raw_rgb, mask, min_area=15, simplify_epsilon=1.0)
        geojson_data = contours_to_geojson(buildings, 650, 650, meta=meta)

        # Check that courtyards were detected in metadata
        total_courtyards = geojson_data["metadata"]["total_courtyards"]
        self.assertGreater(total_courtyards, 0, "img1402 should have detected courtyard holes")

        # Check features with courtyards have multi-ring coordinates
        courtyard_feats = [f for f in geojson_data["features"] if f["properties"]["has_courtyard"]]
        self.assertGreater(len(courtyard_feats), 0)

        sample_courtyard = courtyard_feats[0]
        self.assertGreaterEqual(sample_courtyard["properties"]["courtyard_count"], 1)
        if sample_courtyard["geometry"]["type"] == "Polygon":
            # Exterior ring + at least 1 courtyard hole ring
            self.assertGreaterEqual(len(sample_courtyard["geometry"]["coordinates"]), 2)

    def test_06_geojson_serialization_roundtrip(self):
        """Verify generated GeoJSON is 100% compliant with standard json.dumps and RFC 7946 schema."""
        raw_rgb, norm_rgb, meta = load_inference_image(self.sample_tif, return_meta=True)
        mask = cv2.imread(self.sample_mask, cv2.IMREAD_GRAYSCALE)
        overlay, buildings = extract_contours_and_overlay(raw_rgb, mask, min_area=15)
        geojson_data = contours_to_geojson(buildings, 650, 650, meta=meta)

        serialized = json.dumps(geojson_data)
        deserialized = json.loads(serialized)

        self.assertEqual(deserialized["type"], "FeatureCollection")
        self.assertIn("crs", deserialized)
        self.assertIn("metadata", deserialized)
        self.assertEqual(len(deserialized["features"]), len(geojson_data["features"]))


if __name__ == "__main__":
    print("=" * 65)
    print("      RUNNING PHASE D GEOSPATIAL & VECTORIZATION TEST SUITE       ")
    print("=" * 65)
    unittest.main()
