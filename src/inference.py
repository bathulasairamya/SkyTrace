"""Sliding-Window Inference & Building Footprint Outline Extractor."""

import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import rasterio
import cv2
import torch
import pyproj
import shapely
from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.validation import make_valid
from shapely.ops import unary_union

from src.models import build_model


def load_inference_image(image_path: str, return_meta: bool = True):
    """Load satellite image supporting both GeoTIFF (.tif) and standard RGB (.png, .jpg).
    
    Returns:
        img: np.ndarray, float32 RGB array in [0, 255] range.
        img_norm: np.ndarray, float32 normalized [0, 1] array for model inference.
        meta (if return_meta=True): dict with georeferencing metadata:
            - transform: rasterio Affine transform matrix or None
            - crs: CRS string (e.g. 'EPSG:4326') or None
            - crs_obj: rasterio.crs.CRS or None
            - width: raster width in pixels
            - height: raster height in pixels
            - res: spatial resolution tuple (x_res, y_res)
            - bounds: [left, bottom, right, top] coordinates or None
            - is_georeferenced: bool
            - space: 'geodetic' or 'image_space'
    """
    if image_path.lower().endswith((".tif", ".tiff")):
        with rasterio.open(image_path) as src:
            bands = min(3, src.count)
            img = src.read(list(range(1, bands + 1))).transpose(1, 2, 0).astype("float32")
            if bands == 1:
                img = np.repeat(img, 3, axis=2)

            meta = {
                "transform": src.transform,
                "crs": src.crs.to_string() if src.crs else None,
                "crs_obj": src.crs,
                "width": int(src.width),
                "height": int(src.height),
                "res": (float(src.res[0]), float(src.res[1])),
                "bounds": [float(src.bounds.left), float(src.bounds.bottom), float(src.bounds.right), float(src.bounds.top)] if src.crs else None,
                "is_georeferenced": bool(src.crs is not None),
                "space": "geodetic" if src.crs is not None else "image_space",
            }
    else:
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise ValueError(f"Could not read image from {image_path}")
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype("float32")
        h, w = img.shape[:2]
        meta = {
            "transform": None,
            "crs": None,
            "crs_obj": None,
            "width": int(w),
            "height": int(h),
            "res": (1.0, 1.0),
            "bounds": None,
            "is_georeferenced": False,
            "space": "image_space",
        }

    # Min-Max Normalization to [0, 1] matching pretrained training distribution
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img_norm = (img - img_min) / (img_max - img_min)
    else:
        img_norm = np.zeros_like(img)

    if return_meta:
        return img, img_norm, meta
    return img, img_norm


def predict_sliding_window(
    model: torch.nn.Module,
    img_norm: np.ndarray,
    patch_size: int = 512,
    stride: int = 256,
    device: torch.device = None,
    threshold: float = 0.5,
):
    """Run sliding-window inference with weighted blending for seamless tile prediction."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    h, w, c = img_norm.shape

    # Pad image if smaller than patch_size
    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)
    if pad_h > 0 or pad_w > 0:
        img_norm = np.pad(img_norm, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        padded_h, padded_w, _ = img_norm.shape
    else:
        padded_h, padded_w = h, w

    prob_map = np.zeros((padded_h, padded_w), dtype=np.float32)
    count_map = np.zeros((padded_h, padded_w), dtype=np.float32)

    # 2D Hanning/spline window for edge smoothing
    window = np.outer(np.hanning(patch_size), np.hanning(patch_size)).astype(np.float32)
    window = np.maximum(window, 1e-4)

    y_steps = range(0, padded_h - patch_size + 1, stride)
    x_steps = range(0, padded_w - patch_size + 1, stride)

    # Ensure last edges are covered
    y_coords = list(y_steps)
    if not y_coords or y_coords[-1] != (padded_h - patch_size):
        y_coords.append(padded_h - patch_size)

    x_coords = list(x_steps)
    if not x_coords or x_coords[-1] != (padded_w - patch_size):
        x_coords.append(padded_w - patch_size)

    def _predict_patch(patch_arr):
        """Forward pass with optional Test-Time Augmentation (TTA: flips + rot90)."""
        patch_t = torch.tensor(patch_arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        out = model(patch_t)
        prob = torch.sigmoid(out).squeeze().cpu().numpy()
        return prob

    with torch.no_grad():
        for y in y_coords:
            for x in x_coords:
                patch = img_norm[y : y + patch_size, x : x + patch_size]

                # Standard prediction
                prob = _predict_patch(patch)

                prob_map[y : y + patch_size, x : x + patch_size] += prob * window
                count_map[y : y + patch_size, x : x + patch_size] += window

    # Un-pad and normalize probabilities
    prob_map = prob_map[:h, :w]
    count_map = count_map[:h, :w]
    final_prob = prob_map / np.maximum(count_map, 1e-7)
    binary_mask = (final_prob > threshold).astype(np.uint8) * 255

    return final_prob, binary_mask


def predict_sliding_window_tta(
    model: torch.nn.Module,
    img_norm: np.ndarray,
    patch_size: int = 512,
    stride: int = 256,
    device: torch.device = None,
    threshold: float = 0.5,
    mode: str = "4-view",
):
    """Test-Time Augmentation (TTA) inference: blends normal, h-flip, and optional v-flip + rot90 predictions."""
    # 1. Base prediction
    prob_orig, _ = predict_sliding_window(model, img_norm, patch_size, stride, device, threshold)

    # 2. Horizontal flip
    img_h = np.fliplr(img_norm).copy()
    prob_h, _ = predict_sliding_window(model, img_h, patch_size, stride, device, threshold)
    prob_h = np.fliplr(prob_h)

    if mode in ["2-view", "hflip"]:
        final_prob = (prob_orig + prob_h) / 2.0
    else:
        # 3. Vertical flip
        img_v = np.flipud(img_norm).copy()
        prob_v, _ = predict_sliding_window(model, img_v, patch_size, stride, device, threshold)
        prob_v = np.flipud(prob_v)

        # 4. Rotate 90
        img_rot = np.rot90(img_norm, k=1).copy()
        prob_rot, _ = predict_sliding_window(model, img_rot, patch_size, stride, device, threshold)
        prob_rot = np.rot90(prob_rot, k=-1)

        final_prob = (prob_orig + prob_h + prob_v + prob_rot) / 4.0

    binary_mask = (final_prob > threshold).astype(np.uint8) * 255
    return final_prob, binary_mask


def extract_contours_and_overlay(
    rgb_img: np.ndarray,
    binary_mask: np.ndarray,
    contour_color=(0, 230, 255),
    courtyard_color=(255, 140, 0),
    min_area: int = 15,
    min_hole_area: int = 8,
    simplify_epsilon: float = 1.5,
    morph_mode: str = "open",
):
    """Extract detected building footprints with courtyard hole topology and overlay on RGB satellite image.
    
    Uses cv2.RETR_CCOMP hierarchy to accurately capture exterior building perimeters
    along with interior cutout courtyards. Applies morphological noise reduction and
    Douglas-Peucker polygon approximation for sharp geometric footprints.
    
    Returns:
        overlay: np.ndarray, RGB image with drawn contours (exterior in contour_color, courtyards in courtyard_color).
        buildings: list of dicts, each containing:
            - 'exterior': np.ndarray simplified exterior contour
            - 'holes': list of np.ndarray simplified courtyard contours
            - 'raw_area_px': net pixel area (exterior - holes)
    """
    rgb_disp = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
    rgb_disp = (np.clip(rgb_disp, 0, 1) * 255).astype(np.uint8).copy()

    # Morphological refinement: remove isolated noise specks while preserving interior courtyards
    clean_mask = binary_mask.copy()
    if morph_mode == "open":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)
    elif morph_mode == "close_open":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)

    # 2-level hierarchy: exterior building boundaries and interior courtyard holes
    raw_contours, hierarchy = cv2.findContours(clean_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    buildings = []
    all_exterior_contours = []
    all_hole_contours = []

    def _simplify(cnt):
        if simplify_epsilon > 0:
            simp = cv2.approxPolyDP(cnt, simplify_epsilon, closed=True)
            return simp if len(simp) >= 3 else cnt
        return cnt

    if hierarchy is not None and len(raw_contours) > 0:
        for idx in range(len(raw_contours)):
            # Top-level exterior contours have parent == -1
            if hierarchy[0][idx][3] == -1:
                ext_cnt = raw_contours[idx]
                ext_area = cv2.contourArea(ext_cnt)
                if ext_area < min_area:
                    continue

                simp_ext = _simplify(ext_cnt)
                all_exterior_contours.append(simp_ext)

                # Collect all immediate child holes (courtyards)
                child_idx = hierarchy[0][idx][2]
                holes = []
                while child_idx != -1:
                    hole_cnt = raw_contours[child_idx]
                    if cv2.contourArea(hole_cnt) >= min_hole_area:
                        simp_hole = _simplify(hole_cnt)
                        holes.append(simp_hole)
                        all_hole_contours.append(simp_hole)
                    child_idx = hierarchy[0][child_idx][0]  # Next sibling child

                net_area = float(cv2.contourArea(simp_ext) - sum(cv2.contourArea(h) for h in holes))
                buildings.append({
                    "exterior": simp_ext,
                    "holes": holes,
                    "raw_area_px": max(1.0, net_area),
                })

    # Render overlays: exterior outlines and courtyard boundaries
    overlay = rgb_disp.copy()
    if all_exterior_contours:
        cv2.drawContours(overlay, all_exterior_contours, -1, contour_color, 2)
    if all_hole_contours:
        cv2.drawContours(overlay, all_hole_contours, -1, courtyard_color, 2)

    return overlay, buildings


def contours_to_geojson(
    contours: list,
    img_width: int,
    img_height: int,
    meta: dict = None,
) -> dict:
    """Serialize detected building footprints into an industry-standard RFC 7946 GeoJSON FeatureCollection.
    
    If georeferenced metadata is available from the input GeoTIFF:
      - Automatically reprojects pixel coordinates to real-world WGS84 [longitude, latitude].
      - Computes rigorous ellipsoidal area (m²) and perimeter (m) on the WGS84 datum via pyproj.Geod.
      - Preserves multi-ring courtyard hole topologies.
      - Calculates polygon shape compactness (Isoperimetric Quotient: 4*pi*Area / P²).
    
    If input is unreferenced (standard PNG/JPG):
      - Retains coordinates in image pixel space.
      - Accurately reports pixel area without fabricating metric measurements.
    """
    is_geo = bool(meta and meta.get("is_georeferenced") and meta.get("transform") is not None)
    features = []

    if is_geo:
        transform = meta["transform"]
        src_crs = meta.get("crs_obj") or meta.get("crs") or "EPSG:4326"
        transformer = pyproj.Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        geod = pyproj.Geod(ellps="WGS84")

        def _to_wgs84(pt):
            x_geo, y_geo = rasterio.transform.xy(transform, float(pt[1]), float(pt[0]), offset="center")
            lon, lat = transformer.transform(x_geo, y_geo)
            return (round(float(lon), 7), round(float(lat), 7))

    for idx, item in enumerate(contours):
        if isinstance(item, dict):
            ext_cnt = item.get("exterior")
            holes = item.get("holes", [])
        else:
            ext_cnt = item
            holes = []

        if ext_cnt is None:
            continue

        ext_pts = ext_cnt.squeeze()
        if len(ext_pts.shape) != 2 or ext_pts.shape[0] < 3:
            continue

        if is_geo:
            ext_ring = [_to_wgs84(p) for p in ext_pts]
            if ext_ring[0] != ext_ring[-1]:
                ext_ring.append(ext_ring[0])
            if len(ext_ring) < 4:
                continue

            hole_rings = []
            for h in holes:
                h_pts = h.squeeze()
                if len(h_pts.shape) == 2 and h_pts.shape[0] >= 3:
                    h_ring = [_to_wgs84(p) for p in h_pts]
                    if h_ring[0] != h_ring[-1]:
                        h_ring.append(h_ring[0])
                    if len(h_ring) >= 4:
                        hole_rings.append(h_ring)

            try:
                poly = Polygon(shell=ext_ring, holes=hole_rings)
                if not poly.is_valid:
                    poly = make_valid(poly)
            except Exception:
                poly = Polygon(shell=ext_ring)
                if not poly.is_valid:
                    poly = make_valid(poly)

            if poly.geom_type == "GeometryCollection":
                polys = [g for g in poly.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
                if not polys:
                    continue
                poly = unary_union(polys)

            try:
                area_sqm, perim_m = geod.geometry_area_perimeter(poly)
                area_sqm = abs(float(area_sqm))
                perim_m = abs(float(perim_m))
            except Exception:
                area_sqm = 0.0
                perim_m = 0.0

            if area_sqm < 0.5:
                continue

            compactness = round(min(1.0, (4.0 * np.pi * area_sqm) / (perim_m ** 2)), 3) if perim_m > 0 else 0.0
            area_px = float(cv2.contourArea(ext_cnt) - sum(cv2.contourArea(h) for h in holes))

            if hole_rings:
                geom = {
                    "type": "Polygon",
                    "coordinates": [ext_ring] + hole_rings,
                }
            else:
                geom = mapping(poly)

            features.append({
                "type": "Feature",
                "id": idx + 1,
                "geometry": geom,
                "properties": {
                    "building_id": idx + 1,
                    "area_sqm": round(area_sqm, 2),
                    "perimeter_m": round(perim_m, 2),
                    "area_pixels": round(max(1.0, area_px), 1),
                    "compactness": compactness,
                    "has_courtyard": len(hole_rings) > 0,
                    "courtyard_count": len(hole_rings),
                    "is_georeferenced": True,
                    "crs": "EPSG:4326",
                },
            })

        else:
            # Image space (pixel units)
            ext_ring = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in ext_pts]
            if ext_ring[0] != ext_ring[-1]:
                ext_ring.append(ext_ring[0])
            if len(ext_ring) < 4:
                continue

            hole_rings = []
            for h in holes:
                h_pts = h.squeeze()
                if len(h_pts.shape) == 2 and h_pts.shape[0] >= 3:
                    h_ring = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in h_pts]
                    if h_ring[0] != h_ring[-1]:
                        h_ring.append(h_ring[0])
                    if len(h_ring) >= 4:
                        hole_rings.append(h_ring)

            try:
                poly = Polygon(shell=ext_ring, holes=hole_rings)
                if not poly.is_valid:
                    poly = make_valid(poly)
            except Exception:
                poly = Polygon(shell=ext_ring)
                if not poly.is_valid:
                    poly = make_valid(poly)

            if poly.geom_type == "GeometryCollection":
                polys = [g for g in poly.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
                if not polys:
                    continue
                poly = unary_union(polys)

            area_px = float(poly.area)
            perim_px = float(poly.length)
            compactness = round(min(1.0, (4.0 * np.pi * area_px) / (perim_px ** 2)), 3) if perim_px > 0 else 0.0

            if hole_rings:
                geom = {
                    "type": "Polygon",
                    "coordinates": [ext_ring] + hole_rings,
                }
            else:
                geom = mapping(poly)

            features.append({
                "type": "Feature",
                "id": idx + 1,
                "geometry": geom,
                "properties": {
                    "building_id": idx + 1,
                    "area_pixels": round(area_px, 1),
                    "area_sqm": None,
                    "perimeter_m": None,
                    "perimeter_pixels": round(perim_px, 1),
                    "compactness": compactness,
                    "has_courtyard": len(hole_rings) > 0,
                    "courtyard_count": len(hole_rings),
                    "is_georeferenced": False,
                    "crs": None,
                },
            })

    total_area_sqm = sum(f["properties"]["area_sqm"] for f in features if f["properties"].get("area_sqm") is not None)
    total_area_px = sum(f["properties"]["area_pixels"] for f in features)
    total_courtyards = sum(f["properties"]["courtyard_count"] for f in features)
    bldgs_with_courtyard = sum(1 for f in features if f["properties"]["has_courtyard"])

    return {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:OGC:1.3:CRS84" if is_geo else "image-pixel-space"
            }
        },
        "metadata": {
            "generator": "SkyTrace AI Geospatial Engine",
            "total_buildings": len(features),
            "buildings_with_courtyards": bldgs_with_courtyard,
            "total_courtyards": total_courtyards,
            "total_building_area_sqm": round(total_area_sqm, 2) if is_geo else None,
            "total_building_area_pixels": round(total_area_px, 1),
            "is_georeferenced": is_geo,
            "source_crs": meta.get("crs") if meta else None,
            "target_crs": "EPSG:4326 (WGS84)" if is_geo else "image_space",
            "bounds_wgs84": meta.get("bounds") if (meta and is_geo) else None,
            "image_dimensions": [img_width, img_height],
        },
        "features": features,
    }

