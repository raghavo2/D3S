"""
GPS-based metric scale factor computation for point cloud → real-world dimensions.

Uses Haversine-computed GPS distances between camera positions and compares them
against the corresponding distances in DUSt3R's arbitrary reconstruction space.
The robust median ratio gives the scale factor to convert DUSt3R units → meters.

V3 module — this is the key to showing accurate structure dimensions.
"""

import os
import json
import math
import numpy as np


def _haversine_distance(lat1, lon1, alt1, lat2, lon2, alt2):
    """
    Compute 3D distance between two GPS coordinates in meters.
    
    Uses Haversine formula for horizontal distance + Euclidean altitude difference.
    
    Args:
        lat1, lon1, alt1: First point (degrees, degrees, meters)
        lat2, lon2, alt2: Second point (degrees, degrees, meters)
    
    Returns:
        Distance in meters.
    """
    R = 6371000.0  # Earth radius in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (math.sin(dphi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    horizontal_dist = R * c
    vertical_dist = abs(alt2 - alt1)

    return math.sqrt(horizontal_dist ** 2 + vertical_dist ** 2)


def compute_scale_factor(
    gps_positions,
    reconstruction_positions,
    min_gps_distance=2.0,
    min_pairs=3,
    on_log=None,
):
    """
    Compute the metric scale factor from GPS ↔ reconstruction position pairs.

    Algorithm:
      1. Match GPS positions to reconstruction camera positions by filename/index
      2. Compute pairwise GPS distances (Haversine, meters)
      3. Compute pairwise reconstruction distances (Euclidean, arbitrary units)
      4. For each pair with GPS distance > min_gps_distance:
         ratio = gps_distance / reconstruction_distance
      5. Robust scale factor = median of all valid ratios

    Args:
        gps_positions: list of dicts with keys:
            {filename (or frame_index), latitude, longitude, altitude}
        reconstruction_positions: list of dicts with keys:
            {filename (or frame_index), x, y, z}
            These are camera centers from DUSt3R's global alignment.
        min_gps_distance: Minimum GPS distance (meters) to include a pair.
            Very short distances produce noisy ratios. Default 2.0m.
        min_pairs: Minimum number of valid pairs for reliable scaling.
        on_log: Optional progress callback.

    Returns:
        dict with keys:
            scale_factor: float (multiply reconstruction coords by this to get meters)
            gps_pairs_used: int
            total_pairs_considered: int
            median_residual_m: float (median error in meters after scaling)
            gps_centroid: {latitude, longitude, altitude}
            reconstruction_centroid: {x, y, z}
            reliable: bool (True if enough pairs were used)
            method: str ("gps_haversine" or "fallback_none")
        Returns None if computation fails completely.
    """
    log = on_log or (lambda msg: None)

    if not gps_positions or not reconstruction_positions:
        log("No GPS or reconstruction positions provided — cannot compute scale")
        return {
            "scale_factor": 1.0,
            "reliable": False,
            "method": "fallback_none",
            "gps_pairs_used": 0,
        }

    # --- Build lookup by filename or index ---
    gps_by_key = {}
    for g in gps_positions:
        key = g.get("filename") or str(g.get("frame_index", ""))
        if key:
            gps_by_key[key] = g

    recon_by_key = {}
    for r in reconstruction_positions:
        key = r.get("filename") or str(r.get("frame_index", ""))
        if key:
            recon_by_key[key] = r

    # --- Find common keys ---
    common_keys = sorted(set(gps_by_key.keys()) & set(recon_by_key.keys()))
    if len(common_keys) < 2:
        log(f"Only {len(common_keys)} matched positions — need at least 2 for scaling")
        return {
            "scale_factor": 1.0,
            "reliable": False,
            "method": "fallback_none",
            "gps_pairs_used": 0,
            "matched_positions": len(common_keys),
        }

    log(f"Matched {len(common_keys)} GPS ↔ reconstruction positions")

    # --- Compute pairwise distances ---
    ratios = []
    total_pairs = 0
    valid_pairs = 0

    for i in range(len(common_keys)):
        for j in range(i + 1, len(common_keys)):
            total_pairs += 1
            k_i, k_j = common_keys[i], common_keys[j]
            g_i, g_j = gps_by_key[k_i], gps_by_key[k_j]
            r_i, r_j = recon_by_key[k_i], recon_by_key[k_j]

            # GPS distance (meters)
            gps_dist = _haversine_distance(
                g_i["latitude"], g_i["longitude"], g_i.get("altitude", 0),
                g_j["latitude"], g_j["longitude"], g_j.get("altitude", 0),
            )

            # Skip very short GPS distances (noisy ratio)
            if gps_dist < min_gps_distance:
                continue

            # Reconstruction distance (arbitrary units)
            recon_dist = math.sqrt(
                (r_i["x"] - r_j["x"]) ** 2 +
                (r_i["y"] - r_j["y"]) ** 2 +
                (r_i["z"] - r_j["z"]) ** 2
            )

            if recon_dist < 1e-8:
                continue

            ratio = gps_dist / recon_dist
            ratios.append({
                "pair": (k_i, k_j),
                "gps_distance_m": gps_dist,
                "recon_distance": recon_dist,
                "ratio": ratio,
            })
            valid_pairs += 1

    log(
        f"Scale pairs: {valid_pairs} valid / {total_pairs} total "
        f"(filtered by min GPS distance {min_gps_distance}m)"
    )

    if valid_pairs < min_pairs:
        log(f"⚠ Only {valid_pairs} valid pairs (need {min_pairs}) — unreliable scale")
        if valid_pairs == 0:
            return {
                "scale_factor": 1.0,
                "reliable": False,
                "method": "fallback_none",
                "gps_pairs_used": 0,
                "total_pairs_considered": total_pairs,
            }

    # --- Robust median scale factor ---
    all_ratios = [r["ratio"] for r in ratios]
    scale_factor = float(np.median(all_ratios))

    # --- Compute residual errors ---
    residuals = []
    for r in ratios:
        scaled_recon = r["recon_distance"] * scale_factor
        error_m = abs(r["gps_distance_m"] - scaled_recon)
        residuals.append(error_m)
    median_residual = float(np.median(residuals)) if residuals else 0

    # --- GPS centroid ---
    gps_lats = [gps_by_key[k]["latitude"] for k in common_keys]
    gps_lons = [gps_by_key[k]["longitude"] for k in common_keys]
    gps_alts = [gps_by_key[k].get("altitude", 0) for k in common_keys]
    gps_centroid = {
        "latitude": sum(gps_lats) / len(gps_lats),
        "longitude": sum(gps_lons) / len(gps_lons),
        "altitude": sum(gps_alts) / len(gps_alts),
    }

    # --- Reconstruction centroid ---
    recon_xs = [recon_by_key[k]["x"] for k in common_keys]
    recon_ys = [recon_by_key[k]["y"] for k in common_keys]
    recon_zs = [recon_by_key[k]["z"] for k in common_keys]
    recon_centroid = {
        "x": sum(recon_xs) / len(recon_xs),
        "y": sum(recon_ys) / len(recon_ys),
        "z": sum(recon_zs) / len(recon_zs),
    }

    result = {
        "scale_factor": round(scale_factor, 6),
        "gps_pairs_used": valid_pairs,
        "total_pairs_considered": total_pairs,
        "median_residual_m": round(median_residual, 4),
        "gps_centroid": gps_centroid,
        "reconstruction_centroid": recon_centroid,
        "reliable": valid_pairs >= min_pairs,
        "method": "gps_haversine",
        "ratio_stats": {
            "median": round(float(np.median(all_ratios)), 6),
            "mean": round(float(np.mean(all_ratios)), 6),
            "std": round(float(np.std(all_ratios)), 6),
            "min": round(float(np.min(all_ratios)), 6),
            "max": round(float(np.max(all_ratios)), 6),
        },
    }

    log(
        f"Scale factor: {result['scale_factor']:.4f} "
        f"(1 DUSt3R unit = {result['scale_factor']:.4f}m)"
    )
    log(
        f"Residual: median {result['median_residual_m']:.3f}m, "
        f"ratio std {result['ratio_stats']['std']:.4f}"
    )

    return result


def apply_scale_to_point_cloud(points, scale_factor, center=None):
    """
    Apply the metric scale factor to a point cloud.
    
    Args:
        points: numpy array of shape (N, 3) in DUSt3R's arbitrary units.
        scale_factor: float from compute_scale_factor().
        center: optional (3,) array — if provided, the point cloud is
                re-centered to this origin before scaling.
    
    Returns:
        numpy array of shape (N, 3) in meters.
    """
    scaled = points.copy()

    if center is not None:
        center = np.asarray(center, dtype=np.float64)
        scaled = scaled - center

    scaled = scaled * scale_factor
    return scaled


def save_scale_info(scale_info, output_path, on_log=None):
    """
    Save scale info JSON alongside the point cloud.
    
    Also computes and adds bounding box dimensions in meters if a
    scaled point cloud is available.
    """
    log = on_log or (lambda msg: None)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scale_info, f, indent=2)
    log(f"Scale info saved: {output_path}")


def compute_dimensions_from_points(points, scale_factor=1.0):
    """
    Compute bounding box dimensions from a point cloud.
    
    Args:
        points: numpy array (N, 3) in reconstruction units.
        scale_factor: multiply to get meters.
    
    Returns:
        dict with width_m, height_m, depth_m, volume_m3
    """
    if len(points) == 0:
        return {"width_m": 0, "height_m": 0, "depth_m": 0, "volume_m3": 0}

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extents = (maxs - mins) * scale_factor

    return {
        "width_m": round(float(extents[0]), 3),
        "height_m": round(float(extents[1]), 3),
        "depth_m": round(float(extents[2]), 3),
        "volume_m3": round(float(extents[0] * extents[1] * extents[2]), 3),
        "bbox_min_m": (mins * scale_factor).tolist(),
        "bbox_max_m": (maxs * scale_factor).tolist(),
    }
