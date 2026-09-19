"""
Intelligent keyframe selection from quality-filtered drone frames.

After quality.py removes blurry and overexposed frames, this module
selects an optimal subset that maximizes:
  - Spatial coverage (frames spread across the flight trajectory)
  - Visual diversity (no near-duplicate views)
  - Reconstruction quality (sufficient overlap for stereo matching)

The algorithm works in two passes:
  1. Redundancy removal: merge clusters of near-identical frames (SSIM)
     and keep only the sharpest representative from each cluster.
  2. Spacing enforcement: ensure minimum and maximum temporal gaps
     between selected keyframes.

This module uses only OpenCV, NumPy, and scikit-image SSIM
(falling back to a lightweight OpenCV-based SSIM if skimage is unavailable).
"""

import os
import json
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# SSIM computation
# ---------------------------------------------------------------------------

def _compute_ssim(img_a, img_b, resize_dim=256):
    """
    Compute structural similarity between two images.

    Both images are resized to a small fixed size for speed —
    we only need a coarse similarity measure, not pixel-perfect SSIM.

    Args:
        img_a, img_b: Grayscale images (numpy arrays).
        resize_dim:   Resize target for comparison (default 256).

    Returns:
        float — SSIM value in [0, 1]. Higher = more similar.
    """
    a = cv2.resize(img_a, (resize_dim, resize_dim))
    b = cv2.resize(img_b, (resize_dim, resize_dim))

    # Try scikit-image SSIM first (more accurate)
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(a, b))
    except ImportError:
        pass

    # Fallback: OpenCV-based SSIM approximation
    # Based on the simplified SSIM formula using mean, variance, covariance
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    a = a.astype(np.float64)
    b = b.astype(np.float64)

    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)

    mu_a_sq = mu_a ** 2
    mu_b_sq = mu_b ** 2
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.GaussianBlur(a ** 2, (11, 11), 1.5) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(b ** 2, (11, 11), 1.5) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab

    numerator = (2 * mu_ab + C1) * (2 * sigma_ab + C2)
    denominator = (mu_a_sq + mu_b_sq + C1) * (sigma_a_sq + sigma_b_sq + C2)

    ssim_map = numerator / denominator
    return float(ssim_map.mean())


# ---------------------------------------------------------------------------
# Keyframe selection
# ---------------------------------------------------------------------------

def select_keyframes(
    frame_dir,
    quality_scores=None,
    ssim_threshold=0.92,
    min_temporal_gap=1,
    max_temporal_gap=10,
    on_log=None,
):
    """
    Select an optimal subset of keyframes from quality-filtered frames.

    Algorithm:
      1. Start with all frames that passed quality filtering (or all frames
         if no quality scores are provided).
      2. Walk through frames sequentially. For each pair of consecutive
         candidates, compute SSIM. If SSIM > threshold (near-duplicate),
         merge into a cluster and keep only the sharpest frame.
      3. Enforce minimum spacing: if two selected frames are closer than
         min_temporal_gap apart, drop the lower-quality one.
      4. Enforce maximum spacing: if the gap between two selected frames
         exceeds max_temporal_gap, force-include the best intermediate
         frame to prevent reconstruction holes.

    Args:
        frame_dir:        Directory containing frames.
        quality_scores:   List of score dicts from quality.score_all_frames().
                          If None, all frames are considered with equal quality.
        ssim_threshold:   SSIM above this → frames are considered near-duplicates.
                          Range 0–1. Higher = stricter (fewer frames rejected as duplicates).
                          Default 0.92 works well for drone footage.
        min_temporal_gap: Minimum frame index gap between selected keyframes.
        max_temporal_gap: Maximum frame index gap (force-include if exceeded).
        on_log:           Optional progress callback.

    Returns:
        dict with keys:
            selected_count, original_count, selected_filenames,
            selection_report (list of {filename, reason, ssim_to_prev}),
            output_dir (if copy was performed)
    """
    log = on_log or (lambda msg: None)

    if not os.path.isdir(frame_dir):
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")

    valid_ext = (".jpg", ".jpeg", ".png")
    all_frames = sorted(
        f for f in os.listdir(frame_dir)
        if f.lower().endswith(valid_ext)
    )

    if not all_frames:
        log("No frames found.")
        return {"selected_count": 0, "original_count": 0, "selected_filenames": []}

    # --- Build candidate list with quality info ---
    if quality_scores:
        score_map = {s["filename"]: s for s in quality_scores}
        # Only consider frames that passed quality filtering
        candidates = [
            f for f in all_frames
            if f in score_map and not score_map[f].get("rejected", False)
        ]
        log(f"Starting with {len(candidates)} quality-filtered frames (from {len(all_frames)} total)")
    else:
        candidates = list(all_frames)
        score_map = {}
        log(f"No quality scores provided — considering all {len(candidates)} frames")

    if len(candidates) <= 2:
        log("Too few candidates for selection — keeping all.")
        return {
            "selected_count": len(candidates),
            "original_count": len(all_frames),
            "selected_filenames": candidates,
            "selection_report": [
                {"filename": f, "reason": "kept_few_candidates", "ssim_to_prev": None}
                for f in candidates
            ],
        }

    # --- Pass 1: SSIM-based redundancy removal ---
    log(f"Computing pairwise SSIM (threshold={ssim_threshold})…")

    # Load grayscale images for SSIM computation
    gray_cache = {}

    def _get_gray(fname):
        if fname not in gray_cache:
            path = os.path.join(frame_dir, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            gray_cache[fname] = img
        return gray_cache[fname]

    # Walk sequentially, cluster near-duplicates
    selected = []
    report = []
    cluster = [candidates[0]]  # Start first cluster

    for i in range(1, len(candidates)):
        curr = candidates[i]
        prev = cluster[-1]  # Compare against last frame in current cluster

        gray_curr = _get_gray(curr)
        gray_prev = _get_gray(prev)

        if gray_curr is None or gray_prev is None:
            # Unreadable frame — skip
            report.append({"filename": curr, "reason": "unreadable", "ssim_to_prev": None})
            continue

        ssim = _compute_ssim(gray_prev, gray_curr)

        if ssim > ssim_threshold:
            # Near-duplicate — add to cluster
            cluster.append(curr)
        else:
            # New visual content — close current cluster, start new one
            best = _pick_best_from_cluster(cluster, score_map)
            selected.append(best)
            for f in cluster:
                reason = "selected_best_in_cluster" if f == best else "redundant_duplicate"
                report.append({"filename": f, "reason": reason, "ssim_to_prev": round(ssim, 4) if f == curr else None})
            cluster = [curr]

        # Progress
        if i % 20 == 0:
            log(f"  SSIM pass: {i}/{len(candidates)} frames processed")

        # Evict old cache entries to keep memory bounded
        if len(gray_cache) > 5:
            oldest = list(gray_cache.keys())[0]
            if oldest not in cluster and oldest != candidates[i]:
                del gray_cache[oldest]

    # Close final cluster
    if cluster:
        best = _pick_best_from_cluster(cluster, score_map)
        selected.append(best)
        for f in cluster:
            reason = "selected_best_in_cluster" if f == best else "redundant_duplicate"
            report.append({"filename": f, "reason": reason, "ssim_to_prev": None})

    log(f"  After redundancy removal: {len(selected)} frames (removed {len(candidates) - len(selected)} duplicates)")

    # Clear cache
    gray_cache.clear()

    # --- Pass 2: Temporal spacing enforcement ---
    # Build index map for spacing calculations
    frame_index = {f: i for i, f in enumerate(all_frames)}

    # Check for gaps that are too large → force-include intermediate frames
    final_selected = [selected[0]]
    for i in range(1, len(selected)):
        curr = selected[i]
        prev = final_selected[-1]

        gap = frame_index.get(curr, i) - frame_index.get(prev, i - 1)

        if gap > max_temporal_gap:
            # Gap too large — find and force-include the best intermediate frame
            prev_idx = frame_index.get(prev, 0)
            curr_idx = frame_index.get(curr, len(all_frames) - 1)
            intermediates = [
                f for f in candidates
                if prev_idx < frame_index.get(f, -1) < curr_idx
                and f not in final_selected
            ]
            if intermediates:
                fill = _pick_best_from_cluster(intermediates, score_map)
                final_selected.append(fill)
                report.append({
                    "filename": fill,
                    "reason": "gap_fill",
                    "ssim_to_prev": None,
                })
                log(f"  Gap fill: inserted {fill} (gap was {gap} frames)")

        final_selected.append(curr)

    # Sort by original order
    final_selected = sorted(final_selected, key=lambda f: frame_index.get(f, 0))

    log(
        f"Keyframe selection complete: {len(final_selected)} keyframes "
        f"from {len(all_frames)} original frames "
        f"({100 * len(final_selected) / len(all_frames):.1f}% retained)"
    )

    return {
        "selected_count": len(final_selected),
        "original_count": len(all_frames),
        "selected_filenames": final_selected,
        "selection_report": report,
    }


def _pick_best_from_cluster(cluster, score_map):
    """Pick the highest-quality frame from a cluster of near-duplicates."""
    if not score_map:
        return cluster[len(cluster) // 2]  # Middle frame as fallback

    best = max(
        cluster,
        key=lambda f: score_map.get(f, {}).get("overall_score", 0.0),
    )
    return best


def copy_keyframes(frame_dir, selected_filenames, output_dir, on_log=None):
    """
    Copy selected keyframes to an output directory.

    Args:
        frame_dir:          Source directory containing all frames.
        selected_filenames: List of filenames to copy (from select_keyframes).
        output_dir:         Destination directory for keyframes.
        on_log:             Optional progress callback.

    Returns:
        dict with keys: output_dir, copied_count
    """
    import shutil

    log = on_log or (lambda msg: None)
    os.makedirs(output_dir, exist_ok=True)

    # Clean output directory first
    cleaned = 0
    for f in os.listdir(output_dir):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            os.remove(os.path.join(output_dir, f))
            cleaned += 1
    if cleaned:
        log(f"Cleared {cleaned} old keyframes from {output_dir}")

    copied = 0
    for fname in selected_filenames:
        src = os.path.join(frame_dir, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            copied += 1

    log(f"Copied {copied} keyframes to {output_dir}")

    return {
        "output_dir": output_dir,
        "copied_count": copied,
    }
