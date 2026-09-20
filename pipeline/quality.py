"""
Frame quality assessment for drone video frames.

Scores each frame on two independent axes:
  1. Sharpness — Laplacian variance (higher = sharper)
  2. Exposure  — histogram spread and clipping ratio

Frames that fail either check are marked as rejected.
The surviving frames are passed to the keyframe selector.

This module uses only OpenCV and NumPy (already in the stack).
"""

import os
import json
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Individual quality metrics
# ---------------------------------------------------------------------------

def _sharpness_score(gray):
    """
    Laplacian variance: a standard no-reference blur metric.

    The Laplacian highlights edges. A sharp image has many strong edges →
    high variance. A blurry image has weak/smeared edges → low variance.

    Returns:
        float — Laplacian variance (unbounded, higher is sharper)
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def _exposure_score(gray):
    """
    Assess exposure quality from the grayscale histogram.

    Returns a dict with:
        - mean_brightness: 0–255 (ideal ~100–170 for outdoor drone imagery)
        - clipping_dark:   fraction of pixels at 0–5 (underexposed)
        - clipping_bright: fraction of pixels at 250–255 (overexposed)
        - histogram_spread: std of pixel intensities (low = flat/washed image)
    """
    total_pixels = gray.size
    mean_brightness = float(gray.mean())
    std_brightness = float(gray.std())

    # Clipping: fraction of pixels pinned to extreme values
    clipping_dark = float(np.count_nonzero(gray <= 5) / total_pixels)
    clipping_bright = float(np.count_nonzero(gray >= 250) / total_pixels)

    return {
        "mean_brightness": round(mean_brightness, 2),
        "histogram_spread": round(std_brightness, 2),
        "clipping_dark": round(clipping_dark, 4),
        "clipping_bright": round(clipping_bright, 4),
    }


# ---------------------------------------------------------------------------
# Combined quality scoring
# ---------------------------------------------------------------------------

def score_frame(image_path):
    """
    Compute quality metrics for a single frame.

    Args:
        image_path: Absolute path to the image file.

    Returns:
        dict with keys:
            filename, sharpness, exposure {mean_brightness, histogram_spread,
            clipping_dark, clipping_bright}, overall_score, rejected, reject_reasons
    """
    img = cv2.imread(image_path)
    if img is None:
        return {
            "filename": os.path.basename(image_path),
            "sharpness": 0.0,
            "exposure": {},
            "overall_score": 0.0,
            "rejected": True,
            "reject_reasons": ["unreadable_image"],
        }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sharpness = _sharpness_score(gray)
    exposure = _exposure_score(gray)

    return {
        "filename": os.path.basename(image_path),
        "sharpness": round(sharpness, 2),
        "exposure": exposure,
        # overall_score and rejection are set later by score_all_frames
        # using adaptive thresholds across the full set
        "overall_score": 0.0,
        "rejected": False,
        "reject_reasons": [],
    }


def score_all_frames(
    frame_dir,
    blur_reject_percentile=20,
    max_clipping_ratio=0.05,
    on_log=None,
):
    """
    Score all frames in a directory and apply adaptive rejection thresholds.

    The blur threshold is adaptive: we reject the bottom N-th percentile
    of sharpness scores (default: bottom 20%). This works across different
    cameras and scenes without requiring a fixed threshold.

    Exposure rejection uses an absolute clipping threshold: frames with
    more than 5% of pixels clipped to pure black or pure white are rejected.

    Args:
        frame_dir:              Directory containing extracted frames.
        blur_reject_percentile: Bottom percentile of sharpness to reject (0–100).
        max_clipping_ratio:     Maximum allowed fraction of clipped pixels.
        on_log:                 Optional progress callback.

    Returns:
        list of score dicts (one per frame), sorted by filename.
        Each dict includes: filename, sharpness, exposure, overall_score,
        rejected, reject_reasons.
    """
    log = on_log or (lambda msg: None)

    if not os.path.isdir(frame_dir):
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")

    valid_ext = (".jpg", ".jpeg", ".png")
    frame_files = sorted(
        f for f in os.listdir(frame_dir) if f.lower().endswith(valid_ext)
    )

    if not frame_files:
        log("No frames found in directory.")
        return []

    log(f"Scoring {len(frame_files)} frames for quality…")

    # --- Score each frame ---
    scores = []
    for i, fname in enumerate(frame_files):
        path = os.path.join(frame_dir, fname)
        score = score_frame(path)
        scores.append(score)

        if (i + 1) % 20 == 0:
            log(f"  Scored {i + 1}/{len(frame_files)} frames")

    # --- Compute adaptive blur threshold ---
    sharpness_values = [s["sharpness"] for s in scores if s["sharpness"] > 0]
    if sharpness_values:
        blur_threshold = float(np.percentile(sharpness_values, blur_reject_percentile))
    else:
        blur_threshold = 0.0

    log(f"  Blur threshold (bottom {blur_reject_percentile}%): {blur_threshold:.2f}")

    # --- Apply rejection criteria ---
    rejected_count = 0
    for s in scores:
        reasons = list(s.get("reject_reasons", []))

        # Blur rejection
        if s["sharpness"] < blur_threshold and s["sharpness"] > 0:
            reasons.append("blurry")

        # Exposure rejection
        exp = s.get("exposure", {})
        total_clipping = exp.get("clipping_dark", 0) + exp.get("clipping_bright", 0)
        if total_clipping > max_clipping_ratio:
            reasons.append("exposure_clipped")

        s["reject_reasons"] = reasons
        s["rejected"] = len(reasons) > 0

        if s["rejected"]:
            rejected_count += 1

        # Compute a normalized overall score (0–1) for ranking
        # Sharpness dominates since it's the most impactful quality factor
        if sharpness_values:
            max_sharp = max(sharpness_values)
            min_sharp = min(sharpness_values)
            range_sharp = max_sharp - min_sharp if max_sharp > min_sharp else 1.0
            norm_sharpness = (s["sharpness"] - min_sharp) / range_sharp
        else:
            norm_sharpness = 0.0

        # Exposure score: penalize clipping, reward moderate brightness
        exposure_penalty = min(1.0, total_clipping / max_clipping_ratio)
        brightness = exp.get("mean_brightness", 128)
        # Ideal brightness range: 80–180 for outdoor drone
        brightness_score = 1.0 - min(1.0, max(0.0, abs(brightness - 130) / 130))

        s["overall_score"] = round(
            0.7 * norm_sharpness + 0.2 * brightness_score + 0.1 * (1.0 - exposure_penalty),
            4,
        )

    log(
        f"Quality scoring complete: {rejected_count}/{len(scores)} frames rejected "
        f"({len(scores) - rejected_count} passed)"
    )

    return scores


def filter_frames(frame_dir, scores, output_dir=None, on_log=None):
    """
    Copy or symlink passing frames into a filtered output directory.

    If output_dir is None, returns only the list of passing filenames
    without copying (useful when the keyframe selector handles file I/O).

    Args:
        frame_dir:   Source directory containing all frames.
        scores:      Quality scores from score_all_frames().
        output_dir:  Optional output directory for filtered frames.
        on_log:      Optional progress callback.

    Returns:
        dict with keys:
            passed_count, rejected_count, passed_filenames, quality_report_path
    """
    log = on_log or (lambda msg: None)

    passed = [s for s in scores if not s["rejected"]]
    rejected = [s for s in scores if s["rejected"]]

    passed_filenames = [s["filename"] for s in passed]

    if output_dir:
        import shutil

        os.makedirs(output_dir, exist_ok=True)
        for fname in passed_filenames:
            src = os.path.join(frame_dir, fname)
            dst = os.path.join(output_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)

        log(f"Copied {len(passed_filenames)} quality-filtered frames to {output_dir}")

        # Save quality report alongside filtered frames
        report_path = os.path.join(output_dir, "quality_report.json")
        report = {
            "total_frames": len(scores),
            "passed": len(passed),
            "rejected": len(rejected),
            "blur_reject_reasons": sum(1 for s in rejected if "blurry" in s["reject_reasons"]),
            "exposure_reject_reasons": sum(
                1 for s in rejected if "exposure_clipped" in s["reject_reasons"]
            ),
            "scores": scores,
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        log(f"Quality report saved: {report_path}")
    else:
        report_path = None

    return {
        "passed_count": len(passed),
        "rejected_count": len(rejected),
        "passed_filenames": passed_filenames,
        "quality_report_path": report_path,
    }
