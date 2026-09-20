"""
Motion-compensated object detection and tracking for pipeline integration.

Uses YOLO + ByteTrack for persistent object tracking across keyframes,
combined with ORB feature matching + homography estimation to compensate
for camera ego-motion. Objects are classified as MOVING or STATIC based
on their residual pixel movement after camera compensation.

This is a significant upgrade over naive class-based filtering:
  - A parked car → STATIC → kept in reconstruction
  - A moving car → MOVING → masked from reconstruction
  - A walking person → MOVING → masked from reconstruction
  - A person sitting on a bench → STATIC → kept in reconstruction

The module preserves detection/detect.py and detection/track.py unchanged
as standalone tools. This module wraps the same logic as an importable
pipeline function with configurable parameters and progress callbacks.

Core tracking logic adapted from detection/track.py by team member.
"""

import os
import json
import math
import cv2
import numpy as np


# Classes considered "dynamic" by nature (used as FALLBACK only when
# camera motion compensation is unavailable or tracking fails).
# When motion compensation works, we use actual movement data instead.
DYNAMIC_CLASSES_FALLBACK = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
}


# ---------------------------------------------------------------------------
# ORB Camera Motion Estimation
# ---------------------------------------------------------------------------

def _estimate_camera_motion(prev_gray, curr_gray, orb, bf, max_matches=100):
    """
    Estimate camera ego-motion between two frames using ORB features
    and homography estimation.

    Args:
        prev_gray: Previous frame (grayscale numpy array).
        curr_gray: Current frame (grayscale numpy array).
        orb:       cv2.ORB detector instance.
        bf:        cv2.BFMatcher instance.
        max_matches: Maximum number of matches to use.

    Returns:
        3x3 homography matrix (numpy array), or None if estimation fails.
    """
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)

    if des1 is None or des2 is None:
        return None
    if len(kp1) < 10 or len(kp2) < 10:
        return None

    matches = bf.match(des1, des2)
    if len(matches) < 10:
        return None

    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:max_matches]

    if len(good_matches) < 8:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    return H


# ---------------------------------------------------------------------------
# Main detection + tracking function
# ---------------------------------------------------------------------------

def detect_objects(
    image_dir,
    model_path=None,
    confidence=0.35,
    movement_threshold=5.0,
    orb_features=1500,
    orb_max_matches=100,
    dynamic_classes=None,
    on_log=None,
):
    """
    Run YOLO + ByteTrack detection with camera-motion-compensated
    dynamic/static classification on all images in a directory.

    Algorithm:
      1. Load images in sorted order (preserving temporal sequence).
      2. For each consecutive pair, estimate camera ego-motion via
         ORB features + RANSAC homography.
      3. Run YOLO with ByteTrack (persist=True) for persistent object IDs.
      4. For each tracked object, compute raw pixel movement from previous
         frame, then subtract camera motion to get compensated movement.
      5. Classify: compensated_movement > threshold → MOVING (is_dynamic=True)
                   compensated_movement ≤ threshold → STATIC (is_dynamic=False)
                   first appearance → NEW (is_dynamic=False, benefit of doubt)

    Args:
        image_dir:           Directory containing input images (sorted = temporal order).
        model_path:          Path to YOLO model weights (.pt).
        confidence:          Minimum detection confidence threshold (0–1).
        movement_threshold:  Compensated pixel movement above this = MOVING.
                             Default 5.0 from track.py tuning.
        orb_features:        Number of ORB features to detect per frame.
        orb_max_matches:     Maximum ORB matches for homography estimation.
        dynamic_classes:     Fallback class set (used only when motion
                             compensation fails). Default: DYNAMIC_CLASSES_FALLBACK.
        on_log:              Optional progress callback.

    Returns:
        dict with keys:
            detections: list of per-frame detection dicts
            dynamic_count: total MOVING object detections
            static_count: total STATIC + NEW detections
            frames_with_dynamic: frames containing at least one MOVING object
            class_summary: {class_name: count}
            tracking_stats: {total_tracked_ids, motion_compensated_frames, fallback_frames}
    """
    log = on_log or (lambda msg: None)

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    # Resolve model path
    if model_path is None:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_dir, "models", "D3S_YOLO_80.pt")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"YOLO model not found: {model_path}\n"
            "Place your YOLO .pt model in the models/ directory."
        )

    if dynamic_classes is None:
        dynamic_classes = DYNAMIC_CLASSES_FALLBACK

    # --- Lazy imports ---
    log("Loading YOLO model…")
    from ultralytics import YOLO

    model = YOLO(model_path)
    log(f"YOLO model loaded: {len(model.names)} classes")

    # --- Discover images ---
    valid_ext = (".jpg", ".jpeg", ".png")
    image_files = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith(valid_ext)
    )

    if not image_files:
        log("No images found.")
        return {
            "detections": [],
            "dynamic_count": 0,
            "static_count": 0,
            "frames_with_dynamic": 0,
            "class_summary": {},
            "tracking_stats": {},
        }

    log(f"Running detection + tracking on {len(image_files)} frames…")

    # --- Initialize ORB for camera motion estimation ---
    orb = cv2.ORB_create(nfeatures=orb_features)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    prev_gray = None

    # --- Tracking state ---
    previous_centers = {}  # track_id → (center_x, center_y)
    all_detections = []
    class_summary = {}
    dynamic_total = 0
    static_total = 0
    frames_with_dynamic = 0
    all_track_ids = set()
    motion_compensated_frames = 0
    fallback_frames = 0

    # --- Process each frame ---
    for idx, fname in enumerate(image_files):
        img_path = os.path.join(image_dir, fname)
        frame = cv2.imread(img_path)

        if frame is None:
            all_detections.append({
                "filename": fname,
                "frame_index": idx,
                "objects": [],
                "has_dynamic": False,
            })
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Estimate camera motion ---
        H = None
        if prev_gray is not None:
            H = _estimate_camera_motion(prev_gray, gray, orb, bf, orb_max_matches)
            if H is not None:
                motion_compensated_frames += 1
            else:
                fallback_frames += 1

        # --- YOLO + ByteTrack ---
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=confidence,
            verbose=False,
        )
        result = results[0]

        frame_objects = []
        frame_has_dynamic = False

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            for box, track_id, cls, conf in zip(boxes, track_ids, classes, confidences):
                x1, y1, x2, y2 = map(int, box)
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                class_name = model.names[int(cls)]

                # --- Motion classification ---
                motion = "NEW"
                raw_distance = 0.0
                compensated_distance = 0.0

                if track_id in previous_centers:
                    old_x, old_y = previous_centers[track_id]

                    # Raw pixel movement
                    raw_distance = math.sqrt(
                        (center_x - old_x) ** 2 + (center_y - old_y) ** 2
                    )

                    if H is not None:
                        # Camera-compensated movement
                        old_pt = np.array([[[old_x, old_y]]], dtype=np.float32)
                        predicted = cv2.perspectiveTransform(old_pt, H)
                        predicted_x = float(predicted[0, 0, 0])
                        predicted_y = float(predicted[0, 0, 1])

                        dx = center_x - predicted_x
                        dy = center_y - predicted_y
                        compensated_distance = math.sqrt(dx ** 2 + dy ** 2)

                        motion = "MOVING" if compensated_distance > movement_threshold else "STATIC"
                    else:
                        # Fallback: use raw movement when homography unavailable
                        compensated_distance = raw_distance
                        motion = "MOVING" if raw_distance > movement_threshold else "STATIC"

                # Update tracking state
                previous_centers[track_id] = (center_x, center_y)
                all_track_ids.add(int(track_id))

                # Set is_dynamic based on motion classification
                is_dynamic = (motion == "MOVING")

                detection = {
                    "class_id": int(cls),
                    "class_name": class_name,
                    "confidence": round(float(conf), 4),
                    "bbox": [x1, y1, x2, y2],
                    "center": {"x": center_x, "y": center_y},
                    "track_id": int(track_id),
                    "motion": motion,
                    "raw_pixel_movement": round(raw_distance, 3),
                    "compensated_pixel_movement": round(compensated_distance, 3),
                    "is_dynamic": is_dynamic,
                }
                frame_objects.append(detection)

                # Track counts
                class_summary[class_name] = class_summary.get(class_name, 0) + 1
                if is_dynamic:
                    dynamic_total += 1
                    frame_has_dynamic = True
                else:
                    static_total += 1

        elif result.boxes is not None:
            # ByteTrack didn't assign IDs (can happen on first frame or
            # if tracking fails). Fall back to class-based detection.
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[class_id]

                # Fallback: classify by class name
                is_dynamic = class_name.lower() in dynamic_classes

                detection = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(conf, 4),
                    "bbox": [x1, y1, x2, y2],
                    "center": {
                        "x": int((x1 + x2) / 2),
                        "y": int((y1 + y2) / 2),
                    },
                    "track_id": None,
                    "motion": "UNKNOWN",
                    "raw_pixel_movement": 0.0,
                    "compensated_pixel_movement": 0.0,
                    "is_dynamic": is_dynamic,
                }
                frame_objects.append(detection)

                class_summary[class_name] = class_summary.get(class_name, 0) + 1
                if is_dynamic:
                    dynamic_total += 1
                    frame_has_dynamic = True
                else:
                    static_total += 1

        if frame_has_dynamic:
            frames_with_dynamic += 1

        all_detections.append({
            "filename": fname,
            "frame_index": idx,
            "objects": frame_objects,
            "has_dynamic": frame_has_dynamic,
        })

        # Save grayscale for next frame's camera motion estimation
        prev_gray = gray.copy()

        # Progress
        if (idx + 1) % 10 == 0:
            log(f"  Tracked {idx + 1}/{len(image_files)} frames "
                f"({dynamic_total} moving, {static_total} static)")

    log(
        f"Detection complete: {dynamic_total} MOVING + {static_total} STATIC/NEW "
        f"across {frames_with_dynamic}/{len(image_files)} frames "
        f"({len(all_track_ids)} unique tracked objects)"
    )
    log(
        f"  Camera compensation: {motion_compensated_frames} frames with homography, "
        f"{fallback_frames} fallback"
    )

    return {
        "detections": all_detections,
        "dynamic_count": dynamic_total,
        "static_count": static_total,
        "frames_with_dynamic": frames_with_dynamic,
        "class_summary": class_summary,
        "tracking_stats": {
            "total_tracked_ids": len(all_track_ids),
            "motion_compensated_frames": motion_compensated_frames,
            "fallback_frames": fallback_frames,
            "movement_threshold": movement_threshold,
        },
    }


def save_detections(detections_result, output_path, on_log=None):
    """
    Save detection results to a JSON file.

    Args:
        detections_result: Result dict from detect_objects().
        output_path:       Path to write the JSON file.
        on_log:            Optional progress callback.

    Returns:
        str — path to the saved JSON file.
    """
    log = on_log or (lambda msg: None)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(detections_result, f, indent=2)

    log(f"Detections saved: {output_path}")
    return output_path
