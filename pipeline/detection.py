"""
YOLO-based object detection for pipeline integration.

Wraps YOLO inference as a clean, importable pipeline function with:
  - Configurable model path
  - Dynamic-class filtering (person, car, truck, etc.)
  - Per-frame structured detection output
  - Progress callbacks

The interface is designed so the YOLO model can be swapped for an
improved version without changing downstream code. Any model that
produces bounding boxes with class_name and confidence will work.

This module preserves detection/detect.py unchanged for backward compat.
"""

import os
import json


# Classes considered "dynamic" (likely to move between frames).
# Points from these objects should be masked before reconstruction.
# This list covers COCO classes that represent movable objects.
DYNAMIC_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
}

# Classes considered "static" (part of the permanent scene).
# Detections of these classes are recorded but NOT masked.
STATIC_CLASSES = {
    "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "cell phone", "book", "clock", "vase",
}


def detect_objects(
    image_dir,
    model_path=None,
    confidence=0.35,
    dynamic_classes=None,
    on_log=None,
):
    """
    Run YOLO object detection on all images in a directory.

    Args:
        image_dir:       Directory containing input images.
        model_path:      Path to YOLO model weights (.pt).
                         If None, auto-detects from project models/ dir.
        confidence:      Minimum detection confidence threshold (0–1).
        dynamic_classes: Set of class names to treat as dynamic objects.
                         If None, uses the default DYNAMIC_CLASSES set.
        on_log:          Optional progress callback.

    Returns:
        dict with keys:
            detections: list of per-frame detection dicts
            dynamic_count: total dynamic object detections
            static_count: total static/other object detections
            frames_with_dynamic: number of frames containing dynamic objects
            class_summary: {class_name: count} for all detected classes
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
        dynamic_classes = DYNAMIC_CLASSES

    # --- Lazy import (avoids loading ultralytics at module import time) ---
    log("Loading YOLO model…")
    from ultralytics import YOLO
    import cv2

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
        }

    log(f"Running detection on {len(image_files)} frames…")

    # --- Run detection ---
    all_detections = []
    class_summary = {}
    dynamic_total = 0
    static_total = 0
    frames_with_dynamic = 0

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

        results = model.predict(frame, conf=confidence, verbose=False)
        result = results[0]

        frame_objects = []
        frame_has_dynamic = False

        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[class_id]

                is_dynamic = class_name.lower() in dynamic_classes

                detection = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(conf, 4),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
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

        if frame_has_dynamic:
            frames_with_dynamic += 1

        all_detections.append({
            "filename": fname,
            "frame_index": idx,
            "objects": frame_objects,
            "has_dynamic": frame_has_dynamic,
        })

        # Progress reporting
        if (idx + 1) % 10 == 0:
            log(f"  Detected {idx + 1}/{len(image_files)} frames "
                f"({dynamic_total} dynamic objects so far)")

    log(
        f"Detection complete: {dynamic_total} dynamic + {static_total} other "
        f"objects across {frames_with_dynamic}/{len(image_files)} frames"
    )

    return {
        "detections": all_detections,
        "dynamic_count": dynamic_total,
        "static_count": static_total,
        "frames_with_dynamic": frames_with_dynamic,
        "class_summary": class_summary,
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
