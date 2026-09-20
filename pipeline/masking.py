"""
Dynamic object mask generation from YOLO detections.

Converts per-frame YOLO bounding boxes (for dynamic-class objects)
into binary masks that the reconstruction module uses to exclude
dynamic object pixels from the 3D point cloud.

Mask convention:
  - True  (1) = STATIC pixel → include in reconstruction
  - False (0) = DYNAMIC pixel → exclude from reconstruction

Masks are saved as single-channel 8-bit PNGs (0 or 255) for
easy inspection and debugging.
"""

import os
import cv2
import numpy as np


def generate_masks(
    image_dir,
    detections,
    output_dir,
    padding_px=10,
    on_log=None,
):
    """
    Generate binary masks from YOLO dynamic-object detections.

    For each frame, a mask is created where:
      - Pixels inside dynamic-object bounding boxes (+ padding) are 0 (exclude)
      - All other pixels are 255 (include)

    If a frame has no dynamic objects, a full-white mask is generated
    (all pixels included). This keeps the mask directory complete so
    the reconstruction module can do a simple filename lookup.

    Args:
        image_dir:   Directory containing the source images (for dimensions).
        detections:  List of per-frame detection dicts from detection.detect_objects().
                     Each dict must have: filename, objects (list of {bbox, is_dynamic}).
        output_dir:  Directory to save mask PNGs.
        padding_px:  Pixels to expand each bounding box by (catches shadows/reflections).
        on_log:      Optional progress callback.

    Returns:
        dict with keys:
            mask_count: number of masks generated
            masked_frames: number of frames with at least one masked region
            output_dir: path to mask directory
    """
    log = on_log or (lambda msg: None)
    os.makedirs(output_dir, exist_ok=True)

    if not detections:
        log("No detections provided — skipping mask generation.")
        return {"mask_count": 0, "masked_frames": 0, "output_dir": output_dir}

    # Clean old masks
    cleaned = 0
    for f in os.listdir(output_dir):
        if f.lower().endswith(".png"):
            os.remove(os.path.join(output_dir, f))
            cleaned += 1
    if cleaned:
        log(f"Cleared {cleaned} old masks.")

    masked_frames = 0

    for det in detections:
        fname = det["filename"]
        objects = det.get("objects", [])

        # Read image dimensions
        img_path = os.path.join(image_dir, fname)
        if not os.path.isfile(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        # Start with all-include mask
        mask = np.ones((h, w), dtype=np.uint8) * 255

        # Zero out dynamic object regions
        has_dynamic = False
        for obj in objects:
            if not obj.get("is_dynamic", False):
                continue

            bbox = obj["bbox"]  # [x1, y1, x2, y2]
            x1 = max(0, bbox[0] - padding_px)
            y1 = max(0, bbox[1] - padding_px)
            x2 = min(w, bbox[2] + padding_px)
            y2 = min(h, bbox[3] + padding_px)

            mask[y1:y2, x1:x2] = 0
            has_dynamic = True

        if has_dynamic:
            masked_frames += 1

        # Save mask with same stem as source image, but .png extension
        stem = os.path.splitext(fname)[0]
        mask_path = os.path.join(output_dir, f"{stem}.png")
        cv2.imwrite(mask_path, mask)

    log(
        f"Generated {len(detections)} masks "
        f"({masked_frames} frames have dynamic objects masked)"
    )

    return {
        "mask_count": len(detections),
        "masked_frames": masked_frames,
        "output_dir": output_dir,
    }


def load_mask_for_frame(mask_dir, frame_filename, target_size=None):
    """
    Load a binary mask for a specific frame.

    Used by reconstruct.py at point-extraction time to exclude
    dynamic object pixels.

    Args:
        mask_dir:        Directory containing mask PNGs.
        frame_filename:  Filename of the source frame (e.g. '00005.jpg').
        target_size:     Optional (H, W) tuple to resize the mask to.
                         Required when DUSt3R operates at a different
                         resolution than the original images.

    Returns:
        numpy array of shape (H, W) with dtype bool:
            True  = include this pixel
            False = exclude this pixel (dynamic object)
        Returns None if no mask file is found.
    """
    stem = os.path.splitext(frame_filename)[0]
    mask_path = os.path.join(mask_dir, f"{stem}.png")

    if not os.path.isfile(mask_path):
        return None

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    # Resize if needed (DUSt3R typically uses 512×512)
    if target_size is not None:
        h, w = target_size
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Convert to boolean: 255 → True (include), 0 → False (exclude)
    return mask > 127
