"""
DUSt3R multi-view 3D reconstruction.
Adapted from test_multiview.py — original script is preserved unchanged.

V3 upgrades:
  - Higher resolution (768) when GPU memory allows
  - Denser scene graph (swin-5) for better connectivity
  - Confidence-weighted point filtering
  - More alignment iterations (500)
  - GPS-based metric scale factor extraction (camera centers for scale.py)

NOTE: This module has heavy imports (PyTorch, DUSt3R). They are lazily loaded
inside the function so the server process can import this module without
loading the model into memory.
"""

import os
import sys
import json
import cv2
import numpy as np


def _resize_mask_like_dust3r(mask_bool, target_shape, image_size=512, patch_size=16):
    """
    Resize a full-resolution binary mask to match DUSt3R's internal
    dimensions for a specific view.

    DUSt3R's load_images does NOT simply resize to (image_size, image_size).
    It does:
      1. Resize the long edge to image_size (preserving aspect ratio)
      2. Center-crop to patch-aligned dimensions (16px grid)
      3. For non-square images: force 4:3 ratio (halfh = 3*halfw/4)

    This function replicates that exact transform on the mask so the
    masked regions align pixel-perfectly with the DUSt3R point cloud.

    Args:
        mask_bool: Boolean numpy array (H_orig, W_orig), True=include.
        target_shape: (H_dust3r, W_dust3r) — actual shape from scene.get_masks()[i].
        image_size: DUSt3R image_size parameter (default 512).
        patch_size: DUSt3R patch grid size (default 16).

    Returns:
        Boolean numpy array of shape target_shape, or None if resize fails.
    """
    H_orig, W_orig = mask_bool.shape
    H_target, W_target = target_shape

    # Convert bool mask to uint8 for cv2 operations
    mask_u8 = mask_bool.astype(np.uint8) * 255

    # Step 1: Resize long edge to image_size, preserving aspect ratio
    S = max(H_orig, W_orig)
    new_W = int(round(W_orig * image_size / S))
    new_H = int(round(H_orig * image_size / S))
    mask_resized = cv2.resize(mask_u8, (new_W, new_H), interpolation=cv2.INTER_NEAREST)

    # Step 2: Center crop to patch-aligned dims (replicating DUSt3R logic)
    cx, cy = new_W // 2, new_H // 2
    halfw = ((2 * cx) // patch_size) * patch_size / 2
    halfh = ((2 * cy) // patch_size) * patch_size / 2

    # Step 3: For non-square images, DUSt3R forces 4:3 aspect ratio
    if new_W != new_H:
        halfh = 3 * halfw / 4

    x1 = int(cx - halfw)
    y1 = int(cy - halfh)
    x2 = int(cx + halfw)
    y2 = int(cy + halfh)

    # Clamp to valid range
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(new_W, x2)
    y2 = min(new_H, y2)

    mask_cropped = mask_resized[y1:y2, x1:x2]

    # Final check: if dimensions still don't match (rounding), resize
    if mask_cropped.shape[0] != H_target or mask_cropped.shape[1] != W_target:
        mask_cropped = cv2.resize(
            mask_cropped, (W_target, H_target),
            interpolation=cv2.INTER_NEAREST
        )

    return mask_cropped > 127


def _auto_select_image_size(device, default=512, high=768):
    """
    Auto-select DUSt3R inference resolution based on available GPU memory.
    Uses 768 if ≥8GB VRAM is available, otherwise falls back to 512.
    """
    try:
        import torch
        if device.type == "cuda":
            total_mem = torch.cuda.get_device_properties(device).total_mem
            free_mem = total_mem - torch.cuda.memory_allocated(device)
            # 768 resolution needs ~6GB for inference; 8GB gives safe margin
            if free_mem >= 8 * 1024 * 1024 * 1024:
                return high
    except Exception:
        pass
    return default


def reconstruct_point_cloud(
    image_dir,
    output_ply,
    model_path=None,
    device=None,
    image_size=None,
    batch_size=1,
    scene_graph=None,
    alignment_iters=None,
    mask_dir=None,
    colmap_poses_json=None,
    confidence_threshold=None,
    on_log=None,
):
    """
    Run DUSt3R multi-view stereo reconstruction on a folder of images.

    V3 upgrades:
      - Auto-selects higher resolution (768) when GPU memory allows
      - Uses denser scene graph (swin-5) by default
      - Confidence-based point filtering
      - Extracts camera centers for GPS scale factor computation
      - Returns camera_centers for pipeline.scale

    Args:
        image_dir: Directory containing input images (.jpg, .png).
        output_ply: Path for the output colored point cloud (.ply).
        model_path: Path to DUSt3R checkpoint. Auto-detected if None.
        device: 'cuda' or 'cpu'. Auto-detected if None.
        image_size: Image resize target for inference. Auto-detected if None.
        batch_size: Inference batch size (default 1 for safe VRAM).
        scene_graph: DUSt3R scene graph type. Default from config.
        alignment_iters: Global alignment iterations. Default from config.
        mask_dir: Optional directory containing per-frame binary masks
                  (from pipeline.masking). Pixels marked as dynamic objects
                  are excluded from the output point cloud. DUSt3R still
                  runs inference on full images for context.
        colmap_poses_json: Optional path to COLMAP poses JSON file
                          (from pipeline.colmap). When available, COLMAP camera
                          poses are loaded for comparison and future pose-guided
                          reconstruction (Option B).
        confidence_threshold: DUSt3R confidence threshold for point filtering.
                             Points below this confidence are excluded.
                             Default from config.
        on_log: Optional progress callback.

    Returns:
        dict with keys: point_count, output_ply, masked_points (if masks used),
                        colmap_registered (if COLMAP poses available),
                        camera_centers (list of {filename, x, y, z}),
                        dimensions (bounding box in reconstruction units)
    """
    log = on_log or (lambda msg: None)

    from pipeline.config import ProjectConfig
    config = ProjectConfig()

    # --- Resolve defaults from config ---
    if scene_graph is None:
        scene_graph = config.DEFAULT_SCENE_GRAPH
    if alignment_iters is None:
        alignment_iters = config.DEFAULT_ALIGNMENT_ITERS
    if confidence_threshold is None:
        confidence_threshold = config.DEFAULT_CONFIDENCE_THRESHOLD

    # --- Resolve paths ---
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if model_path is None:
        model_path = os.path.join(
            project_dir, "checkpoints", "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
        )
    d3r_path = os.path.join(project_dir, "dust3r")

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"DUSt3R model not found: {model_path}. Run download_model.py first."
        )

    # --- Load COLMAP poses if available ---
    colmap_data = None
    if colmap_poses_json and os.path.isfile(colmap_poses_json):
        log(f"Loading COLMAP poses from {colmap_poses_json}…")
        with open(colmap_poses_json, "r", encoding="utf-8") as f:
            colmap_data = json.load(f)
        registered = colmap_data.get("registered_images", 0)
        total = colmap_data.get("total_images", 0)
        log(f"  COLMAP registered {registered}/{total} images")
        log(f"  Camera models: {len(colmap_data.get('cameras', {}))}")

    # --- Setup sys.path for DUSt3R imports ---
    if d3r_path not in sys.path:
        sys.path.insert(0, d3r_path)
    croco_path = os.path.join(d3r_path, "croco")
    if croco_path not in sys.path:
        sys.path.insert(0, croco_path)

    # --- Lazy imports (heavy) ---
    log("Loading PyTorch and DUSt3R…")
    import argparse
    import torch
    import numpy as np
    import trimesh

    torch.serialization.add_safe_globals([argparse.Namespace])

    # pyrefly: ignore [missing-import]
    from dust3r.model import AsymmetricCroCo3DStereo
    # pyrefly: ignore [missing-import]
    from dust3r.inference import inference
    # pyrefly: ignore [missing-import]
    from dust3r.utils.image import load_images
    # pyrefly: ignore [missing-import]
    from dust3r.image_pairs import make_pairs
    # pyrefly: ignore [missing-import]
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

    # --- Device selection ---
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    log(f"Device: {device}")

    # --- Auto-select image size (V3) ---
    if image_size is None:
        image_size = _auto_select_image_size(
            device, config.DEFAULT_IMAGE_SIZE, config.DEFAULT_IMAGE_SIZE_HIGH
        )
    log(f"Inference resolution: {image_size}px")
    log(f"Scene graph: {scene_graph}")
    log(f"Alignment iterations: {alignment_iters}")

    # --- Discover images ---
    valid_ext = (".jpg", ".jpeg", ".png")
    image_paths = sorted(
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(valid_ext)
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")
    log(f"Found {len(image_paths)} images.")

    # --- Load model ---
    log("Loading DUSt3R model (this may take a moment)…")
    model = AsymmetricCroCo3DStereo.from_pretrained(
        model_path, img_size=(image_size, image_size)
    )
    model.to(device)
    model.eval()
    log("Model loaded.")

    # --- Load images ---
    log("Loading images…")
    images = load_images(image_paths, size=image_size)
    log(f"Images loaded: {len(images)}")

    # --- Create pairs ---
    log("Creating image pairs…")
    pairs = make_pairs(images, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    log(f"Pairs: {len(pairs)}")

    # --- Inference ---
    log("Running DUSt3R inference (GPU-intensive)…")
    output = inference(pairs, model, device=device, batch_size=batch_size, verbose=True)
    log("Inference complete.")

    # --- Global alignment ---
    log("Starting global alignment…")
    scene = global_aligner(
        output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer
    )
    log(f"Optimizing global scene ({alignment_iters} iterations)…")
    scene.compute_global_alignment(
        init="mst", niter=alignment_iters, schedule="cosine", lr=0.01
    )
    log("Global alignment complete.")

    # --- Extract camera centers for scale factor computation (V3) ---
    camera_centers = []
    try:
        im_poses = scene.get_im_poses()
        for idx, img_path in enumerate(image_paths):
            fname = os.path.basename(img_path)
            pose_mat = im_poses[idx].detach().cpu().numpy()
            # Camera center = translation column of the pose matrix (4x4)
            center = pose_mat[:3, 3]
            camera_centers.append({
                "filename": fname,
                "x": float(center[0]),
                "y": float(center[1]),
                "z": float(center[2]),
            })
        log(f"Extracted {len(camera_centers)} camera centers for scale computation")
    except Exception as e:
        log(f"Warning: Could not extract camera centers: {e}")

    # --- Save COLMAP pose comparison data alongside output ---
    if colmap_data is not None:
        colmap_comparison_path = os.path.join(
            os.path.dirname(output_ply), "pose_comparison.json"
        )
        try:
            # Extract DUSt3R poses for comparison
            dust3r_poses = []
            for idx, img_path in enumerate(image_paths):
                fname = os.path.basename(img_path)
                pose_mat = im_poses[idx].detach().cpu().numpy()
                dust3r_poses.append({
                    "filename": fname,
                    "dust3r_pose": pose_mat.tolist(),
                })

            comparison = {
                "dust3r_poses": dust3r_poses,
                "colmap_poses": colmap_data.get("poses", {}),
                "colmap_cameras": colmap_data.get("cameras", {}),
            }
            with open(colmap_comparison_path, "w", encoding="utf-8") as f:
                json.dump(comparison, f, indent=2)
            log(f"Pose comparison data saved: {colmap_comparison_path}")
        except Exception as e:
            log(f"Warning: Could not save pose comparison: {e}")


    # --- Extract colored point cloud ---
    log("Extracting colored point cloud…")
    imgs = scene.imgs
    pts3d = scene.get_pts3d()
    masks = scene.get_masks()

    # --- V3: Confidence-based filtering ---
    confidences = None
    try:
        # DUSt3R provides confidence maps via get_conf()
        confidences = scene.get_conf()
        log(f"Confidence filtering enabled (threshold: {confidence_threshold})")
    except (AttributeError, Exception):
        log("Confidence maps not available — skipping confidence filtering")

    # Pre-load original-resolution masks for DUSt3R-aligned resizing
    # We must replicate DUSt3R's load_images resize logic on the mask:
    #   1. Resize long edge to image_size (preserving aspect ratio)
    #   2. Center-crop to patch-aligned dimensions (16px grid)
    #   3. For non-square images, force 4:3 ratio (halfh = 3*halfw/4)
    mask_load_fn = None
    if mask_dir and os.path.isdir(mask_dir):
        from pipeline.masking import load_mask_for_frame
        log(f"Loading dynamic object masks from {mask_dir}…")
        mask_load_fn = load_mask_for_frame

    all_points = []
    all_colors = []
    total_masked_points = 0
    total_low_confidence = 0

    for i in range(len(pts3d)):
        points = pts3d[i].detach().cpu().numpy()
        mask = masks[i].detach().cpu().numpy()
        image = imgs[i]

        if hasattr(image, "detach"):
            image = image.detach().cpu().numpy()
        image = np.asarray(image)

        if image.ndim == 4:
            image = image[0]
        if image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)

        # V3: Apply confidence threshold
        if confidences is not None:
            conf_map = confidences[i].detach().cpu().numpy()
            low_conf_mask = conf_map < confidence_threshold
            count_before = int(mask.sum())
            mask = mask & (~low_conf_mask)
            excluded = count_before - int(mask.sum())
            if excluded > 0:
                total_low_confidence += excluded

        # Apply dynamic object mask — resize to match DUSt3R's ACTUAL shape
        if mask_load_fn is not None and i < len(image_paths):
            fname = os.path.basename(image_paths[i])
            # Load mask at original resolution (no target_size)
            dm_orig = mask_load_fn(mask_dir, fname, target_size=None)
            if dm_orig is not None:
                # Replicate DUSt3R's exact resize + crop on the mask
                dm = _resize_mask_like_dust3r(
                    dm_orig, mask.shape, image_size, patch_size=16
                )
                if dm is not None and dm.shape == mask.shape:
                    before_count = int(mask.sum())
                    mask = mask & dm
                    excluded = before_count - int(mask.sum())
                    if excluded > 0:
                        total_masked_points += excluded

        points = points[mask]
        colors = image[mask]
        valid = np.isfinite(points).all(axis=1)
        points = points[valid]
        colors = colors[valid]

        all_points.append(points)
        all_colors.append(colors)

    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    # Flip point cloud bottom-to-top (negate Z axis)
    all_points[:, 2] *= 1

    if total_masked_points > 0:
        log(f"Excluded {total_masked_points} dynamic-object points")
    if total_low_confidence > 0:
        log(f"Excluded {total_low_confidence} low-confidence points")
    log(f"Total colored points: {len(all_points)}")

    # --- Compute bounding box dimensions (V3) ---
    from pipeline.scale import compute_dimensions_from_points
    dimensions = compute_dimensions_from_points(all_points, scale_factor=1.0)
    log(
        f"Bounding box (reconstruction units): "
        f"{dimensions['width_m']:.2f} × {dimensions['height_m']:.2f} × {dimensions['depth_m']:.2f}"
    )

    # --- Export PLY ---
    os.makedirs(os.path.dirname(output_ply) or ".", exist_ok=True)
    cloud = trimesh.PointCloud(all_points, colors=all_colors)
    cloud.export(output_ply)
    log(f"Point cloud saved: {output_ply}")

    # --- Clean up GPU memory ---
    del model, scene, output, pairs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "point_count": len(all_points),
        "output_ply": output_ply,
        "masked_points": total_masked_points,
        "low_confidence_points": total_low_confidence,
        "camera_centers": camera_centers,
        "dimensions_raw": dimensions,
    }
    if colmap_data is not None:
        result["colmap_registered"] = colmap_data.get("registered_images", 0)
    return result
