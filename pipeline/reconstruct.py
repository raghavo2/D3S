"""
DUSt3R multi-view 3D reconstruction.
Adapted from test_multiview.py — original script is preserved unchanged.

NOTE: This module has heavy imports (PyTorch, DUSt3R). They are lazily loaded
inside the function so the server process can import this module without
loading the model into memory.
"""

import os
import sys
import json


def reconstruct_point_cloud(
    image_dir,
    output_ply,
    model_path=None,
    device=None,
    image_size=512,
    batch_size=1,
    scene_graph="swin-3",
    alignment_iters=300,
    on_log=None,
):
    """
    Run DUSt3R multi-view stereo reconstruction on a folder of images.

    Args:
        image_dir: Directory containing input images (.jpg, .png).
        output_ply: Path for the output colored point cloud (.ply).
        model_path: Path to DUSt3R checkpoint. Auto-detected if None.
        device: 'cuda' or 'cpu'. Auto-detected if None.
        image_size: Image resize target for inference (default 512).
        batch_size: Inference batch size (default 1 for safe VRAM).
        scene_graph: DUSt3R scene graph type (default 'swin-3').
        alignment_iters: Global alignment iterations (default 300).
        on_log: Optional progress callback.

    Returns:
        dict with keys: point_count, output_ply
    """
    log = on_log or (lambda msg: None)

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

    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

    # --- Device selection ---
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    log(f"Device: {device}")

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
    log("Optimizing global scene…")
    scene.compute_global_alignment(
        init="mst", niter=alignment_iters, schedule="cosine", lr=0.01
    )
    log("Global alignment complete.")

    # --- Extract colored point cloud ---
    log("Extracting colored point cloud…")
    imgs = scene.imgs
    pts3d = scene.get_pts3d()
    masks = scene.get_masks()

    all_points = []
    all_colors = []

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

        points = points[mask]
        colors = image[mask]
        valid = np.isfinite(points).all(axis=1)
        points = points[valid]
        colors = colors[valid]

        all_points.append(points)
        all_colors.append(colors)

    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    log(f"Total colored points: {len(all_points)}")

    # --- Export PLY ---
    os.makedirs(os.path.dirname(output_ply) or ".", exist_ok=True)
    cloud = trimesh.PointCloud(all_points, colors=all_colors)
    cloud.export(output_ply)
    log(f"Point cloud saved: {output_ply}")

    # --- Clean up GPU memory ---
    del model, scene, output, pairs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "point_count": len(all_points),
        "output_ply": output_ply,
    }
