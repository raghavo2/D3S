"""
COLMAP sparse reconstruction as a pipeline stage.

Runs COLMAP's feature extraction → matching → sparse mapping pipeline
on keyframe images to produce accurate camera poses (intrinsics + extrinsics).

These camera poses are used by the reconstruction module (Option B):
  - COLMAP handles camera pose estimation (it's better at this)
  - DUSt3R handles dense depth estimation (it's better at this)
  - Combining both gives better results than either alone

The module wraps the existing run_colmap.py logic as an importable
function with configurable paths and progress callbacks. The original
run_colmap.py is preserved unchanged.
"""

import os
import re
import json
import shutil
import subprocess
import numpy as np


# ---------------------------------------------------------------------------
# COLMAP text model parser
# ---------------------------------------------------------------------------

def _parse_colmap_cameras(cameras_txt):
    """
    Parse COLMAP cameras.txt to extract intrinsic parameters.

    Returns:
        dict mapping camera_id (int) → {model, width, height, params}
    """
    cameras = {}
    with open(cameras_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            cameras[cam_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def _parse_colmap_images(images_txt):
    """
    Parse COLMAP images.txt to extract camera extrinsics (poses).

    COLMAP stores camera-to-world transforms as:
      QW QX QY QZ TX TY TZ (quaternion + translation)

    Returns:
        dict mapping image_name → {
            image_id, camera_id, qvec (4,), tvec (3,),
            rotation_matrix (3x3), camera_center (3,)
        }
    """
    images = {}
    with open(images_txt, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    # images.txt has pairs of lines: pose line, then 2D points line
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        if len(parts) < 10:
            continue

        image_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        camera_id = int(parts[8])
        image_name = parts[9]

        # Quaternion to rotation matrix
        qvec = np.array([qw, qx, qy, qz])
        R = _qvec_to_rotmat(qvec)

        # Camera center in world coordinates: C = -R^T * t
        tvec = np.array([tx, ty, tz])
        camera_center = -R.T @ tvec

        images[image_name] = {
            "image_id": image_id,
            "camera_id": camera_id,
            "qvec": qvec.tolist(),
            "tvec": tvec.tolist(),
            "rotation_matrix": R.tolist(),
            "camera_center": camera_center.tolist(),
        }

    return images


def _qvec_to_rotmat(qvec):
    """Convert COLMAP quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = qvec
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])
    return R


# ---------------------------------------------------------------------------
# COLMAP execution
# ---------------------------------------------------------------------------

def run_colmap(
    image_dir,
    output_dir,
    colmap_exe=None,
    feature_type="ALIKED",
    matching_type="ALIKED_BRUTEFORCE",
    use_gpu=True,
    gpu_index=0,
    on_log=None,
):
    """
    Run COLMAP sparse reconstruction pipeline on a set of images.

    Steps:
      1. Feature extraction (ALIKED by default, GPU-accelerated)
      2. Sequential matching (GPU-accelerated)
      3. Sparse mapping (incremental SfM)
      4. Model export to text format for pose parsing

    Args:
        image_dir:     Directory containing input images.
        output_dir:    Directory for COLMAP output (database, sparse model).
        colmap_exe:    Path to COLMAP executable. Auto-detected if None.
        feature_type:  Feature type for extraction (default "ALIKED").
        matching_type: Matcher type (default "ALIKED_BRUTEFORCE").
        use_gpu:       Whether to use GPU acceleration.
        gpu_index:     GPU device index.
        on_log:        Optional progress callback.

    Returns:
        dict with keys:
            success: bool
            registered_images: number of images COLMAP registered
            total_images: total input images
            sparse_points: number of sparse 3D points
            cameras: parsed camera intrinsics
            poses: parsed camera extrinsics (per image)
            model_dir: path to the sparse model directory
    """
    log = on_log or (lambda msg: None)

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    # --- Resolve COLMAP executable ---
    if colmap_exe is None:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        colmap_exe = os.path.join(project_dir, "colmap-x64-windows-cuda", "COLMAP.bat")

    if not os.path.isfile(colmap_exe):
        raise FileNotFoundError(
            f"COLMAP executable not found: {colmap_exe}\n"
            "Download COLMAP and place it in the colmap-x64-windows-cuda/ directory."
        )

    # --- Setup paths ---
    os.makedirs(output_dir, exist_ok=True)
    database_path = os.path.join(output_dir, "database.db")
    sparse_dir = os.path.join(output_dir, "sparse")

    # Clean previous run
    if os.path.exists(database_path):
        log("Removing old COLMAP database…")
        os.remove(database_path)
    if os.path.exists(sparse_dir):
        log("Removing old sparse model…")
        shutil.rmtree(sparse_dir)
    os.makedirs(sparse_dir, exist_ok=True)

    # --- Setup environment (CUDA + COLMAP paths) ---
    env = os.environ.copy()
    colmap_dir = os.path.dirname(colmap_exe)
    colmap_bin = os.path.join(colmap_dir, "bin") if os.path.isdir(os.path.join(colmap_dir, "bin")) else colmap_dir

    # Add CUDA toolkit paths for Windows
    cuda_paths = []
    for ver in ["v12.6", "v12.5", "v12.4", "v12.3", "v12.2", "v12.1", "v12.0", "v11.8"]:
        cuda_bin = os.path.join(
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            "NVIDIA GPU Computing Toolkit", "CUDA", ver, "bin"
        )
        if os.path.isdir(cuda_bin):
            cuda_paths.append(cuda_bin)
            break

    env["PATH"] = ";".join(cuda_paths + [colmap_bin]) + ";" + env.get("PATH", "")
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    gpu_flag = "1" if use_gpu else "0"

    def _run_cmd(cmd, step_name):
        log(f"Running: {step_name}…")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                env=env, timeout=600,  # 10 min timeout per step
            )
            if result.returncode != 0:
                log(f"  Warning: {step_name} returned code {result.returncode}")
                if result.stderr:
                    # Log first few lines of stderr
                    for line in result.stderr.strip().split("\n")[:5]:
                        log(f"  stderr: {line}")
                return False
            return True
        except subprocess.TimeoutExpired:
            log(f"  Error: {step_name} timed out after 600 seconds")
            return False
        except FileNotFoundError:
            log(f"  Error: COLMAP executable not found")
            return False

    # --- Step 1: Feature Extraction ---
    log("Step 1/4: Feature extraction…")
    cmd = [
        colmap_exe, "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_dir,
        "--FeatureExtraction.type", feature_type,
        "--FeatureExtraction.use_gpu", gpu_flag,
        "--FeatureExtraction.gpu_index", str(gpu_index),
    ]
    if not _run_cmd(cmd, "Feature extraction"):
        return {"success": False, "error": "Feature extraction failed"}

    # --- Step 2: Feature Matching ---
    log("Step 2/4: Sequential matching…")
    cmd = [
        colmap_exe, "sequential_matcher",
        "--database_path", database_path,
        "--FeatureMatching.type", matching_type,
        "--FeatureMatching.use_gpu", gpu_flag,
        "--FeatureMatching.gpu_index", str(gpu_index),
    ]
    if not _run_cmd(cmd, "Feature matching"):
        return {"success": False, "error": "Feature matching failed"}

    # --- Step 3: Sparse Mapping ---
    log("Step 3/4: Sparse mapping (incremental SfM)…")
    cmd = [
        colmap_exe, "mapper",
        "--database_path", database_path,
        "--image_path", image_dir,
        "--output_path", sparse_dir,
    ]
    if not _run_cmd(cmd, "Sparse mapping"):
        return {"success": False, "error": "Sparse mapping failed"}

    # --- Step 4: Export to text format ---
    model_dir = os.path.join(sparse_dir, "0")
    if not os.path.exists(model_dir):
        log("Error: No model directory '0' created — reconstruction may have failed.")
        return {"success": False, "error": "No sparse model produced"}

    text_model_dir = os.path.join(output_dir, "sparse_text")
    os.makedirs(text_model_dir, exist_ok=True)

    log("Step 4/4: Exporting model to text format…")
    cmd = [
        colmap_exe, "model_converter",
        "--input_path", model_dir,
        "--output_path", text_model_dir,
        "--output_type", "TXT",
    ]
    if not _run_cmd(cmd, "Model export"):
        log("Warning: Text export failed, trying to parse binary model…")
        text_model_dir = model_dir  # Fall back to binary dir

    # --- Parse camera poses ---
    cameras_file = os.path.join(text_model_dir, "cameras.txt")
    images_file = os.path.join(text_model_dir, "images.txt")

    cameras = {}
    poses = {}

    if os.path.isfile(cameras_file):
        cameras = _parse_colmap_cameras(cameras_file)
        log(f"  Parsed {len(cameras)} camera models")

    if os.path.isfile(images_file):
        poses = _parse_colmap_images(images_file)
        log(f"  Parsed {len(poses)} camera poses")

    # Count registered images
    valid_ext = (".jpg", ".jpeg", ".png")
    total_images = len([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(valid_ext)
    ])

    # Save poses as JSON for pipeline consumption
    poses_json_path = os.path.join(output_dir, "colmap_poses.json")
    with open(poses_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "cameras": cameras,
            "poses": poses,
            "registered_images": len(poses),
            "total_images": total_images,
        }, f, indent=2)
    log(f"Camera poses saved: {poses_json_path}")

    log(
        f"COLMAP complete: {len(poses)}/{total_images} images registered, "
        f"{len(cameras)} camera models"
    )

    return {
        "success": True,
        "registered_images": len(poses),
        "total_images": total_images,
        "cameras": cameras,
        "poses": poses,
        "model_dir": model_dir,
        "poses_json_path": poses_json_path,
    }
