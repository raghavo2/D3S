import os
import sys
import argparse
import torch

# 1. Allow the PyTorch 2.6 weights to load
torch.serialization.add_safe_globals([argparse.Namespace])

# 2. Tell Python where the dust3r and croco folders are
d3r_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dust3r")
sys.path.insert(0, d3r_path)
sys.path.insert(0, os.path.join(d3r_path, "croco"))

# 3. Safely import DUSt3R
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

# Use local model to avoid re-downloading 2GB
MODEL_NAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")

# Point to the actual folder containing your drone images
IMAGE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "AGZ_subset",
    "MAV Images"
)

# Automatically grab all image files and sort them sequentially
valid_extensions = ('.jpg', '.jpeg', '.png')
IMAGE_PATHS = sorted([
    os.path.join(IMAGE_DIR, f) 
    for f in os.listdir(IMAGE_DIR) 
    if f.lower().endswith(valid_extensions)
])

# Safety check just in case the folder is empty
if not IMAGE_PATHS:
    print(f"Error: No images found in {IMAGE_DIR}!")
    sys.exit(1)

print("=" * 60)
print("DUSt3R COLORED MULTI-VIEW TEST")
print("=" * 60)

# Switched to CUDA so it doesn't take hours
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print()
print("Device:", device)

# --------------------------------------------------
# Load model
# --------------------------------------------------

print()
print("Loading DUSt3R model...")

model = AsymmetricCroCo3DStereo.from_pretrained(
    MODEL_NAME,
    img_size=(512, 512)
)

model.to(device)
model.eval()

print("Model loaded.")

# --------------------------------------------------
# Load images
# --------------------------------------------------

print()
print("Loading images...")

images = load_images(
    IMAGE_PATHS,
    size=512
)

print("Images loaded:", len(images))

# --------------------------------------------------
# Create pairs
# --------------------------------------------------

print()
print("Creating image pairs...")

pairs = make_pairs(
    images,
    scene_graph="swin-3",
    prefilter=None,
    symmetrize=True
)

print("Number of pairs:", len(pairs))

# --------------------------------------------------
# Inference
# --------------------------------------------------

print()
print("Running DUSt3R inference...")

output = inference(
    pairs,
    model,
    device=device,
    batch_size=1, # Kept at 1 for safe VRAM usage
    verbose=True
)

print()
print("Inference completed.")

# --------------------------------------------------
# Global alignment
# --------------------------------------------------

print()
print("Starting global alignment...")

scene = global_aligner(
    output,
    device=device,
    mode=GlobalAlignerMode.PointCloudOptimizer
)

print("Global aligner created.")

print()
print("Optimizing global scene...")

scene.compute_global_alignment(
    init="mst",
    niter=300,
    schedule="cosine",
    lr=0.01
)
import numpy as np
import trimesh

print()
print("Exporting colored point cloud...")

imgs = scene.imgs
pts3d = scene.get_pts3d()
masks = scene.get_masks()

all_points = []
all_colors = []

for i in range(len(pts3d)):

    points = pts3d[i].detach().cpu().numpy()

    mask = masks[i].detach().cpu().numpy()

    # DUSt3R image is H x W x 3
    image = imgs[i]

    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()

    image = np.asarray(image)

    # Remove possible batch dimension
    if image.ndim == 4:
        image = image[0]

    # Convert colors to 0-255
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

print("Total colored points:", len(all_points))

cloud = trimesh.PointCloud(
    all_points,
    colors=all_colors
)

cloud.export("dust3r_multiview_colored.ply")

print("Saved:")
print("dust3r_multiview_colored.ply")
print()
print("Global alignment completed.")

# --------------------------------------------------
# Clean inconsistent points
# --------------------------------------------------

print()
print("Cleaning point cloud...")

scene = scene.clean_pointcloud(
    tol=0.001,
    bad_conf=0
)

print("Point cloud cleaned.")

# --------------------------------------------------
# DUSt3R colored viewer
# --------------------------------------------------

print()
print("Opening DUSt3R colored viewer...")
print()
print("Inspect the reconstruction from different angles.")
print("Close the viewer when finished.")

scene.show(
    show_pw_cams=False,
    show_pw_pts3d=False
)

print()
print("Finished.")