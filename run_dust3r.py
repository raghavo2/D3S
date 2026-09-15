import os
import sys
import torch
import argparse

# Tell PyTorch 2.6+ it's safe to load the Namespace object inside DUSt3R's checkpoint
torch.serialization.add_safe_globals([argparse.Namespace])
from glob import glob

# Add dust3r path
d3r_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dust3r")
sys.path.insert(0, d3r_path)

# Ensure croco is in path
sys.path.insert(0, os.path.join(d3r_path, "croco"))

# pyrefly: ignore [missing-import]
from dust3r.inference import inference
# pyrefly: ignore [missing-import]
from dust3r.model import AsymmetricCroCo3DStereo
# pyrefly: ignore [missing-import]
from dust3r.utils.image import load_images
# pyrefly: ignore [missing-import]
from dust3r.image_pairs import make_pairs
# pyrefly: ignore [missing-import]
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Path to the pretrained model (we will download this)
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please download it first.")
        sys.exit(1)
        
    print("Loading model...")
    model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)
    
    IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AGZ_subset", "AGZ_subset", "MAV_Images")
    print(f"Loading images from {IMAGE_DIR}...")
    image_files = sorted(glob(os.path.join(IMAGE_DIR, "*.jpg")) + glob(os.path.join(IMAGE_DIR, "*.png")))
    
    if not image_files:
        print(f"No images found in {IMAGE_DIR}")
        sys.exit(1)
        
    print(f"Found {len(image_files)} images.")
    imgs = load_images(image_files, size=512)
    
    # Create pairs for inference
    print("Making image pairs...")
    pairs = make_pairs(imgs, scene_graph='swin-2', prefilter=None, symmetrize=True)
    
    # Run inference
    print("Running inference (this may take a while depending on GPU)...")
    output = inference(pairs, model, device, batch_size=2)
    
    # Global Alignment
    print("Performing global alignment...")
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init='mst', niter=300, schedule='linear', lr=0.01)
    print(f"Final alignment loss: {loss}")
    
    # Export to PLY
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dust3r_output")
    os.makedirs(out_dir, exist_ok=True)
    
    print("Cleaning pointcloud and exporting...")
    scene = scene.clean_pointcloud()
    ply_out = os.path.join(out_dir, "dust3r_point_cloud.ply")
    scene.export_pointcloud(ply_out)
    
    print(f"\nSuccess! 3D point cloud saved to: {ply_out}")

if __name__ == "__main__":
    main()
