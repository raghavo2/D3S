import os
import shutil
import subprocess
import sys

# Set paths based on your d3s directory structure
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
COLMAP_DIR = os.path.join(PROJECT_DIR, "colmap-x64-windows-cuda")
COLMAP_EXE = os.path.join(COLMAP_DIR, "COLMAP.bat")

# Add COLMAP bin and CUDA toolkit to PATH so Windows loads CUDA DLLs correctly
colmap_bin = os.path.join(COLMAP_DIR, "bin")
cuda_12_6_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
os.environ["PATH"] = f"{cuda_12_6_bin};{colmap_bin};" + os.environ.get("PATH", "")

# Target primary GPU (RTX 4060 Laptop)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

IMAGE_DIR = os.path.join(PROJECT_DIR, "AGZ_subset", "AGZ_subset", "MAV_Images")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "colmap_output")
DATABASE_PATH = os.path.join(OUTPUT_DIR, "database.db")
SPARSE_DIR = os.path.join(OUTPUT_DIR, "sparse")

def run_command(cmd):
    print(f"\n--- Running: {' '.join(cmd)} ---\n")
    try:
        # env=os.environ guarantees the batch script sees your CUDA target
        subprocess.run(cmd, check=True, env=os.environ)
    except subprocess.CalledProcessError:
        print(f"\nError executing command. Check COLMAP output above.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"\nError: Could not find COLMAP at {COLMAP_EXE}.")
        sys.exit(1)

def main():
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Could not find image directory at {IMAGE_DIR}")
        sys.exit(1)

    # Clean up previous run to avoid SQLite constraint conflicts
    if os.path.exists(DATABASE_PATH):
        print(f"Removing old database: {DATABASE_PATH}")
        os.remove(DATABASE_PATH)
    if os.path.exists(SPARSE_DIR):
        print(f"Removing old sparse directory: {SPARSE_DIR}")
        shutil.rmtree(SPARSE_DIR)

    os.makedirs(SPARSE_DIR, exist_ok=True)

    # 1. Feature Extraction — ALIKED (neural network via ONNX Runtime CUDA)
    #    SIFT GPU uses SiftGPU CUDA kernels which lack sm_89 (Ada Lovelace)
    #    support in this build. ALIKED runs through ONNX Runtime's CUDA
    #    provider which fully supports RTX 40-series GPUs.
    run_command([
        COLMAP_EXE, "feature_extractor",
        "--database_path", DATABASE_PATH,
        "--image_path", IMAGE_DIR,
        "--FeatureExtraction.type", "ALIKED",
        "--FeatureExtraction.use_gpu", "1",
        "--FeatureExtraction.gpu_index", "0",
    ])

    # 2. Feature Matching — ALIKED brute-force matching (GPU-accelerated)
    run_command([
        COLMAP_EXE, "sequential_matcher",
        "--database_path", DATABASE_PATH,
        "--FeatureMatching.type", "ALIKED_BRUTEFORCE",
        "--FeatureMatching.use_gpu", "1",
        "--FeatureMatching.gpu_index", "0",
    ])

    # 3. Sparse Mapping (3D Reconstruction)
    run_command([
        COLMAP_EXE, "mapper",
        "--database_path", DATABASE_PATH,
        "--image_path", IMAGE_DIR,
        "--output_path", SPARSE_DIR,
    ])

    # 4. Export to PLY
    model_0_dir = os.path.join(SPARSE_DIR, "0")
    if os.path.exists(model_0_dir):
        ply_output = os.path.join(OUTPUT_DIR, "point_cloud.ply")
        run_command([
            COLMAP_EXE, "model_converter",
            "--input_path", model_0_dir,
            "--output_path", ply_output,
            "--output_type", "PLY",
        ])
        print(f"\nSuccess! 3D point cloud saved to: {ply_output}")
    else:
        print("\nReconstruction failed. No model directory '0' was created in the sparse folder.")

if __name__ == "__main__":
    main()