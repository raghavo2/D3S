# D3S — Drone Video to 3D Model

An AI-powered platform that transforms a single drone flight video into georeferenced, metrically accurate 3D models. No Ground Control Points (GCPs), no multi-pass overlap required.

![Python](https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-19-61dafb?logo=react&logoColor=white)
![Three.js](https://img.shields.io/badge/Three.js-r186-000000?logo=three.js&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi&logoColor=white)

---

## Features

- **AI-Driven Reconstruction** — DUSt3R multi-view stereo infers depth and constructs dense point clouds from single-pass imagery
- **Intelligent Frame Selection** — Adaptive quality filtering + SSIM-based deduplication reduces input to optimal keyframes
- **Dynamic Object Masking** — YOLO + ByteTrack with camera-motion-compensated tracking removes moving objects from reconstruction
- **Dual Pose Estimation** — DUSt3R global alignment + COLMAP sparse SfM for independent camera pose comparison
- **High-Fidelity Meshing** — Screened Poisson Surface Reconstruction with density equalization and Taubin smoothing
- **Flight Telemetry** — Extracts EXIF GPS coordinates and parses flight logs for real-world alignment
- **Multi-Format Export** — Outputs to GLB (web), LAS (GIS), OBJ (CAD), PLY (point cloud)
- **Interactive 3D Viewer** — Browser-based mesh and point cloud viewer with orbit controls
- **Real-Time Pipeline Monitor** — WebSocket-powered live progress tracking across 9 pipeline stages
- **GPS Flight Map** — Leaflet-based flight path visualization with altitude markers
- **Job History** — Persistent job tracking across server restarts
- **Graceful Degradation** — Every non-critical stage degrades safely; system never breaks

---

## Architecture

```
D3S/
├── server.py                     # FastAPI backend (REST + WebSocket)
├── run_pipeline.py               # 9-stage pipeline orchestrator (subprocess)
│
├── pipeline/                     # Modular pipeline stages
│   ├── __init__.py               #   Package exports
│   ├── config.py                 #   Centralized config, paths, stage definitions
│   ├── frames.py                 #   Video → frame extraction (OpenCV)
│   ├── quality.py                #   Frame quality scoring (blur + exposure)
│   ├── keyframes.py              #   Keyframe selection (SSIM + temporal spacing)
│   ├── detection.py              #   YOLO + ByteTrack + ORB motion tracking
│   ├── masking.py                #   Detection → binary masks
│   ├── colmap.py                 #   COLMAP sparse SfM wrapper
│   ├── metadata.py               #   EXIF/GPS + telemetry parsing
│   ├── reconstruct.py            #   DUSt3R point cloud generation
│   ├── mesh.py                   #   Poisson surface meshing (Open3D)
│   └── export.py                 #   GLB/LAS/OBJ deliverable export
│
├── detection/                    # Standalone detection tools
│   ├── detect.py                 #   V1 detection-only script
│   ├── track.py                  #   Tracking + camera compensation
│   └── sample_output/            #   Reference output formats
│
├── frontend/                     # React + Vite SPA
│   └── src/
│       ├── components/           #   Navbar, ModelViewer, PointCloudViewer,
│       │                         #   FlightMap, FrameGallery, ExportPanel
│       ├── pages/                #   Landing, Upload, Pipeline, Viewer, History
│       ├── hooks/                #   useApi, useWebSocket
│       └── utils/                #   Constants & config
│
├── dust3r/                       # DUSt3R source (submodule)
├── checkpoints/                  # DUSt3R model weights (git-ignored)
├── models/                       # YOLO model weights
│   └── D3S_YOLO_80.pt
├── colmap-x64-windows-cuda/      # COLMAP binary distribution
├── jobs/                         # Per-job working directories (git-ignored)
└── uploads/                      # Uploaded videos (git-ignored)
```

---

## Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.10+ | Backend + Pipeline |
| Node.js | 18+ | Frontend build |
| CUDA Toolkit | 11.8+ | GPU acceleration (DUSt3R, COLMAP) |
| PyTorch | 2.0+ with CUDA | DUSt3R inference |
| NVIDIA GPU | 6GB+ VRAM | RTX 4060 or higher recommended |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/raghavo2/D3S.git
cd D3S
```

### 2. Backend setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install PyTorch with CUDA (must be done first)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install all dependencies
pip install -r requirements_server.txt
```

### 3. Download model weights

```bash
# DUSt3R model (required for reconstruction)
python download_model.py
```

Place the YOLO model at `models/D3S_YOLO_80.pt` (provided by team).

### 4. COLMAP setup (optional, for Phase 2C)

Download [COLMAP](https://colmap.github.io/install.html) pre-built binaries for Windows with CUDA and extract to `colmap-x64-windows-cuda/`. The pipeline auto-detects `COLMAP.bat` in this directory.

### 5. Frontend setup

```bash
cd frontend
npm install
cd ..
```

---

## Dependencies

### Python — Server (`requirements_server.txt`)

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | ≥0.104.0 | REST API + WebSocket server |
| `uvicorn[standard]` | ≥0.24.0 | ASGI server |
| `python-multipart` | ≥0.0.6 | File upload handling |
| `websockets` | ≥12.0 | WebSocket support |

### Python — Pipeline (used by `run_pipeline.py` subprocess)

| Package | Version | Purpose | Used By |
|---------|---------|---------|---------|
| `torch` | ≥2.0 | GPU inference | `reconstruct.py` |
| `torchvision` | ≥0.15 | Image transforms | `reconstruct.py` |
| `opencv-python` | latest | Video frames, image I/O, ORB features | `frames.py`, `quality.py`, `keyframes.py`, `detection.py`, `masking.py` |
| `numpy` | latest | Array operations | All modules |
| `scipy` | latest | Signal processing | `keyframes.py` |
| `Pillow` | latest | Image loading | `metadata.py`, DUSt3R |
| `trimesh` | latest | Point cloud export | `reconstruct.py`, `export.py` |
| `open3d` | latest | Poisson meshing, outlier removal | `mesh.py` |
| `laspy` | latest | LAS point cloud export | `export.py` |
| `ultralytics` | latest | YOLO model inference + ByteTrack | `detection.py` |
| `einops` | latest | Tensor operations | DUSt3R |
| `roma` | latest | Rotation utilities | DUSt3R |
| `tqdm` | latest | Progress bars | DUSt3R |
| `huggingface-hub[torch]` | ≥0.22 | Model download | `download_model.py` |
| `rosbags` | latest | ROS bag telemetry parsing | `metadata.py` |

### Python — Standard Library (no install needed)

| Module | Used By |
|--------|---------|
| `os`, `sys`, `json` | All modules |
| `math` | `detection.py` (distance calculations) |
| `shutil` | `colmap.py` (directory cleanup) |
| `subprocess` | `colmap.py` (COLMAP execution) |
| `argparse` | `run_pipeline.py`, `reconstruct.py` |
| `traceback` | `run_pipeline.py` (error reporting) |
| `datetime` | `run_pipeline.py` (timestamps) |
| `collections.defaultdict` | `detection/track.py` |
| `re` | `colmap.py` |
| `uuid`, `asyncio` | `server.py` |

### Frontend (`frontend/package.json`)

| Package | Version | Purpose |
|---------|---------|---------|
| `react` | ^19.2.8 | UI framework |
| `react-dom` | ^19.2.8 | DOM rendering |
| `react-router-dom` | ^7.18.3 | Client-side routing |
| `three` | ^0.186.0 | 3D rendering engine |
| `@react-three/fiber` | ^9.7.0 | React renderer for Three.js |
| `@react-three/drei` | ^10.7.8 | Three.js helpers (OrbitControls, etc.) |
| `leaflet` | ^1.9.4 | GPS flight path maps |
| `react-leaflet` | ^5.0.0 | React wrapper for Leaflet |

### Frontend Dev Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `vite` | ^8.3.0 | Build tool + HMR |
| `@vitejs/plugin-react` | ^6.1.1 | React fast refresh |
| `oxlint` | ^1.81.0 | Linter |

### External Tools (not pip-installable)

| Tool | Location | Purpose |
|------|----------|---------|
| COLMAP | `colmap-x64-windows-cuda/` | Sparse SfM for camera pose estimation |
| CUDA Toolkit | System install | GPU compute for COLMAP ALIKED features |

---

## Running

### Start the backend

```bash
python -m uvicorn server:app --port 8000
```

### Start the frontend (separate terminal)

```bash
cd frontend
npm run dev
```

The app will be available at **http://localhost:5173**

### Health check

```bash
curl http://localhost:8000/api/health
```

### Run pipeline directly (CLI, no server)

```bash
python run_pipeline.py --job-id my-test --video path/to/drone_video.mp4 --fps 2.0
```

---

## Usage

1. **Upload** — Navigate to `/upload`, drag and drop your drone video, set FPS (1–10), and click "Upload & Start Reconstruction"
2. **Monitor** — Watch real-time progress on the Pipeline Monitor with 9-stage tracking, elapsed timer, and live logs
3. **View** — After completion, auto-redirects to the 3D Viewer where you can orbit, pan, zoom the mesh or point cloud
4. **Export** — Download output files (GLB, PLY, LAS, OBJ) from the Export panel in the viewer toolbar
5. **History** — View past jobs at `/history`, click to re-open in viewer or pipeline

---

## Pipeline Stages (9-Stage V2)

| # | Stage | Module | Description |
|---|-------|--------|-------------|
| 1 | 🎞️ **Frame Extraction** | `pipeline/frames.py` | Extracts evenly-spaced frames from video using OpenCV |
| 2 | 🔍 **Quality Filtering** | `pipeline/quality.py` | Rejects blurry (Laplacian variance) and overexposed frames |
| 3 | 🎯 **Keyframe Selection** | `pipeline/keyframes.py` | Removes near-duplicates (SSIM > 0.92), enforces temporal spacing |
| 4 | 🚨 **Object Detection** | `pipeline/detection.py` | YOLO + ByteTrack + ORB homography → classifies MOVING vs STATIC |
| 5 | 📰 **COLMAP Poses** | `pipeline/colmap.py` | ALIKED features → sequential matching → incremental SfM |
| 6 | 📡 **Metadata Extraction** | `pipeline/metadata.py` | Parses EXIF GPS, altitude, heading + ROS bag telemetry |
| 7 | ☁️ **Point Cloud Generation** | `pipeline/reconstruct.py` | DUSt3R multi-view stereo inference + dynamic object masking |
| 8 | 🔺 **Surface Meshing** | `pipeline/mesh.py` | Poisson reconstruction (depth 11) + Taubin smoothing |
| 9 | 📦 **Deliverable Export** | `pipeline/export.py` | Converts to GLB, LAS, OBJ with proper coordinate systems |

**Stages 2–6 are non-critical.** If any fail or their dependencies are missing, the pipeline continues with reduced functionality (same as V1 output quality).

### Data Flow

```
Video (.mp4) → frames/ → quality scores → keyframes/ → YOLO tracking → masks/
                                                      → COLMAP → colmap_poses.json
                                                      → GPS metadata
                                    ↓
                    DUSt3R + masks + poses → point_cloud.ply → mesh.obj → deliverables/
```

### Per-Job Output

```
jobs/{job_id}/
├── frames/                       All extracted frames
├── keyframes/                    Selected keyframes + selection_report.json
├── masks/                        Binary masks (255=keep, 0=exclude)
├── colmap/                       COLMAP database + sparse model + colmap_poses.json
├── metadata/                     gps.csv, gps.json, telemetry.json
└── output/
    ├── detections.json           Full tracking results with motion labels
    ├── pose_comparison.json      DUSt3R vs COLMAP camera poses
    ├── point_cloud.ply           Colored point cloud (dynamic objects removed)
    ├── mesh.obj                  Surface mesh
    └── deliverables/
        ├── model.glb             Web-viewable 3D model
        └── point_cloud.las       GIS-compatible point cloud
```

---

## Configuration

All pipeline defaults are in `pipeline/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEFAULT_FPS` | 2.0 | Frame extraction rate |
| `DEFAULT_IMAGE_SIZE` | 512 | DUSt3R inference resolution |
| `DEFAULT_BLUR_REJECT_PERCENTILE` | 20 | Reject bottom 20% sharpness |
| `DEFAULT_MAX_CLIPPING_RATIO` | 0.05 | Reject if >5% pixels clipped |
| `DEFAULT_SSIM_THRESHOLD` | 0.92 | SSIM above this = near-duplicate |
| `DEFAULT_MIN_TEMPORAL_GAP` | 1 | Minimum frame gap between keyframes |
| `DEFAULT_MAX_TEMPORAL_GAP` | 10 | Force-include if gap exceeds this |
| `DEFAULT_YOLO_CONFIDENCE` | 0.35 | Minimum detection confidence |
| `DEFAULT_MOVEMENT_THRESHOLD` | 5.0 | Compensated px movement → MOVING |
| `DEFAULT_MASK_PADDING_PX` | 10 | Bounding box padding for masks |
| `DEFAULT_ORB_FEATURES` | 1500 | ORB features for camera motion |
| `DEFAULT_ALIGNMENT_ITERS` | 300 | DUSt3R global alignment iterations |
| `DEFAULT_COLMAP_FEATURE_TYPE` | ALIKED | COLMAP neural feature type |
| `DEFAULT_COLMAP_MATCHING_TYPE` | ALIKED_BRUTEFORCE | COLMAP matcher type |

---

## API Reference

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Server health + GPU status |
| `POST` | `/api/upload` | Upload video file (multipart) |
| `POST` | `/api/jobs/{id}/start` | Start pipeline for uploaded job |
| `GET` | `/api/jobs` | List all jobs |
| `GET` | `/api/jobs/{id}` | Get job status + metadata |
| `DELETE` | `/api/jobs/{id}` | Delete job and files |
| `GET` | `/api/jobs/{id}/metadata` | Get extracted GPS/EXIF metadata |
| `GET` | `/api/jobs/{id}/frames` | List extracted frame files |
| `GET` | `/api/jobs/{id}/model` | Serve GLB model file |
| `GET` | `/api/jobs/{id}/pointcloud` | Serve PLY point cloud |
| `GET` | `/api/jobs/{id}/gps` | Get GPS track as GeoJSON |
| `GET` | `/api/jobs/{id}/outputs` | List all output files |
| `GET` | `/api/jobs/{id}/outputs/{file}` | Download specific output file |
| `POST` | `/api/demos` | Save job as demo |
| `GET` | `/api/demos` | List demos |
| `DELETE` | `/api/demos/{id}` | Delete demo |

### WebSocket

| Path | Description |
|------|-------------|
| `ws://localhost:8000/ws/{job_id}` | Real-time pipeline logs + stage updates |

**Event types:**
```json
{ "type": "stage", "stage": "object_detection", "status": "running" }
{ "type": "log", "stage": "object_detection", "message": "Tracked 20/35 frames" }
{ "type": "stage", "stage": "object_detection", "status": "complete", "result": { "dynamic_objects": 12, "static_objects": 45 } }
{ "type": "pipeline", "status": "complete", "job_id": "abc-123" }
```

---

## Tech Stack

### Backend
- **FastAPI** — REST API + WebSocket server
- **DUSt3R** — Multi-view stereo 3D reconstruction (ViT-Large)
- **YOLO + ByteTrack** — Object detection + multi-object tracking
- **COLMAP** — Sparse Structure-from-Motion (ALIKED features)
- **Open3D** — Point cloud processing + Poisson meshing
- **OpenCV** — Video extraction, ORB feature matching, homography estimation
- **PyTorch** — GPU-accelerated inference

### Frontend
- **React 19** — UI framework
- **Vite 8** — Build tool + HMR
- **Three.js** — 3D rendering (via react-three-fiber)
- **Leaflet** — GPS flight path maps (via react-leaflet)
- **Inter + Outfit** — Typography

---

## V2 Improvements Over V1

| Feature | V1 | V2 |
|---------|----|----|
| Frame selection | All extracted frames | Quality filter → SSIM deduplicate → temporal spacing |
| Dynamic objects | Not handled | Motion-compensated tracking → per-pixel masks |
| Camera poses | DUSt3R only | DUSt3R + COLMAP (independent comparison) |
| Point cloud | Raw DUSt3R output | Masked (no moving objects) |
| Pipeline stages | 5 stages | 9 stages |
| Observability | Basic progress | Per-stage JSON events with detailed results |
| Robustness | Fails on errors | Non-critical stages degrade gracefully |

---

## Project Status

- [x] V1 — Pipeline orchestration, 3D viewer, demo gallery, GPS maps, job history
- [x] V2 Phase 2A — Intelligent keyframe selection (quality + SSIM)
- [x] V2 Phase 2B — Dynamic object masking (YOLO + ByteTrack + ORB)
- [x] V2 Phase 2C — COLMAP pose estimation integration
- [ ] V2 Validation — Full pipeline test on target GPU
- [ ] V3 — Voxel confidence grid, pose-guided depth projection, georeferencing
- [ ] V4 — Second-pass flight planning, advanced exports

---

## License

MIT