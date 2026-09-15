# D3S — Drone Video to 3D Model

An AI-powered platform that transforms a single drone flight video into georeferenced, metrically accurate 3D models. No Ground Control Points (GCPs), no multi-pass overlap required.

![Python](https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-19-61dafb?logo=react&logoColor=white)
![Three.js](https://img.shields.io/badge/Three.js-r186-000000?logo=three.js&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi&logoColor=white)

---

## Features

- **AI-Driven Reconstruction** — DUSt3R multi-view stereo infers depth and constructs dense point clouds from single-pass imagery
- **High-Fidelity Meshing** — Screened Poisson Surface Reconstruction with Taubin smoothing
- **Flight Telemetry** — Extracts EXIF GPS coordinates and parses flight logs for real-world alignment
- **Multi-Format Export** — Outputs to GLB (web), LAS (GIS), OBJ (CAD), PLY (point cloud)
- **Interactive 3D Viewer** — Browser-based mesh and point cloud viewer with orbit controls
- **Real-Time Pipeline Monitor** — WebSocket-powered live progress tracking with log streaming
- **GPS Flight Map** — Leaflet-based flight path visualization with altitude markers
- **Job History** — Persistent job tracking across server restarts with deletion support
- **Demo Gallery** — Save completed reconstructions as shareable demos

---

## Architecture

```
d3s/
├── server.py                   # FastAPI backend (REST + WebSocket)
├── run_pipeline.py             # Pipeline orchestrator (subprocess)
├── pipeline/                   # Modular pipeline stages
│   ├── config.py               #   Stage definitions & paths
│   ├── frames.py               #   Video → frame extraction (OpenCV)
│   ├── metadata.py             #   EXIF/GPS parsing
│   ├── reconstruct.py          #   DUSt3R point cloud generation
│   ├── mesh.py                 #   Poisson surface meshing (Open3D)
│   └── export.py               #   GLB/LAS/OBJ deliverable export
├── frontend/                   # React + Vite SPA
│   └── src/
│       ├── components/         #   Reusable UI components
│       │   ├── Navbar           #   Navigation bar with server status
│       │   ├── ModelViewer      #   Three.js GLB mesh viewer
│       │   ├── PointCloudViewer #   Three.js point cloud viewer
│       │   ├── FlightMap        #   Leaflet GPS flight path map
│       │   ├── FrameGallery     #   Thumbnail grid with lightbox
│       │   └── ExportPanel      #   File download dropdown
│       ├── pages/              #   Route pages
│       │   ├── Landing          #   Hero + features + pipeline overview
│       │   ├── Upload           #   Drag-and-drop video upload
│       │   ├── Pipeline         #   Real-time pipeline monitor
│       │   ├── Viewer           #   3D model/point cloud viewer
│       │   └── History          #   Job history management
│       ├── hooks/              #   Custom React hooks
│       │   ├── useApi           #   REST API client
│       │   └── useWebSocket     #   WebSocket connection manager
│       └── utils/              #   Constants & config
├── dust3r/                     # DUSt3R model (submodule/local)
├── checkpoints/                # Model weights (git-ignored)
└── jobs/                       # Job workspace (git-ignored)
```

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| Node.js | 18+ |
| CUDA | 11.8+ (for GPU acceleration) |
| PyTorch | 2.0+ with CUDA |

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

# Install dependencies
pip install -r requirements_server.txt

# Install PyTorch with CUDA (if not already installed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Download DUSt3R model weights

```bash
python download_model.py
```

This downloads the DUSt3R checkpoint to `checkpoints/`.

### 4. Frontend setup

```bash
cd frontend
npm install
cd ..
```

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

---

## Usage

1. **Upload** — Navigate to `/upload`, drag and drop your drone video, set FPS (1–10), and click "Upload & Start Reconstruction"
2. **Monitor** — Watch real-time progress on the Pipeline Monitor with stage tracking, elapsed timer, and live logs
3. **View** — After completion, auto-redirects to the 3D Viewer where you can orbit, pan, zoom the mesh or point cloud
4. **Export** — Download output files (GLB, PLY, LAS, OBJ) from the Export panel in the viewer toolbar
5. **History** — View past jobs at `/history`, click to re-open in viewer or pipeline

---

## Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| **Frame Extraction** | `pipeline/frames.py` | Extracts evenly-spaced frames from video using OpenCV |
| **Metadata Extraction** | `pipeline/metadata.py` | Parses EXIF GPS, altitude, heading from each frame |
| **Point Cloud Generation** | `pipeline/reconstruct.py` | Runs DUSt3R multi-view stereo inference on GPU |
| **Surface Meshing** | `pipeline/mesh.py` | Poisson reconstruction + Taubin smoothing via Open3D |
| **Deliverable Export** | `pipeline/export.py` | Converts to GLB, LAS, OBJ with proper coordinate systems |

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

**Message types:**
- `{ "type": "stage", "stage": "...", "status": "running|complete|error" }`
- `{ "type": "log", "message": "...", "stage": "..." }`
- `{ "type": "pipeline", "status": "complete|failed" }`

---

## Tech Stack

### Backend
- **FastAPI** — REST API + WebSocket server
- **DUSt3R** — Multi-view stereo 3D reconstruction
- **Open3D** — Point cloud processing + Poisson meshing
- **OpenCV** — Video frame extraction
- **PyTorch** — GPU-accelerated inference

### Frontend
- **React 19** — UI framework
- **Vite 8** — Build tool + HMR
- **Three.js** — 3D rendering (via react-three-fiber)
- **Leaflet** — GPS flight path maps (via react-leaflet)
- **Inter + Outfit** — Typography

---

## Project Status

- [x] Phase 1 — Pipeline orchestration, 3D viewer, demo gallery
- [x] Phase 2 — GPS maps, job history, export panel, frame gallery, pipeline UX
- [ ] Phase 3 — Authentication, deployment, advanced exports

---

## License

MIT