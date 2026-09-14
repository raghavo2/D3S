# Single-Pass Drone Video to Accurate 3D Model Generation System

An AI-enabled computer vision pipeline designed to generate georeferenced, metrically accurate 3D models from a single drone flight pass, eliminating the need for extensive Ground Control Points (GCPs) or multi-pass overlap.

## Core Features
* **AI-Driven Reconstruction:** Utilizes DUSt3R (Multi-View Stereo) to infer depth and construct dense point clouds directly from single-pass drone imagery.
* **High-Fidelity Meshing:** Employs Open3D Screened Poisson Surface Reconstruction with Taubin smoothing to create crisp, continuous surfaces while preserving right angles on buildings and infrastructure.
* **Flight Telemetry Integration:** Extracts raw EXIF GPS coordinates and parses `.bag` flight logs to constrain the global alignment and provide real-world metric scale.
* **Multi-Format Deliverables:** Automatically compiles reconstruction outputs into industry-standard formats: `.obj`, `.glb` (Web Viewer), and `.las` (GIS integration).

## Setup & Installation
Ensure you have PyTorch with CUDA enabled, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Execution Pipeline
**0. Video Ingestion**
Extracts perfectly spaced image frames from a raw drone video and prepares them for reconstruction.
```bash
python extract_frames.py path/to/your/video.mp4 --fps 2
```

**1. Extract Metadata**
Parses the drone image EXIF tags and flight `.bag` files for trajectory mapping.
```bash
python extract_gps.py
python extract_telemetry.py
```

**2. Point Cloud Generation**
```bash
python test_multiview.py
```

**3. Surface Meshing**
```bash
python process_dust3r.py
```

**4. Deliverable Export**
```bash
python export_deliverables.py
```