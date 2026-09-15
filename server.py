"""
D3S FastAPI Server
==================
REST + WebSocket API for the 3D reconstruction pipeline.

Start with:
    uvicorn server:app --reload --port 8000
"""

import os
import sys
import json
import uuid
import asyncio
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# --- App Configuration ---
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(PROJECT_DIR, "jobs")
UPLOADS_DIR = os.path.join(PROJECT_DIR, "uploads")
DELIVERABLES_DIR = os.path.join(PROJECT_DIR, "drone_deliverables")

# --- Pipeline Python ---
# The server may run in a lightweight venv (e.g. Python 3.14) that lacks
# heavy ML deps (torch, cv2, open3d). The pipeline subprocess must use
# a Python that has them all installed. We auto-detect or fall back.
def _find_pipeline_python():
    """Find a Python interpreter that has the ML pipeline dependencies."""
    import subprocess
    # First, try the current interpreter
    candidates = [sys.executable]
    # Add known system Python paths (Windows)
    for ver in ["Python311", "Python312", "Python310"]:
        candidates.append(os.path.join(
            os.environ.get("LOCALAPPDATA", ""), "Programs", "Python", ver, "python.exe"
        ))
    for cand in candidates:
        if not os.path.isfile(cand):
            continue
        try:
            result = subprocess.run(
                [cand, "-c", "import cv2, torch, open3d"],
                capture_output=True, timeout=15,
            )
            if result.returncode == 0:
                return cand
        except Exception:
            continue
    # Last resort
    return sys.executable

PIPELINE_PYTHON = _find_pipeline_python()

os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# --- In-memory job state ---
# Each job: {id, status, stage, created_at, video_path, fps, logs[], result{}}
jobs = {}

# WebSocket connections per job: {job_id: set[WebSocket]}
ws_connections: dict[str, set] = {}


def _save_job_state(job_id: str):
    """Persist job metadata to disk so jobs survive server restarts."""
    if job_id not in jobs:
        return
    job = jobs[job_id]
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    # Save a serialisable subset (skip logs to keep file small)
    state = {k: v for k, v in job.items() if k != "logs"}
    state["log_count"] = len(job.get("logs", []))
    state_path = os.path.join(job_dir, "status.json")
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2, default=str)


def _load_jobs_from_disk():
    """Scan jobs/ directory on startup and reload persisted job metadata."""
    if not os.path.isdir(JOBS_DIR):
        return
    for entry in os.listdir(JOBS_DIR):
        state_path = os.path.join(JOBS_DIR, entry, "status.json")
        if os.path.isfile(state_path) and entry not in jobs:
            try:
                with open(state_path, "r") as f:
                    data = json.load(f)
                    data.setdefault("logs", [])
                    # Mark running jobs as failed (server was restarted)
                    if data.get("status") == "running":
                        data["status"] = "interrupted"
                    jobs[data["id"]] = data
            except Exception:
                pass

_load_jobs_from_disk()


def create_app():
    app = FastAPI(
        title="D3S — Drone Video to 3D Model",
        description="REST + WebSocket API for AI-driven 3D reconstruction",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


app = create_app()


# ──────────────────────────────────────────────────────────────────────
# Health Check
# ──────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    """Server health check with system info."""
    import subprocess as _sp
    gpu_available = False
    gpu_name = None
    # Probe the pipeline Python (which has torch) for GPU info
    try:
        r = _sp.run(
            [PIPELINE_PYTHON, "-c",
             "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"],
            capture_output=True, text=True, timeout=15,
        )
        lines = r.stdout.strip().splitlines()
        if lines and lines[0].strip() == "True":
            gpu_available = True
            gpu_name = lines[1].strip() if len(lines) > 1 else "Unknown"
    except Exception:
        pass

    model_path = os.path.join(PROJECT_DIR, "checkpoints", "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")

    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "pipeline_python": PIPELINE_PYTHON,
        "model_loaded": os.path.isfile(model_path),
        "active_jobs": sum(1 for j in jobs.values() if j["status"] == "running"),
        "timestamp": datetime.now().isoformat(),
    }


# ──────────────────────────────────────────────────────────────────────
# Upload
# ──────────────────────────────────────────────────────────────────────

MAX_UPLOAD_BYTES = 1024 * 1024 * 1024  # 1 GB


@app.post("/api/upload")
async def upload_video(
    file: UploadFile = File(...),
    fps: float = Form(default=2.0),
):
    """Upload a video file and create a new reconstruction job."""
    # Validate file type
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(400, "Unsupported file type. Use MP4, AVI, MOV, or MKV.")

    job_id = str(uuid.uuid4())
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # Save uploaded video
    video_path = os.path.join(job_dir, "video.mp4")
    total_bytes = 0

    with open(video_path, "wb") as f:
        while chunk := await file.read(8192):
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                f.close()
                os.remove(video_path)
                raise HTTPException(413, f"File exceeds {MAX_UPLOAD_BYTES // (1024*1024)}MB limit.")
            f.write(chunk)

    # Register job
    jobs[job_id] = {
        "id": job_id,
        "status": "uploaded",
        "stage": None,
        "stages": {},
        "created_at": datetime.now().isoformat(),
        "video_path": video_path,
        "video_size_mb": round(total_bytes / (1024 * 1024), 2),
        "fps": fps,
        "logs": [],
        "metadata": {},
        "result": {},
    }

    # Save job state to disk
    _save_job_state(job_id)

    return {"job_id": job_id, "status": "uploaded", "video_size_mb": jobs[job_id]["video_size_mb"]}


# ──────────────────────────────────────────────────────────────────────
# Job Management
# ──────────────────────────────────────────────────────────────────────

@app.post("/api/jobs/{job_id}/start")
async def start_job(job_id: str):
    """Start the reconstruction pipeline for an uploaded job."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    if job["status"] == "running":
        raise HTTPException(409, "Job is already running")

    job["status"] = "running"
    job["started_at"] = datetime.now().isoformat()
    _save_job_state(job_id)

    # Launch pipeline as background task
    asyncio.create_task(_run_pipeline_subprocess(job_id))

    return {"job_id": job_id, "status": "running"}


@app.get("/api/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """Get the current status of a job."""
    if job_id not in jobs:
        # Try to load from disk
        state_path = os.path.join(JOBS_DIR, job_id, "status.json")
        if os.path.isfile(state_path):
            with open(state_path, "r") as f:
                jobs[job_id] = json.load(f)
        else:
            raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    return {
        "id": job["id"],
        "status": job["status"],
        "stage": job.get("stage"),
        "stages": job.get("stages", {}),
        "created_at": job["created_at"],
        "video_size_mb": job.get("video_size_mb"),
        "fps": job.get("fps"),
        "metadata": job.get("metadata", {}),
        "result": job.get("result", {}),
    }


@app.get("/api/jobs")
async def list_jobs():
    """List all known jobs with summary info."""
    return {"jobs": [
        {
            "id": j["id"],
            "status": j["status"],
            "created_at": j["created_at"],
            "video_size_mb": j.get("video_size_mb"),
            "fps": j.get("fps"),
            "stage": j.get("stage"),
            "completed_at": j.get("completed_at"),
            "video_filename": os.path.basename(j.get("video_path", "")),
        }
        for j in sorted(jobs.values(), key=lambda x: x.get("created_at", ""), reverse=True)
    ]}


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and all its files from disk."""
    import shutil
    job_dir = os.path.join(JOBS_DIR, job_id)
    if os.path.isdir(job_dir):
        shutil.rmtree(job_dir, ignore_errors=True)
    jobs.pop(job_id, None)
    ws_connections.pop(job_id, None)
    return {"deleted": True}


# ──────────────────────────────────────────────────────────────────────
# File Serving
# ──────────────────────────────────────────────────────────────────────

@app.get("/api/jobs/{job_id}/model")
async def get_job_model(job_id: str):
    """Serve the GLB model for a completed job."""
    glb_path = os.path.join(JOBS_DIR, job_id, "output", "deliverables", "model.glb")
    if not os.path.isfile(glb_path):
        raise HTTPException(404, "Model not ready or not found")
    return FileResponse(glb_path, media_type="model/gltf-binary", filename="model.glb")


@app.get("/api/jobs/{job_id}/pointcloud")
async def get_job_pointcloud(job_id: str):
    """Serve the PLY point cloud for a completed job."""
    ply_path = os.path.join(JOBS_DIR, job_id, "output", "point_cloud.ply")
    if not os.path.isfile(ply_path):
        raise HTTPException(404, "Point cloud not ready or not found")
    return FileResponse(ply_path, media_type="application/octet-stream", filename="point_cloud.ply")


@app.get("/api/jobs/{job_id}/metadata")
async def get_job_metadata(job_id: str):
    """Return extracted GPS/telemetry metadata for a job."""
    metadata_dir = os.path.join(JOBS_DIR, job_id, "metadata")
    result = {"gps": None, "telemetry": None}

    gps_path = os.path.join(metadata_dir, "gps.json")
    if os.path.isfile(gps_path):
        with open(gps_path, "r") as f:
            result["gps"] = json.load(f)

    telemetry_path = os.path.join(metadata_dir, "telemetry.json")
    if os.path.isfile(telemetry_path):
        with open(telemetry_path, "r") as f:
            result["telemetry"] = json.load(f)

    return result


@app.get("/api/jobs/{job_id}/frames")
async def get_job_frames(job_id: str):
    """List extracted frame thumbnails."""
    frames_dir = os.path.join(JOBS_DIR, job_id, "frames")
    if not os.path.isdir(frames_dir):
        return {"frames": []}

    frames = sorted(
        f for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    return {
        "frames": frames,
        "count": len(frames),
        "base_url": f"/api/jobs/{job_id}/frames/file",
    }


@app.get("/api/jobs/{job_id}/frames/file/{filename}")
async def get_frame_file(job_id: str, filename: str):
    """Serve an individual extracted frame."""
    frame_path = os.path.join(JOBS_DIR, job_id, "frames", filename)
    if not os.path.isfile(frame_path):
        raise HTTPException(404, "Frame not found")
    return FileResponse(frame_path, media_type="image/jpeg")


@app.get("/api/jobs/{job_id}/gps")
async def get_job_gps(job_id: str):
    """Return GPS track as GeoJSON for map rendering."""
    gps_path = os.path.join(JOBS_DIR, job_id, "metadata", "gps.json")
    # Also check the project-level CSV for built-in data
    gps_csv_path = os.path.join(PROJECT_DIR, "drone_flight_gps.csv")

    gps_points = []
    if os.path.isfile(gps_path):
        with open(gps_path, "r") as f:
            gps_points = json.load(f)
    elif os.path.isfile(gps_csv_path):
        import csv
        with open(gps_csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    gps_points.append({
                        "filename": row["filename"],
                        "latitude": float(row["latitude"]),
                        "longitude": float(row["longitude"]),
                        "altitude": float(row["altitude"]),
                    })
                except (ValueError, KeyError):
                    pass

    if not gps_points:
        return {"type": "FeatureCollection", "features": [], "points": []}

    # Build GeoJSON
    coordinates = [[p["longitude"], p["latitude"], p.get("altitude", 0)] for p in gps_points]
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coordinates},
            "properties": {
                "name": "Flight Path",
                "point_count": len(gps_points),
                "alt_min": min(p.get("altitude", 0) for p in gps_points),
                "alt_max": max(p.get("altitude", 0) for p in gps_points),
            },
        }
    ]

    return {
        "type": "FeatureCollection",
        "features": features,
        "points": gps_points,
        "bounds": {
            "lat_min": min(p["latitude"] for p in gps_points),
            "lat_max": max(p["latitude"] for p in gps_points),
            "lon_min": min(p["longitude"] for p in gps_points),
            "lon_max": max(p["longitude"] for p in gps_points),
        },
    }


@app.get("/api/jobs/{job_id}/outputs")
async def get_job_outputs(job_id: str):
    """List all output files for a job with file sizes."""
    output_dir = os.path.join(JOBS_DIR, job_id, "output")
    deliverables_dir = os.path.join(output_dir, "deliverables")

    files = []
    for search_dir in [output_dir, deliverables_dir]:
        if not os.path.isdir(search_dir):
            continue
        for fname in os.listdir(search_dir):
            fpath = os.path.join(search_dir, fname)
            if os.path.isfile(fpath):
                rel = os.path.relpath(fpath, os.path.join(JOBS_DIR, job_id, "output"))
                files.append({
                    "filename": fname,
                    "path": rel.replace("\\", "/"),
                    "size_bytes": os.path.getsize(fpath),
                    "size_mb": round(os.path.getsize(fpath) / (1024 * 1024), 2),
                    "download_url": f"/api/jobs/{job_id}/outputs/{fname}",
                })
    return {"files": files}


@app.get("/api/jobs/{job_id}/outputs/{filename}")
async def get_job_output_file(job_id: str, filename: str):
    """Serve any output file from a job."""
    # Search in output/ and output/deliverables/
    for subdir in ["", "deliverables"]:
        fpath = os.path.join(JOBS_DIR, job_id, "output", subdir, filename)
        if os.path.isfile(fpath):
            return FileResponse(fpath, filename=filename)
    raise HTTPException(404, "Output file not found")


# ──────────────────────────────────────────────────────────────────────
# Demo Gallery — browse and manage saved reconstructions
# ──────────────────────────────────────────────────────────────────────

DEMOS_DIR = os.path.join(PROJECT_DIR, "demos")
os.makedirs(DEMOS_DIR, exist_ok=True)


def _load_demos():
    """Load all demo metadata from disk."""
    demos = []
    # Built-in demo (from drone_deliverables/)
    builtin_glb = os.path.join(DELIVERABLES_DIR, "model_viewer.glb")
    builtin_ply = os.path.join(PROJECT_DIR, "dust3r_multiview_colored.ply")
    if os.path.isfile(builtin_glb):
        demos.append({
            "id": "builtin",
            "name": "AGZ Drone Flight (Built-in)",
            "created_at": "2026-01-01T00:00:00",
            "source": "builtin",
            "has_model": True,
            "has_pointcloud": os.path.isfile(builtin_ply),
        })
    # Saved demos from demos/ directory
    if os.path.isdir(DEMOS_DIR):
        for fname in sorted(os.listdir(DEMOS_DIR)):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(DEMOS_DIR, fname), "r") as f:
                        meta = json.load(f)
                        meta["has_model"] = os.path.isfile(meta.get("glb_path", ""))
                        meta["has_pointcloud"] = os.path.isfile(meta.get("ply_path", ""))
                        demos.append(meta)
                except Exception:
                    pass
    return demos


@app.get("/api/demos")
async def list_demos():
    """List all available demos (built-in + saved from completed jobs)."""
    return {"demos": _load_demos()}


@app.post("/api/jobs/{job_id}/save-demo")
async def save_job_as_demo(job_id: str, name: str = Form(default=None)):
    """Save a completed job as a named demo."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] != "complete":
        raise HTTPException(400, "Job is not complete yet")

    demo_id = job_id[:8]
    demo_name = name or f"Reconstruction {demo_id}"

    # Resolve file paths
    glb_path = os.path.join(JOBS_DIR, job_id, "output", "deliverables", "model.glb")
    ply_path = os.path.join(JOBS_DIR, job_id, "output", "point_cloud.ply")

    meta = {
        "id": demo_id,
        "job_id": job_id,
        "name": demo_name,
        "created_at": job.get("completed_at", job["created_at"]),
        "source": "job",
        "glb_path": glb_path,
        "ply_path": ply_path,
        "video_size_mb": job.get("video_size_mb"),
        "fps": job.get("fps"),
    }

    meta_path = os.path.join(DEMOS_DIR, f"{demo_id}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {"id": demo_id, "name": demo_name, "saved": True}


@app.delete("/api/demos/{demo_id}")
async def delete_demo(demo_id: str):
    """Remove a demo from the gallery (does not delete the job files)."""
    if demo_id == "builtin":
        raise HTTPException(400, "Cannot delete the built-in demo")
    meta_path = os.path.join(DEMOS_DIR, f"{demo_id}.json")
    if not os.path.isfile(meta_path):
        raise HTTPException(404, "Demo not found")
    os.remove(meta_path)
    return {"deleted": True}


@app.get("/api/demos/{demo_id}/model")
async def get_demo_model(demo_id: str):
    """Serve a demo's GLB model."""
    if demo_id == "builtin":
        glb_path = os.path.join(DELIVERABLES_DIR, "model_viewer.glb")
    else:
        meta_path = os.path.join(DEMOS_DIR, f"{demo_id}.json")
        if not os.path.isfile(meta_path):
            raise HTTPException(404, "Demo not found")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        glb_path = meta.get("glb_path", "")
    if not os.path.isfile(glb_path):
        raise HTTPException(404, "Model file not found")
    return FileResponse(glb_path, media_type="model/gltf-binary", filename="model.glb")


@app.get("/api/demos/{demo_id}/pointcloud")
async def get_demo_pointcloud(demo_id: str):
    """Serve a demo's PLY point cloud."""
    if demo_id == "builtin":
        ply_path = os.path.join(PROJECT_DIR, "dust3r_multiview_colored.ply")
    else:
        meta_path = os.path.join(DEMOS_DIR, f"{demo_id}.json")
        if not os.path.isfile(meta_path):
            raise HTTPException(404, "Demo not found")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        ply_path = meta.get("ply_path", "")
    if not os.path.isfile(ply_path):
        raise HTTPException(404, "Point cloud file not found")
    return FileResponse(ply_path, media_type="application/octet-stream", filename="pointcloud.ply")


# Keep old endpoints for backward compat
@app.get("/api/demo/model")
async def get_legacy_demo_model():
    return await get_demo_model("builtin")

@app.get("/api/demo/pointcloud")
async def get_legacy_demo_pointcloud():
    return await get_demo_pointcloud("builtin")


# ──────────────────────────────────────────────────────────────────────
# WebSocket — real-time pipeline progress
# ──────────────────────────────────────────────────────────────────────

@app.websocket("/ws/jobs/{job_id}")
async def websocket_job(websocket: WebSocket, job_id: str):
    """Stream real-time pipeline progress for a job."""
    await websocket.accept()

    if job_id not in ws_connections:
        ws_connections[job_id] = set()
    ws_connections[job_id].add(websocket)

    try:
        # Send current job state on connect
        if job_id in jobs:
            await websocket.send_json({
                "type": "state",
                "job": {
                    "id": jobs[job_id]["id"],
                    "status": jobs[job_id]["status"],
                    "stage": jobs[job_id].get("stage"),
                    "stages": jobs[job_id].get("stages", {}),
                },
            })
            # Send buffered logs
            for log_entry in jobs[job_id].get("logs", [])[-50:]:
                await websocket.send_json(log_entry)

        # Keep connection alive until client disconnects
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_connections[job_id].discard(websocket)


async def _broadcast_to_job(job_id: str, message: dict):
    """Send a message to all WebSocket clients watching a job."""
    if job_id not in ws_connections:
        return
    dead = set()
    for ws in ws_connections[job_id]:
        try:
            await ws.send_json(message)
        except Exception:
            dead.add(ws)
    ws_connections[job_id] -= dead


# ──────────────────────────────────────────────────────────────────────
# Pipeline Subprocess Runner
# ──────────────────────────────────────────────────────────────────────

async def _run_pipeline_subprocess(job_id: str):
    """Spawn run_pipeline.py as a subprocess and stream its JSON output."""
    job = jobs[job_id]
    pipeline_script = os.path.join(PROJECT_DIR, "run_pipeline.py")

    cmd = [
        PIPELINE_PYTHON, pipeline_script,
        "--job-id", job_id,
        "--video", job["video_path"],
        "--fps", str(job["fps"]),
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=PROJECT_DIR,
        )

        async for line in process.stdout:
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue

            # Try to parse as JSON event
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                # Plain text log line
                event = {"type": "log", "message": text}

            # Update job state based on event type
            if event.get("type") == "stage":
                stage_id = event.get("stage")
                status = event.get("status")
                job["stage"] = stage_id
                if "stages" not in job:
                    job["stages"] = {}
                job["stages"][stage_id] = {
                    "status": status,
                    "result": event.get("result"),
                    "error": event.get("error"),
                }
                # If metadata stage is complete, store metadata in job
                if stage_id == "metadata_extraction" and status == "complete":
                    gps_path = os.path.join(JOBS_DIR, job_id, "metadata", "gps.json")
                    if os.path.isfile(gps_path):
                        with open(gps_path, "r") as f:
                            job["metadata"]["gps"] = json.load(f)

            elif event.get("type") == "pipeline":
                status = event.get("status")
                job["status"] = "complete" if status == "complete" else status
                if status == "complete":
                    job["completed_at"] = datetime.now().isoformat()
                    # Collect result file paths
                    job["result"] = _collect_job_results(job_id)

            # Buffer log entry
            job["logs"].append(event)
            # Keep last 500 log entries in memory
            if len(job["logs"]) > 500:
                job["logs"] = job["logs"][-500:]

            # Broadcast to WebSocket clients
            await _broadcast_to_job(job_id, event)

        await process.wait()

        # If process exited with error and job wasn't marked failed
        if process.returncode != 0 and job["status"] != "failed":
            job["status"] = "failed"
            error_event = {"type": "pipeline", "status": "failed",
                           "error": f"Process exited with code {process.returncode}"}
            job["logs"].append(error_event)
            await _broadcast_to_job(job_id, error_event)

    except Exception as e:
        job["status"] = "failed"
        error_event = {"type": "pipeline", "status": "failed", "error": str(e)}
        job["logs"].append(error_event)
        await _broadcast_to_job(job_id, error_event)

    finally:
        _save_job_state(job_id)


def _collect_job_results(job_id: str) -> dict:
    """Collect paths and sizes of output files for a completed job."""
    result = {}
    files_to_check = {
        "point_cloud": os.path.join(JOBS_DIR, job_id, "output", "point_cloud.ply"),
        "mesh": os.path.join(JOBS_DIR, job_id, "output", "mesh.obj"),
        "glb": os.path.join(JOBS_DIR, job_id, "output", "deliverables", "model.glb"),
        "las": os.path.join(JOBS_DIR, job_id, "output", "deliverables", "point_cloud.las"),
    }
    for key, path in files_to_check.items():
        if os.path.isfile(path):
            result[key] = {
                "available": True,
                "size_mb": round(os.path.getsize(path) / (1024 * 1024), 2),
            }
        else:
            result[key] = {"available": False}
    return result


def _save_job_state(job_id: str):
    """Persist job state to disk."""
    if job_id not in jobs:
        return
    state_path = os.path.join(JOBS_DIR, job_id, "status.json")
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    # Write a serializable subset (skip log entries to save space)
    state = {k: v for k, v in jobs[job_id].items() if k != "logs"}
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ──────────────────────────────────────────────────────────────────────
# Startup Events
# ──────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Load previously saved job states on startup."""
    if not os.path.isdir(JOBS_DIR):
        return
    for name in os.listdir(JOBS_DIR):
        state_path = os.path.join(JOBS_DIR, name, "status.json")
        if os.path.isfile(state_path):
            try:
                with open(state_path, "r") as f:
                    job_state = json.load(f)
                    job_state["logs"] = []  # Don't reload logs into memory
                    jobs[job_state["id"]] = job_state
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
