"""
Pipeline Orchestrator — runs the full reconstruction pipeline for a job.
Outputs structured JSON lines to stdout for the server to capture and
stream via WebSocket.

Usage:
    python run_pipeline.py --job-id <uuid> --video <path> [--fps 2.0]
    python run_pipeline.py --job-id <uuid> --images <dir> [--fps 2.0]
"""

import os
import sys
import json
import argparse
import traceback
from datetime import datetime


def emit(event_type, **data):
    """Print a structured JSON event to stdout for the server to capture."""
    event = {"type": event_type, "timestamp": datetime.now().isoformat(), **data}
    print(json.dumps(event), flush=True)


def make_logger(stage_id):
    """Create a log callback that emits structured JSON log lines."""
    def log(message):
        emit("log", stage=stage_id, message=message)
    return log

def run_pipeline(job_id, video_path=None, images_dir=None, fps=2.0):
    """Run the full pipeline for a given job."""

    # Import pipeline modules (lazy — avoids loading heavy deps at parse time)
    from pipeline.config import ProjectConfig
    from pipeline.frames import extract_frames
    from pipeline.quality import score_all_frames, filter_frames
    from pipeline.keyframes import select_keyframes, copy_keyframes
    from pipeline.metadata import extract_gps_from_images, extract_telemetry_from_bag
    from pipeline.reconstruct import reconstruct_point_cloud
    from pipeline.mesh import generate_mesh
    from pipeline.export import export_deliverables

    config = ProjectConfig()

    frames_dir = config.job_frames_dir(job_id)
    keyframes_dir = config.job_keyframes_dir(job_id)
    metadata_dir = config.job_metadata_dir(job_id)
    output_dir = config.job_output_dir(job_id)
    deliverables_dir = config.job_deliverables_dir(job_id)

    output_ply = os.path.join(output_dir, "point_cloud.ply")
    output_obj = os.path.join(output_dir, "mesh.obj")

    emit("pipeline", status="started", job_id=job_id)

    # ── Stage 1: Frame Extraction ─────────────────────────────────────
    if video_path:
        emit("stage", stage="frame_extraction", status="running")
        try:
            result = extract_frames(
                video_path, frames_dir, fps=fps,
                on_log=make_logger("frame_extraction")
            )
            emit("stage", stage="frame_extraction", status="complete", result=result)
            images_dir = frames_dir
        except Exception as e:
            emit("stage", stage="frame_extraction", status="error", error=str(e))
            emit("pipeline", status="failed", error=str(e))
            return
    elif images_dir:
        emit("stage", stage="frame_extraction", status="skipped",
             result={"message": "Using provided image directory"})
    else:
        emit("pipeline", status="failed", error="No video or image directory provided")
        return

    # ── Stage 2: Quality Filtering ────────────────────────────────────
    emit("stage", stage="quality_filtering", status="running")
    try:
        quality_scores = score_all_frames(
            images_dir,
            blur_reject_percentile=config.DEFAULT_BLUR_REJECT_PERCENTILE,
            max_clipping_ratio=config.DEFAULT_MAX_CLIPPING_RATIO,
            on_log=make_logger("quality_filtering"),
        )
        filter_result = filter_frames(
            images_dir, quality_scores,
            on_log=make_logger("quality_filtering"),
        )
        emit("stage", stage="quality_filtering", status="complete", result={
            "total_frames": len(quality_scores),
            "passed": filter_result["passed_count"],
            "rejected": filter_result["rejected_count"],
        })
    except Exception as e:
        # Quality filtering is non-critical — log warning and continue
        # with all frames if it fails
        emit("stage", stage="quality_filtering", status="warning", error=str(e))
        quality_scores = None
        filter_result = None

    # ── Stage 3: Keyframe Selection ───────────────────────────────────
    emit("stage", stage="keyframe_selection", status="running")
    try:
        kf_result = select_keyframes(
            images_dir,
            quality_scores=quality_scores,
            ssim_threshold=config.DEFAULT_SSIM_THRESHOLD,
            min_temporal_gap=config.DEFAULT_MIN_TEMPORAL_GAP,
            max_temporal_gap=config.DEFAULT_MAX_TEMPORAL_GAP,
            on_log=make_logger("keyframe_selection"),
        )
        copy_result = copy_keyframes(
            images_dir, kf_result["selected_filenames"], keyframes_dir,
            on_log=make_logger("keyframe_selection"),
        )

        # Save keyframe selection report for later analysis
        report_path = os.path.join(keyframes_dir, "selection_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(kf_result, f, indent=2)

        emit("stage", stage="keyframe_selection", status="complete", result={
            "original_frames": kf_result["original_count"],
            "selected_keyframes": kf_result["selected_count"],
            "retention_pct": round(100 * kf_result["selected_count"] / max(1, kf_result["original_count"]), 1),
        })

        # Point all downstream stages at the keyframes directory
        images_dir = keyframes_dir

    except Exception as e:
        # Keyframe selection is non-critical — fall back to using all frames
        emit("stage", stage="keyframe_selection", status="warning", error=str(e))

    # ── Stage 4: Metadata Extraction ──────────────────────────────────
    emit("stage", stage="metadata_extraction", status="running")
    try:
        gps_data = extract_gps_from_images(
            images_dir,
            output_csv=os.path.join(metadata_dir, "gps.csv"),
            output_json=os.path.join(metadata_dir, "gps.json"),
            on_log=make_logger("metadata_extraction"),
        )
        # Attempt telemetry extraction if bag file exists
        bag_path = os.path.join(config.project_dir, "AGZ_subset", "AGZ.bag")
        telemetry = None
        if os.path.isfile(bag_path):
            telemetry = extract_telemetry_from_bag(
                bag_path,
                output_json=os.path.join(metadata_dir, "telemetry.json"),
                on_log=make_logger("metadata_extraction"),
            )
        emit("stage", stage="metadata_extraction", status="complete",
             result={"gps_points": len(gps_data),
                      "has_telemetry": telemetry is not None})
    except Exception as e:
        # Metadata is non-critical — log and continue
        emit("stage", stage="metadata_extraction", status="warning", error=str(e))

    # ── Stage 5: Point Cloud Generation ───────────────────────────────
    emit("stage", stage="point_cloud_generation", status="running")
    try:
        result = reconstruct_point_cloud(
            images_dir, output_ply,
            model_path=config.model_path,
            on_log=make_logger("point_cloud_generation"),
        )
        emit("stage", stage="point_cloud_generation", status="complete", result=result)
    except Exception as e:
        emit("stage", stage="point_cloud_generation", status="error",
             error=str(e), traceback=traceback.format_exc())
        emit("pipeline", status="failed", error=str(e))
        return

    # ── Stage 6: Surface Meshing ──────────────────────────────────────
    emit("stage", stage="surface_meshing", status="running")
    try:
        result = generate_mesh(
            output_ply, output_obj,
            on_log=make_logger("surface_meshing"),
        )
        emit("stage", stage="surface_meshing", status="complete", result=result)
    except Exception as e:
        emit("stage", stage="surface_meshing", status="error",
             error=str(e), traceback=traceback.format_exc())
        emit("pipeline", status="failed", error=str(e))
        return

    # ── Stage 7: Deliverable Export ───────────────────────────────────
    emit("stage", stage="deliverable_export", status="running")
    try:
        result = export_deliverables(
            output_ply, output_obj, deliverables_dir,
            on_log=make_logger("deliverable_export"),
        )
        emit("stage", stage="deliverable_export", status="complete", result=result)
    except Exception as e:
        emit("stage", stage="deliverable_export", status="error",
             error=str(e), traceback=traceback.format_exc())
        emit("pipeline", status="failed", error=str(e))
        return

    emit("pipeline", status="complete", job_id=job_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D3S Pipeline Orchestrator")
    parser.add_argument("--job-id", required=True, help="Unique job identifier")
    parser.add_argument("--video", default=None, help="Path to input video")
    parser.add_argument("--images", default=None, help="Path to image directory")
    parser.add_argument("--fps", type=float, default=2.0, help="Frame extraction FPS")

    args = parser.parse_args()
    run_pipeline(args.job_id, video_path=args.video, images_dir=args.images, fps=args.fps)
