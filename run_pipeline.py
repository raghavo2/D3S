"""
Pipeline Orchestrator — runs the full reconstruction pipeline for a job.
Outputs structured JSON lines to stdout for the server to capture and
stream via WebSocket.

V3 upgrades:
  - Stage 0: ROS Bag Extraction (optional, when .bag file is provided)
  - Stage 7.5: Metric Scaling (GPS-based scale factor computation)
  - Higher accuracy defaults (resolution, scene graph, alignment iters)
  - --bag CLI argument for ROS bag input
  - Auto-detect: bag-only mode extracts frames + telemetry from bag

Usage:
    python run_pipeline.py --job-id <uuid> --video <path> [--fps 2.0]
    python run_pipeline.py --job-id <uuid> --images <dir> [--fps 2.0]
    python run_pipeline.py --job-id <uuid> --bag <path> [--fps 2.0]
    python run_pipeline.py --job-id <uuid> --video <path> --bag <path> [--fps 2.0]
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

def run_pipeline(job_id, video_path=None, images_dir=None, bag_path=None, fps=2.0):
    """Run the full pipeline for a given job."""

    from pipeline.config import ProjectConfig
    from pipeline.frames import extract_frames
    from pipeline.quality import score_all_frames, filter_frames
    from pipeline.keyframes import select_keyframes, copy_keyframes
    from pipeline.detection import detect_objects, save_detections
    from pipeline.masking import generate_masks
    from pipeline.colmap import run_colmap
    from pipeline.metadata import (
        extract_gps_from_images,
        extract_telemetry_from_bag,
        extract_full_telemetry_from_bag,
        extract_frames_from_bag,
    )
    from pipeline.reconstruct import reconstruct_point_cloud
    from pipeline.mesh import generate_mesh
    from pipeline.export import export_deliverables
    from pipeline.scale import (
        compute_scale_factor,
        save_scale_info,
        compute_dimensions_from_points,
    )

    config = ProjectConfig()

    frames_dir = config.job_frames_dir(job_id)
    keyframes_dir = config.job_keyframes_dir(job_id)
    masks_dir = config.job_masks_dir(job_id)
    colmap_dir = config.job_colmap_dir(job_id)
    metadata_dir = config.job_metadata_dir(job_id)
    output_dir = config.job_output_dir(job_id)
    deliverables_dir = config.job_deliverables_dir(job_id)

    output_ply = os.path.join(output_dir, "point_cloud.ply")
    output_obj = os.path.join(output_dir, "mesh.obj")

    # Track telemetry data extracted from bag (used for GPS scaling)
    bag_telemetry = None
    bag_gps_data = None

    emit("pipeline", status="started", job_id=job_id)

    # ── Stage 0: ROS Bag Extraction (optional) ───────────────────────
    if bag_path and os.path.isfile(bag_path):
        emit("stage", stage="rosbag_extraction", status="running")
        try:
            # Extract full GPS + IMU telemetry
            bag_telemetry = extract_full_telemetry_from_bag(
                bag_path,
                output_json=os.path.join(metadata_dir, "telemetry_full.json"),
                on_log=make_logger("rosbag_extraction"),
            )

            if bag_telemetry:
                bag_gps_data = bag_telemetry.get("gps_data", [])

                # If bag has image topics AND no video was provided,
                # extract frames from the bag
                if bag_telemetry.get("has_images") and not video_path:
                    make_logger("rosbag_extraction")(
                        "No video provided — extracting frames from bag image topics"
                    )
                    bag_frames_result = extract_frames_from_bag(
                        bag_path, frames_dir, fps=fps,
                        on_log=make_logger("rosbag_extraction"),
                    )
                    if bag_frames_result and bag_frames_result["frame_count"] > 0:
                        images_dir = frames_dir
                        make_logger("rosbag_extraction")(
                            f"Extracted {bag_frames_result['frame_count']} frames from bag"
                        )

                # Save GPS as the standard gps.json for downstream compatibility
                if bag_gps_data:
                    gps_for_json = []
                    for g in bag_gps_data:
                        gps_for_json.append({
                            "latitude": g["latitude"],
                            "longitude": g["longitude"],
                            "altitude": g.get("altitude", 0),
                        })
                    gps_json_path = os.path.join(metadata_dir, "gps.json")
                    with open(gps_json_path, "w", encoding="utf-8") as f:
                        json.dump(gps_for_json, f, indent=2)
                    make_logger("rosbag_extraction")(
                        f"Saved {len(gps_for_json)} GPS points from bag to gps.json"
                    )

                emit("stage", stage="rosbag_extraction", status="complete", result={
                    "gps_messages": bag_telemetry.get("gps_count", 0),
                    "imu_messages": bag_telemetry.get("imu_count", 0),
                    "has_images": bag_telemetry.get("has_images", False),
                    "topics": len(bag_telemetry.get("topics", [])),
                    "duration_s": bag_telemetry.get("time_range_s", 0),
                    "frames_extracted": images_dir == frames_dir,
                })
            else:
                emit("stage", stage="rosbag_extraction", status="warning",
                     error="Could not parse bag telemetry")

        except Exception as e:
            emit("stage", stage="rosbag_extraction", status="warning",
                 error=str(e))
    else:
        emit("stage", stage="rosbag_extraction", status="skipped",
             result={"message": "No ROS bag provided"})

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
        emit("pipeline", status="failed", error="No video, image directory, or bag file provided")
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
        # Keyframe selection is non-critical -- fall back to using all frames
        emit("stage", stage="keyframe_selection", status="warning", error=str(e))

    # == Stage 4: Object Detection and Dynamic Masking ====================
    mask_dir_for_reconstruction = None
    emit("stage", stage="object_detection", status="running")
    try:
        # Check if YOLO model exists
        yolo_model_path = os.path.join(config.project_dir, "models", "D3S_YOLO_80.pt")
        if not os.path.isfile(yolo_model_path):
            emit("stage", stage="object_detection", status="skipped",
                 result={"message": "YOLO model not found -- skipping detection"})
        else:
            det_result = detect_objects(
                images_dir,
                model_path=yolo_model_path,
                confidence=config.DEFAULT_YOLO_CONFIDENCE,
                movement_threshold=config.DEFAULT_MOVEMENT_THRESHOLD,
                orb_features=config.DEFAULT_ORB_FEATURES,
                on_log=make_logger("object_detection"),
            )

            # Save detections JSON
            det_json_path = os.path.join(
                config.job_output_dir(job_id), "detections.json"
            )
            save_detections(det_result, det_json_path,
                           on_log=make_logger("object_detection"))

            # Generate masks if dynamic objects were found
            if det_result["dynamic_count"] > 0:
                mask_result = generate_masks(
                    images_dir,
                    det_result["detections"],
                    masks_dir,
                    padding_px=config.DEFAULT_MASK_PADDING_PX,
                    on_log=make_logger("object_detection"),
                )
                mask_dir_for_reconstruction = masks_dir
            else:
                mask_result = {"mask_count": 0, "masked_frames": 0}

            emit("stage", stage="object_detection", status="complete", result={
                "dynamic_objects": det_result["dynamic_count"],
                "static_objects": det_result["static_count"],
                "frames_with_dynamic": det_result["frames_with_dynamic"],
                "masks_generated": mask_result.get("masked_frames", 0),
                "class_summary": det_result["class_summary"],
                "tracking_stats": det_result.get("tracking_stats", {}),
            })
    except Exception as e:
        # Object detection is non-critical -- continue without masks
        emit("stage", stage="object_detection", status="warning", error=str(e))

    # == Stage 5: COLMAP Pose Estimation ===================================
    colmap_poses = None
    emit("stage", stage="colmap_pose_estimation", status="running")
    try:
        if not os.path.isfile(config.colmap_exe):
            emit("stage", stage="colmap_pose_estimation", status="skipped",
                 result={"message": "COLMAP executable not found -- skipping"})
        else:
            colmap_result = run_colmap(
                images_dir,
                colmap_dir,
                colmap_exe=config.colmap_exe,
                feature_type=config.DEFAULT_COLMAP_FEATURE_TYPE,
                matching_type=config.DEFAULT_COLMAP_MATCHING_TYPE,
                on_log=make_logger("colmap_pose_estimation"),
            )

            if colmap_result["success"]:
                colmap_poses = colmap_result.get("poses_json_path")
                emit("stage", stage="colmap_pose_estimation", status="complete", result={
                    "registered_images": colmap_result["registered_images"],
                    "total_images": colmap_result["total_images"],
                    "camera_models": len(colmap_result.get("cameras", {})),
                })
            else:
                emit("stage", stage="colmap_pose_estimation", status="warning",
                     error=colmap_result.get("error", "Unknown error"))
    except Exception as e:
        # COLMAP is non-critical -- continue with DUSt3R-only reconstruction
        emit("stage", stage="colmap_pose_estimation", status="warning", error=str(e))

    # == Stage 6: Metadata Extraction ======================================
    emit("stage", stage="metadata_extraction", status="running")
    gps_data_for_scaling = []
    try:
        gps_data = extract_gps_from_images(
            images_dir,
            output_csv=os.path.join(metadata_dir, "gps.csv"),
            output_json=os.path.join(metadata_dir, "gps.json"),
            on_log=make_logger("metadata_extraction"),
        )
        gps_data_for_scaling = gps_data  # EXIF GPS data

        # Attempt legacy telemetry extraction if bag file exists
        bag_for_telemetry = bag_path
        if not bag_for_telemetry:
            bag_for_telemetry = os.path.join(config.project_dir, "AGZ_subset", "AGZ.bag")
        
        telemetry = None
        if bag_for_telemetry and os.path.isfile(bag_for_telemetry) and not bag_telemetry:
            telemetry = extract_telemetry_from_bag(
                bag_for_telemetry,
                output_json=os.path.join(metadata_dir, "telemetry.json"),
                on_log=make_logger("metadata_extraction"),
            )

        # If bag telemetry had GPS but EXIF didn't, use bag GPS
        if not gps_data_for_scaling and bag_gps_data:
            gps_data_for_scaling = bag_gps_data
            make_logger("metadata_extraction")(
                f"Using {len(bag_gps_data)} GPS points from ROS bag for scaling"
            )

        emit("stage", stage="metadata_extraction", status="complete",
             result={"gps_points": len(gps_data),
                      "bag_gps_points": len(bag_gps_data) if bag_gps_data else 0,
                      "has_telemetry": telemetry is not None or bag_telemetry is not None})
    except Exception as e:
        # Metadata is non-critical -- log and continue
        emit("stage", stage="metadata_extraction", status="warning", error=str(e))

    # == Stage 7: Point Cloud Generation ===================================
    emit("stage", stage="point_cloud_generation", status="running")
    recon_result = None
    try:
        recon_result = reconstruct_point_cloud(
            images_dir, output_ply,
            model_path=config.model_path,
            mask_dir=mask_dir_for_reconstruction,
            colmap_poses_json=colmap_poses,
            on_log=make_logger("point_cloud_generation"),
        )
        emit("stage", stage="point_cloud_generation", status="complete", result={
            "point_count": recon_result["point_count"],
            "masked_points": recon_result.get("masked_points", 0),
            "low_confidence_points": recon_result.get("low_confidence_points", 0),
            "camera_centers_extracted": len(recon_result.get("camera_centers", [])),
        })
    except Exception as e:
        emit("stage", stage="point_cloud_generation", status="error",
             error=str(e), traceback=traceback.format_exc())
        emit("pipeline", status="failed", error=str(e))
        return

    # ── Point cloud is ready — notify frontend for early preview ──
    emit("pipeline", status="viewable", job_id=job_id)

    # Stages 7.5+ run in the background. Failures are non-fatal since
    # the point cloud is already viewable.
    post_viewable_ok = True

    # == Stage 7.5: Metric Scaling (V3) ====================================
    scale_factor = None
    emit("stage", stage="metric_scaling", status="running")
    try:
        camera_centers = recon_result.get("camera_centers", [])
        
        if gps_data_for_scaling and camera_centers:
            # Match GPS positions to camera centers by filename
            scale_info = compute_scale_factor(
                gps_positions=gps_data_for_scaling,
                reconstruction_positions=camera_centers,
                min_gps_distance=config.DEFAULT_GPS_MIN_DISTANCE,
                min_pairs=config.DEFAULT_GPS_SCALE_PAIRS_MIN,
                on_log=make_logger("metric_scaling"),
            )

            if scale_info and scale_info.get("reliable"):
                scale_factor = scale_info["scale_factor"]

                # Compute real-world dimensions
                import numpy as np
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(output_ply)
                points = np.asarray(pcd.points)
                dimensions = compute_dimensions_from_points(points, scale_factor)
                scale_info["dimensions_m"] = dimensions

                make_logger("metric_scaling")(
                    f"Structure dimensions: "
                    f"{dimensions['width_m']:.1f}m × "
                    f"{dimensions['height_m']:.1f}m × "
                    f"{dimensions['depth_m']:.1f}m"
                )
            elif scale_info:
                make_logger("metric_scaling")(
                    f"Scale factor computed but unreliable "
                    f"({scale_info.get('gps_pairs_used', 0)} pairs)"
                )
                scale_factor = scale_info.get("scale_factor")

            # Save scale info regardless of reliability
            scale_info_path = os.path.join(output_dir, "scale_info.json")
            save_scale_info(scale_info, scale_info_path,
                           on_log=make_logger("metric_scaling"))

            emit("stage", stage="metric_scaling", status="complete", result={
                "scale_factor": scale_info.get("scale_factor", 1.0),
                "reliable": scale_info.get("reliable", False),
                "gps_pairs_used": scale_info.get("gps_pairs_used", 0),
                "median_residual_m": scale_info.get("median_residual_m", 0),
                "method": scale_info.get("method", "unknown"),
            })
        else:
            reason = []
            if not gps_data_for_scaling:
                reason.append("no GPS data available")
            if not camera_centers:
                reason.append("no camera centers extracted")
            emit("stage", stage="metric_scaling", status="skipped",
                 result={"message": f"Skipped: {', '.join(reason)}"})

    except Exception as e:
        emit("stage", stage="metric_scaling", status="warning", error=str(e))

    # == Stage 8: Surface Meshing ==========================================
    emit("stage", stage="surface_meshing", status="running")
    try:
        result = generate_mesh(
            output_ply, output_obj,
            scale_factor=scale_factor,
            on_log=make_logger("surface_meshing"),
        )
        emit("stage", stage="surface_meshing", status="complete", result={
            "vertex_count": result["vertex_count"],
            "triangle_count": result["triangle_count"],
            "dimensions": result.get("dimensions"),
        })
    except Exception as e:
        emit("stage", stage="surface_meshing", status="error",
             error=str(e), traceback=traceback.format_exc())
        post_viewable_ok = False

    # == Stage 9: Deliverable Export ========================================
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
        post_viewable_ok = False

    if post_viewable_ok:
        emit("pipeline", status="complete", job_id=job_id)
    else:
        emit("pipeline", status="complete_partial", job_id=job_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D3S Pipeline Orchestrator")
    parser.add_argument("--job-id", required=True, help="Unique job identifier")
    parser.add_argument("--video", default=None, help="Path to input video")
    parser.add_argument("--images", default=None, help="Path to image directory")
    parser.add_argument("--bag", default=None, help="Path to ROS bag file (V3)")
    parser.add_argument("--fps", type=float, default=2.0, help="Frame extraction FPS")

    args = parser.parse_args()
    run_pipeline(
        args.job_id,
        video_path=args.video,
        images_dir=args.images,
        bag_path=args.bag,
        fps=args.fps,
    )
