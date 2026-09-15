"""
D3S Pipeline Package
====================
Modularized 3D reconstruction pipeline for drone video processing.
Each module wraps an existing standalone script as an importable function
with configurable paths and progress callbacks.

Original standalone scripts are preserved unchanged for backward compatibility.
"""

from pipeline.config import ProjectConfig
from pipeline.frames import extract_frames
from pipeline.metadata import extract_gps_from_images, extract_telemetry_from_bag
from pipeline.reconstruct import reconstruct_point_cloud
from pipeline.mesh import generate_mesh
from pipeline.export import export_deliverables

__all__ = [
    "ProjectConfig",
    "extract_frames",
    "extract_gps_from_images",
    "extract_telemetry_from_bag",
    "reconstruct_point_cloud",
    "generate_mesh",
    "export_deliverables",
]
