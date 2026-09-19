"""
Shared configuration for the D3S pipeline.
All paths are computed relative to the project root directory.
"""

import os


class ProjectConfig:
    """Centralized project paths and pipeline configuration."""

    def __init__(self, project_dir=None):
        # Project root is one level above the pipeline/ package
        if project_dir is None:
            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.project_dir = project_dir

        # --- Core paths ---
        self.checkpoints_dir = os.path.join(self.project_dir, "checkpoints")
        self.model_path = os.path.join(
            self.checkpoints_dir, "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
        )
        self.dust3r_dir = os.path.join(self.project_dir, "dust3r")

        # --- I/O paths ---
        self.uploads_dir = os.path.join(self.project_dir, "uploads")
        self.jobs_dir = os.path.join(self.project_dir, "jobs")
        self.deliverables_dir = os.path.join(self.project_dir, "drone_deliverables")

        # --- COLMAP paths (Windows) ---
        self.colmap_dir = os.path.join(self.project_dir, "colmap-x64-windows-cuda")
        self.colmap_exe = os.path.join(self.colmap_dir, "COLMAP.bat")

    # --- Pipeline stage definitions ---
    STAGES = [
        {"id": "frame_extraction", "label": "Frame Extraction", "icon": "🎞️"},
        {"id": "quality_filtering", "label": "Quality Filtering", "icon": "🔍"},
        {"id": "keyframe_selection", "label": "Keyframe Selection", "icon": "🎯"},
        {"id": "metadata_extraction", "label": "Metadata Extraction", "icon": "📡"},
        {"id": "point_cloud_generation", "label": "Point Cloud Generation", "icon": "☁️"},
        {"id": "surface_meshing", "label": "Surface Meshing", "icon": "🔺"},
        {"id": "deliverable_export", "label": "Deliverable Export", "icon": "📦"},
    ]

    # --- Defaults ---
    DEFAULT_FPS = 2.0
    DEFAULT_IMAGE_SIZE = 512
    DEFAULT_BATCH_SIZE = 1
    DEFAULT_ALIGNMENT_ITERS = 300
    MAX_UPLOAD_SIZE_MB = 1024  # 1 GB cap

    # --- Quality filtering defaults ---
    DEFAULT_BLUR_REJECT_PERCENTILE = 20   # Reject bottom 20% sharpness
    DEFAULT_MAX_CLIPPING_RATIO = 0.05     # Reject if >5% pixels clipped

    # --- Keyframe selection defaults ---
    DEFAULT_SSIM_THRESHOLD = 0.92         # SSIM above this = near-duplicate
    DEFAULT_MIN_TEMPORAL_GAP = 1          # Min frame gap between keyframes
    DEFAULT_MAX_TEMPORAL_GAP = 10         # Force-include if gap exceeds this

    def job_dir(self, job_id):
        """Return the directory for a specific job, creating it if needed."""
        path = os.path.join(self.jobs_dir, job_id)
        os.makedirs(path, exist_ok=True)
        return path

    def job_frames_dir(self, job_id):
        path = os.path.join(self.job_dir(job_id), "frames")
        os.makedirs(path, exist_ok=True)
        return path

    def job_keyframes_dir(self, job_id):
        """Return the directory for selected keyframes."""
        path = os.path.join(self.job_dir(job_id), "keyframes")
        os.makedirs(path, exist_ok=True)
        return path

    def job_metadata_dir(self, job_id):
        path = os.path.join(self.job_dir(job_id), "metadata")
        os.makedirs(path, exist_ok=True)
        return path

    def job_output_dir(self, job_id):
        path = os.path.join(self.job_dir(job_id), "output")
        os.makedirs(path, exist_ok=True)
        return path

    def job_deliverables_dir(self, job_id):
        path = os.path.join(self.job_dir(job_id), "output", "deliverables")
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def model_exists(self):
        return os.path.isfile(self.model_path)
