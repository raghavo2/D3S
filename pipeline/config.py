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
        {"id": "rosbag_extraction", "label": "ROS Bag Extraction", "icon": "📡"},
        {"id": "frame_extraction", "label": "Frame Extraction", "icon": "🎞️"},
        {"id": "quality_filtering", "label": "Quality Filtering", "icon": "🔍"},
        {"id": "keyframe_selection", "label": "Keyframe Selection", "icon": "🎯"},
        {"id": "object_detection", "label": "Object Detection", "icon": "🚨"},
        {"id": "colmap_pose_estimation", "label": "COLMAP Pose Estimation", "icon": "📰"},
        {"id": "metadata_extraction", "label": "Metadata Extraction", "icon": "📡"},
        {"id": "point_cloud_generation", "label": "Point Cloud Generation", "icon": "☁️"},
        {"id": "metric_scaling", "label": "Metric Scaling", "icon": "📏"},
        {"id": "surface_meshing", "label": "Surface Meshing", "icon": "🔺"},
        {"id": "deliverable_export", "label": "Deliverable Export", "icon": "📦"},
    ]

    # --- Defaults ---
    DEFAULT_FPS = 2.0
    DEFAULT_IMAGE_SIZE = 512
    DEFAULT_IMAGE_SIZE_HIGH = 768         # V3: higher resolution when GPU allows
    DEFAULT_BATCH_SIZE = 1
    DEFAULT_ALIGNMENT_ITERS = 500         # V3: more iterations for tighter convergence
    DEFAULT_SCENE_GRAPH = "swin-5"        # V3: denser connectivity for real drone footage
    MAX_UPLOAD_SIZE_MB = 1024  # 1 GB cap

    # --- Quality filtering defaults ---
    DEFAULT_BLUR_REJECT_PERCENTILE = 20   # Reject bottom 20% sharpness
    DEFAULT_MAX_CLIPPING_RATIO = 0.05     # Reject if >5% pixels clipped

    # --- Keyframe selection defaults ---
    DEFAULT_SSIM_THRESHOLD = 0.88         # V3: lowered for outdoor drone footage with varied viewpoints
    DEFAULT_MIN_TEMPORAL_GAP = 1          # Min frame gap between keyframes
    DEFAULT_MAX_TEMPORAL_GAP = 10         # Force-include if gap exceeds this

    # --- Object detection defaults ---
    DEFAULT_YOLO_CONFIDENCE = 0.35        # Minimum detection confidence
    DEFAULT_MASK_PADDING_PX = 10          # Bbox expansion for shadows/reflections
    DEFAULT_MOVEMENT_THRESHOLD = 5.0      # Compensated px movement → MOVING if above
    DEFAULT_ORB_FEATURES = 1500           # ORB features for camera motion estimation

    # --- COLMAP defaults ---
    DEFAULT_COLMAP_FEATURE_TYPE = "ALIKED"           # Neural features for RTX 40-series
    DEFAULT_COLMAP_MATCHING_TYPE = "ALIKED_BRUTEFORCE"

    # --- GPS / Metric scaling defaults (V3) ---
    DEFAULT_GPS_MIN_DISTANCE = 2.0        # Min GPS distance (m) for scale factor pairs
    DEFAULT_GPS_SCALE_PAIRS_MIN = 3       # Min number of valid pairs for reliable scaling
    DEFAULT_CONFIDENCE_THRESHOLD = 1.5    # DUSt3R confidence: exclude points below this

    # --- ROS bag topic priorities (V3) ---
    # Topics are searched in order; first match wins for each category
    ROSBAG_GPS_TOPICS = [
        "/mavros/global_position/global",
        "/mavros/global_position/raw/fix",
        "/gps/fix",
        "/fix",
        "/navsat/fix",
        "/ublox/fix",
        "/piksi/navsatfix",
        "/dji_sdk/gps_position",
    ]
    ROSBAG_IMU_TOPICS = [
        "/mavros/imu/data",
        "/mavros/imu/data_raw",
        "/imu/data",
        "/imu/data_raw",
        "/dji_sdk/imu",
        "/camera/imu",
    ]
    ROSBAG_IMAGE_TOPICS = [
        "/camera/image_raw/compressed",
        "/camera/image_raw",
        "/image_raw/compressed",
        "/image_raw",
        "/stereo/left/image_raw/compressed",
        "/dji_sdk/image_raw",
        "/usb_cam/image_raw/compressed",
    ]
    ROSBAG_GPS_MSG_TYPES = [
        "sensor_msgs/NavSatFix",
        "sensor_msgs/msg/NavSatFix",
    ]
    ROSBAG_IMU_MSG_TYPES = [
        "sensor_msgs/Imu",
        "sensor_msgs/msg/Imu",
    ]
    ROSBAG_IMAGE_MSG_TYPES = [
        "sensor_msgs/CompressedImage",
        "sensor_msgs/msg/CompressedImage",
        "sensor_msgs/Image",
        "sensor_msgs/msg/Image",
    ]

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

    def job_masks_dir(self, job_id):
        """Return the directory for dynamic object masks."""
        path = os.path.join(self.job_dir(job_id), "masks")
        os.makedirs(path, exist_ok=True)
        return path

    def job_metadata_dir(self, job_id):
        path = os.path.join(self.job_dir(job_id), "metadata")
        os.makedirs(path, exist_ok=True)
        return path

    def job_colmap_dir(self, job_id):
        """Return the directory for COLMAP output (database, sparse model, poses)."""
        path = os.path.join(self.job_dir(job_id), "colmap")
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

    def job_bag_path(self, job_id):
        """Return the expected path for a ROS bag file in a job directory."""
        return os.path.join(self.job_dir(job_id), "input.bag")

    @property
    def model_exists(self):
        return os.path.isfile(self.model_path)
