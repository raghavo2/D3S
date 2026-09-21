/* API base URL */
export const API_BASE = 'http://localhost:8000';
export const WS_BASE = 'ws://localhost:8000';

/* Pipeline stage definitions (mirrors backend config — V3) */
export const PIPELINE_STAGES = [
  { id: 'rosbag_extraction', label: 'ROS Bag Extraction', icon: '📡', description: 'Extracting GPS, IMU, and frames from ROS bag' },
  { id: 'frame_extraction', label: 'Frame Extraction', icon: '🎞️', description: 'Extracting key frames from drone video' },
  { id: 'quality_filtering', label: 'Quality Filtering', icon: '🔍', description: 'Filtering blurry and overexposed frames' },
  { id: 'keyframe_selection', label: 'Keyframe Selection', icon: '🎯', description: 'Selecting optimal keyframes via SSIM' },
  { id: 'object_detection', label: 'Object Detection', icon: '🚨', description: 'Detecting and tracking dynamic objects' },
  { id: 'colmap_pose_estimation', label: 'COLMAP Poses', icon: '📰', description: 'Estimating camera poses via COLMAP SfM' },
  { id: 'metadata_extraction', label: 'Metadata Extraction', icon: '📡', description: 'Parsing GPS coordinates & flight telemetry' },
  { id: 'point_cloud_generation', label: 'Point Cloud Generation', icon: '☁️', description: 'Running DUSt3R AI-driven 3D reconstruction' },
  { id: 'metric_scaling', label: 'Metric Scaling', icon: '📏', description: 'Computing GPS-based real-world scale factor' },
  { id: 'surface_meshing', label: 'Surface Meshing', icon: '🔺', description: 'Poisson surface reconstruction with Taubin smoothing' },
  { id: 'deliverable_export', label: 'Deliverable Export', icon: '📦', description: 'Generating GLB, LAS, and other output formats' },
];

/* Supported input formats (V3: video + ROS bag) */
export const SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv'];
export const SUPPORTED_BAG_FORMATS = ['.bag'];
export const SUPPORTED_FORMATS = [...SUPPORTED_VIDEO_FORMATS, ...SUPPORTED_BAG_FORMATS];

/* Max upload size in bytes (1 GB) */
export const MAX_UPLOAD_SIZE = 1024 * 1024 * 1024;

/* FPS extraction range */
export const FPS_MIN = 0.5;
export const FPS_MAX = 10;
export const FPS_DEFAULT = 2;
