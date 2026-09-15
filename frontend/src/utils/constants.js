/* API base URL */
export const API_BASE = 'http://localhost:8000';
export const WS_BASE = 'ws://localhost:8000';

/* Pipeline stage definitions (mirrors backend config) */
export const PIPELINE_STAGES = [
  { id: 'frame_extraction', label: 'Frame Extraction', icon: '🎞️', description: 'Extracting key frames from drone video' },
  { id: 'metadata_extraction', label: 'Metadata Extraction', icon: '📡', description: 'Parsing GPS coordinates & flight telemetry' },
  { id: 'point_cloud_generation', label: 'Point Cloud Generation', icon: '☁️', description: 'Running DUSt3R AI-driven 3D reconstruction' },
  { id: 'surface_meshing', label: 'Surface Meshing', icon: '🔺', description: 'Poisson surface reconstruction with Taubin smoothing' },
  { id: 'deliverable_export', label: 'Deliverable Export', icon: '📦', description: 'Generating GLB, LAS, and other output formats' },
];

/* Supported video formats */
export const SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv'];

/* Max upload size in bytes (1 GB) */
export const MAX_UPLOAD_SIZE = 1024 * 1024 * 1024;

/* FPS extraction range */
export const FPS_MIN = 0.5;
export const FPS_MAX = 10;
export const FPS_DEFAULT = 2;
