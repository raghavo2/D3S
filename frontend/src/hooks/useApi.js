import { API_BASE } from '../utils/constants';

/**
 * Lightweight API client for the D3S backend.
 */
export function useApi() {

  async function request(path, options = {}) {
    const res = await fetch(`${API_BASE}${path}`, options);
    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(error.detail || `API error: ${res.status}`);
    }
    return res.json();
  }

  /** Health check */
  const getHealth = () => request('/api/health');

  /** Upload a video file and create a job, with optional flight data attachments */
  const uploadVideo = async (file, fps, onProgress, extraFiles = {}) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('fps', fps.toString());

    // Append optional flight data files
    if (extraFiles.gpsFile) formData.append('gps_file', extraFiles.gpsFile);
    if (extraFiles.flightDataFile) formData.append('flight_data_file', extraFiles.flightDataFile);
    if (extraFiles.telemetryFile) formData.append('telemetry_file', extraFiles.telemetryFile);

    const xhr = new XMLHttpRequest();
    return new Promise((resolve, reject) => {
      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable && onProgress) {
          onProgress(Math.round((e.loaded / e.total) * 100));
        }
      });
      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(JSON.parse(xhr.responseText));
        } else {
          try {
            const err = JSON.parse(xhr.responseText);
            reject(new Error(err.detail || `Upload failed: ${xhr.status}`));
          } catch {
            reject(new Error(`Upload failed: ${xhr.status}`));
          }
        }
      });
      xhr.addEventListener('error', () => reject(new Error('Upload failed — network error')));
      xhr.addEventListener('abort', () => reject(new Error('Upload cancelled')));
      xhr.open('POST', `${API_BASE}/api/upload`);
      xhr.send(formData);
    });
  };

  /** Start a pipeline job */
  const startJob = (jobId) => request(`/api/jobs/${jobId}/start`, { method: 'POST' });

  /** Get job status */
  const getJobStatus = (jobId) => request(`/api/jobs/${jobId}/status`);

  /** List all jobs */
  const listJobs = () => request('/api/jobs');

  /** Get job metadata (GPS/telemetry) */
  const getJobMetadata = (jobId) => request(`/api/jobs/${jobId}/metadata`);

  /** Get job frames list */
  const getJobFrames = (jobId) => request(`/api/jobs/${jobId}/frames`);

  /** Get model URL for a job (for Three.js loader) */
  const getModelUrl = (jobId) => `${API_BASE}/api/jobs/${jobId}/model`;

  /** Get point cloud URL for a job */
  const getPointCloudUrl = (jobId) => `${API_BASE}/api/jobs/${jobId}/pointcloud`;

  /** Get demo model URL */
  const getDemoModelUrl = () => `${API_BASE}/api/demo/model`;

  /** Get demo point cloud URL */
  const getDemoPointCloudUrl = () => `${API_BASE}/api/demo/pointcloud`;

  /** List all available demos */
  const listDemos = () => request('/api/demos');

  /** Save a completed job as a demo */
  const saveJobAsDemo = (jobId, name) => {
    const formData = new FormData();
    if (name) formData.append('name', name);
    return request(`/api/jobs/${jobId}/save-demo`, { method: 'POST', body: formData });
  };

  /** Delete a demo */
  const deleteDemo = (demoId) => request(`/api/demos/${demoId}`, { method: 'DELETE' });

  /** Get demo model URL by demo ID */
  const getDemoModelUrlById = (demoId) => `${API_BASE}/api/demos/${demoId}/model`;

  /** Get demo point cloud URL by demo ID */
  const getDemoPointCloudUrlById = (demoId) => `${API_BASE}/api/demos/${demoId}/pointcloud`;

  /** Get GPS track as GeoJSON */
  const getJobGps = (jobId) => request(`/api/jobs/${jobId}/gps`);

  /** Delete a job */
  const deleteJob = (jobId) => request(`/api/jobs/${jobId}`, { method: 'DELETE' });

  /** Get job output files */
  const getJobOutputs = (jobId) => request(`/api/jobs/${jobId}/outputs`);

  return {
    request,
    getHealth,
    uploadVideo,
    startJob,
    getJobStatus,
    listJobs,
    getJobMetadata,
    getJobFrames,
    getJobGps,
    deleteJob,
    getJobOutputs,
    getModelUrl,
    getPointCloudUrl,
    getDemoModelUrl,
    getDemoPointCloudUrl,
    listDemos,
    saveJobAsDemo,
    deleteDemo,
    getDemoModelUrlById,
    getDemoPointCloudUrlById,
  };
}
