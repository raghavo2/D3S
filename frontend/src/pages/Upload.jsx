import { useState, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApi } from '../hooks/useApi';
import { SUPPORTED_FORMATS, MAX_UPLOAD_SIZE, FPS_MIN, FPS_MAX, FPS_DEFAULT } from '../utils/constants';
import './Upload.css';

export default function Upload() {
  const navigate = useNavigate();
  const { uploadVideo, startJob } = useApi();
  const fileInputRef = useRef(null);

  const [file, setFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [fps, setFps] = useState(FPS_DEFAULT);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);

  const validateFile = (f) => {
    const ext = '.' + f.name.split('.').pop().toLowerCase();
    if (!SUPPORTED_FORMATS.includes(ext)) {
      setError(`Unsupported format: ${ext}. Use ${SUPPORTED_FORMATS.join(', ')}`);
      return false;
    }
    if (f.size > MAX_UPLOAD_SIZE) {
      setError(`File too large (${(f.size / (1024 * 1024)).toFixed(0)}MB). Max: ${MAX_UPLOAD_SIZE / (1024 * 1024)}MB`);
      return false;
    }
    return true;
  };

  const handleFileSelect = (f) => {
    setError(null);
    if (!validateFile(f)) return;
    setFile(f);
    setVideoUrl(URL.createObjectURL(f));
  };

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFileSelect(f);
  }, []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setDragOver(false);
  }, []);

  const removeFile = () => {
    setFile(null);
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setVideoUrl(null);
    setError(null);
  };

  const handleSubmit = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);
    setProgress(0);

    try {
      // 1. Upload video
      const result = await uploadVideo(file, fps, (pct) => setProgress(pct));

      // 2. Auto-start the pipeline
      await startJob(result.job_id);

      // 3. Navigate to pipeline monitor
      navigate(`/pipeline/${result.job_id}`);
    } catch (err) {
      setError(err.message);
      setUploading(false);
    }
  };

  const formatSize = (bytes) => {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="upload-page page">
      <div className="upload-container animate-fade-in">
        <div className="upload-header">
          <h1 className="gradient-text">Upload Video</h1>
          <p>Drop your drone flight video to begin 3D reconstruction</p>
        </div>

        {/* Drop Zone */}
        <div
          className={`drop-zone ${dragOver ? 'drag-over' : ''} ${file ? 'has-file' : ''}`}
          onClick={() => !file && fileInputRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".mp4,.avi,.mov,.mkv"
            onChange={(e) => e.target.files[0] && handleFileSelect(e.target.files[0])}
          />
          {!file ? (
            <>
              <span className="drop-zone-icon">↑</span>
              <p className="drop-zone-text">Drop video here or click to browse</p>
              <p className="drop-zone-hint">
                Supports {SUPPORTED_FORMATS.join(', ')} • Max {MAX_UPLOAD_SIZE / (1024 * 1024 * 1024)}GB
              </p>
            </>
          ) : (
            <>
              <span className="drop-zone-icon">✓</span>
              <p className="drop-zone-text">Video ready for processing</p>
            </>
          )}
        </div>

        {/* Video Preview */}
        {file && videoUrl && (
          <div className="video-preview animate-fade-in">
            <video src={videoUrl} controls muted />
            <div className="video-info">
              <span className="video-name">{file.name}</span>
              <span className="video-size">{formatSize(file.size)}</span>
              <button className="video-remove" onClick={removeFile}>Remove</button>
            </div>
          </div>
        )}

        {/* Configuration */}
        {file && (
          <div className="config-section glass animate-fade-in">
            <h3 className="config-title">Extraction Settings</h3>
            <div className="config-row">
              <span className="config-label">Frames per second</span>
              <input
                type="range"
                className="config-slider"
                min={FPS_MIN}
                max={FPS_MAX}
                step={0.5}
                value={fps}
                onChange={(e) => setFps(parseFloat(e.target.value))}
              />
              <span className="config-value">{fps} FPS</span>
            </div>
          </div>
        )}

        {/* Upload Progress */}
        {uploading && (
          <div className="upload-progress animate-fade-in">
            <div className="progress-bar-container">
              <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
            </div>
            <p className="progress-text">
              {progress < 100 ? `Uploading… ${progress}%` : 'Starting pipeline…'}
            </p>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="upload-error animate-fade-in">
            {error}
          </div>
        )}

        {/* Submit */}
        {file && !uploading && (
          <div className="upload-submit animate-fade-in">
            <button className="btn btn-primary" onClick={handleSubmit}>
              Upload & Start Reconstruction →
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
