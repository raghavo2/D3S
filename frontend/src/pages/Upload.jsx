import { useState, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApi } from '../hooks/useApi';
import { SUPPORTED_FORMATS, MAX_UPLOAD_SIZE, FPS_MIN, FPS_MAX, FPS_DEFAULT } from '../utils/constants';
import './Upload.css';

const FORMAT_TABS = [
  {
    id: 'gps-json',
    label: 'GPS JSON',
    ext: '.json',
    desc: 'Array of coordinate objects.',
    code: `[
  { "filename": "frame_001.jpg", "latitude": 28.6139, "longitude": 77.2090, "altitude": 120.5 },
  { "filename": "frame_002.jpg", "latitude": 28.6141, "longitude": 77.2092, "altitude": 121.0 }
]`,
  },
  {
    id: 'geojson',
    label: 'GeoJSON',
    ext: '.geojson',
    desc: 'FeatureCollection with Point geometries.',
    code: `{
  "type": "FeatureCollection",
  "features": [
    { "type": "Feature", "geometry": { "type": "Point", "coordinates": [77.209, 28.614, 120.5] } }
  ]
}`,
  },
  {
    id: 'csv',
    label: 'Flight CSV',
    ext: '.csv',
    desc: 'Columns: latitude/lat, longitude/lon/lng, altitude/alt/elevation, filename (opt)',
    code: `filename,latitude,longitude,altitude
frame_001.jpg,28.6139,77.2090,120.5
frame_002.jpg,28.6141,77.2092,121.0`,
  },
  {
    id: 'srt',
    label: 'SRT / Telemetry',
    ext: '.srt .json .txt',
    desc: 'DJI-style SRT subtitles with embedded GPS.',
    code: `1
00:00:00,000 --> 00:00:01,000
[latitude: 28.6139] [longitude: 77.2090]
[altitude: 120.5] [iso: 200] [shutter: 1/500]`,
  },
];

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

  // Optional flight data files
  const [gpsFile, setGpsFile] = useState(null);
  const [flightDataFile, setFlightDataFile] = useState(null);
  const [telemetryFile, setTelemetryFile] = useState(null);
  const gpsInputRef = useRef(null);
  const flightDataInputRef = useRef(null);
  const telemetryInputRef = useRef(null);

  // Format guide
  const [showFormatGuide, setShowFormatGuide] = useState(false);
  const [activeFormatTab, setActiveFormatTab] = useState('gps-json');

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
      // 1. Upload video + optional flight data
      const extraFiles = { gpsFile, flightDataFile, telemetryFile };
      const result = await uploadVideo(file, fps, (pct) => setProgress(pct), extraFiles);

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

  const activeFormat = FORMAT_TABS.find(t => t.id === activeFormatTab);
  const attachCount = [gpsFile, flightDataFile, telemetryFile].filter(Boolean).length;

  return (
    <div className="upload-page page">
      <div className="upload-container animate-fade-in">
        <div className="upload-header">
          <h1 className="gradient-text">Upload Video</h1>
          <p>Drop your drone flight video to begin 3D reconstruction</p>
        </div>

        <div className="upload-columns">
          {/* ─── Left Column: Drop Zone + Preview ─── */}
          <div className="upload-col-left">
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
                <p className="drop-zone-ready">✓ Video ready — {file.name}</p>
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
          </div>

          {/* ─── Right Column: Settings + Flight Data + Actions ─── */}
          <div className="upload-col-right">
            {/* Configuration */}
            <div className="config-section glass">
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

            {/* ─── Flight Data Attachments ─── */}
            <div className="flight-data-section glass">
              <h3 className="config-title">
                Flight Data
                <span className="optional-badge">Optional</span>
                {attachCount > 0 && <span className="attach-count">{attachCount} attached</span>}
              </h3>

              {/* GPS File */}
              <div className="attachment-row">
                <div className="attachment-info">
                  <div className="attachment-details">
                    <span className="attachment-label">GPS Data</span>
                    <span className="attachment-hint">.json / .geojson</span>
                  </div>
                </div>
                <input ref={gpsInputRef} type="file" accept=".json,.geojson" style={{ display: 'none' }}
                  onChange={(e) => e.target.files[0] && setGpsFile(e.target.files[0])} />
                {gpsFile ? (
                  <div className="attachment-selected">
                    <span className="attachment-filename">{gpsFile.name}</span>
                    <button className="attachment-remove" onClick={() => { setGpsFile(null); gpsInputRef.current.value = ''; }}>✕</button>
                  </div>
                ) : (
                  <button className="btn btn-ghost attachment-btn" onClick={() => gpsInputRef.current?.click()}>+ Attach</button>
                )}
              </div>

              {/* Flight Data CSV */}
              <div className="attachment-row">
                <div className="attachment-info">
                  <div className="attachment-details">
                    <span className="attachment-label">Flight Log</span>
                    <span className="attachment-hint">.csv</span>
                  </div>
                </div>
                <input ref={flightDataInputRef} type="file" accept=".csv,.tsv" style={{ display: 'none' }}
                  onChange={(e) => e.target.files[0] && setFlightDataFile(e.target.files[0])} />
                {flightDataFile ? (
                  <div className="attachment-selected">
                    <span className="attachment-filename">{flightDataFile.name}</span>
                    <button className="attachment-remove" onClick={() => { setFlightDataFile(null); flightDataInputRef.current.value = ''; }}>✕</button>
                  </div>
                ) : (
                  <button className="btn btn-ghost attachment-btn" onClick={() => flightDataInputRef.current?.click()}>+ Attach</button>
                )}
              </div>

              {/* Telemetry / SRT */}
              <div className="attachment-row">
                <div className="attachment-info">
                  <div className="attachment-details">
                    <span className="attachment-label">Telemetry / SRT</span>
                    <span className="attachment-hint">.srt / .json / .txt</span>
                  </div>
                </div>
                <input ref={telemetryInputRef} type="file" accept=".srt,.json,.txt" style={{ display: 'none' }}
                  onChange={(e) => e.target.files[0] && setTelemetryFile(e.target.files[0])} />
                {telemetryFile ? (
                  <div className="attachment-selected">
                    <span className="attachment-filename">{telemetryFile.name}</span>
                    <button className="attachment-remove" onClick={() => { setTelemetryFile(null); telemetryInputRef.current.value = ''; }}>✕</button>
                  </div>
                ) : (
                  <button className="btn btn-ghost attachment-btn" onClick={() => telemetryInputRef.current?.click()}>+ Attach</button>
                )}
              </div>
            </div>

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

        {/* ─── Format Reference — full width below columns ─── */}
        <div className="format-reference">
          <button
            className={`format-guide-toggle ${showFormatGuide ? 'open' : ''}`}
            onClick={() => setShowFormatGuide(!showFormatGuide)}
          >
            <span>Supported Data Formats</span>
            <span className="format-guide-arrow">{showFormatGuide ? '▲' : '▼'}</span>
          </button>

          {showFormatGuide && (
            <div className="format-guide glass animate-fade-in">
              <div className="format-tabs">
                {FORMAT_TABS.map(tab => (
                  <button
                    key={tab.id}
                    className={`format-tab ${activeFormatTab === tab.id ? 'active' : ''}`}
                    onClick={() => setActiveFormatTab(tab.id)}
                  >
                    <span>{tab.label}</span>
                  </button>
                ))}
              </div>
              {activeFormat && (
                <div className="format-content">
                  <div className="format-content-header">
                    <span className="format-block-ext">{activeFormat.ext}</span>
                    <span className="format-block-desc">{activeFormat.desc}</span>
                  </div>
                  <pre className="format-code">{activeFormat.code}</pre>
                </div>
              )}
            </div>
          )}
        </div>

      </div>
    </div>
  );
}
