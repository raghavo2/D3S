import { useState, useEffect, useMemo } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import ModelViewer from '../components/ModelViewer';
import PointCloudViewer from '../components/PointCloudViewer';
import ExportPanel from '../components/ExportPanel';
import FlightMap from '../components/FlightMap';
import { useApi } from '../hooks/useApi';
import './Viewer.css';

export default function Viewer() {
  const [searchParams, setSearchParams] = useSearchParams();
  const jobId = searchParams.get('job');
  const demoId = searchParams.get('demo');

  const {
    getModelUrl, getPointCloudUrl,
    getDemoModelUrlById,
    getDemoPointCloudUrlById,
    listDemos,
    getJobGps,
  } = useApi();

  const [viewMode, setViewMode] = useState('mesh');
  const [showGrid, setShowGrid] = useState(true);
  const [autoRotate, setAutoRotate] = useState(false);
  const [pointSize, setPointSize] = useState(0.02);
  const [darkBg, setDarkBg] = useState(true);

  // Demo gallery state
  const [demos, setDemos] = useState([]);
  const [galleryOpen, setGalleryOpen] = useState(false);
  const [activeDemo, setActiveDemo] = useState(demoId || (jobId ? null : 'builtin'));
  const [gpsData, setGpsData] = useState(null);
  const [showMap, setShowMap] = useState(false);

  // Fetch demos list
  useEffect(() => {
    listDemos()
      .then(data => setDemos(data.demos || []))
      .catch(() => {});
  }, []);

  // Fetch GPS data for active job
  useEffect(() => {
    if (jobId && !activeDemo) {
      getJobGps(jobId)
        .then(data => {
          if (data.points && data.points.length > 0) setGpsData(data);
        })
        .catch(() => {});
    } else {
      setGpsData(null);
    }
  }, [jobId, activeDemo]);

  // Resolve URLs based on source (job or demo)
  const modelUrl = useMemo(() => {
    if (jobId && !activeDemo) return getModelUrl(jobId);
    if (activeDemo) return getDemoModelUrlById(activeDemo);
    return getDemoModelUrlById('builtin');
  }, [jobId, activeDemo]);

  const pointCloudUrl = useMemo(() => {
    if (jobId && !activeDemo) return getPointCloudUrl(jobId);
    if (activeDemo) return getDemoPointCloudUrlById(activeDemo);
    return getDemoPointCloudUrlById('builtin');
  }, [jobId, activeDemo]);

  // Active item name
  const activeName = useMemo(() => {
    if (jobId && !activeDemo) return `Job ${jobId.slice(0, 8)}…`;
    const found = demos.find(d => d.id === activeDemo);
    return found?.name || 'Demo Model';
  }, [jobId, activeDemo, demos]);

  const selectDemo = (id) => {
    setActiveDemo(id);
    setSearchParams({ demo: id });
    setGalleryOpen(false);
  };

  return (
    <div className="viewer-page">
      {/* Toolbar */}
      <div className="viewer-toolbar">
        <div className="viewer-toolbar-left">
          <h2 className="viewer-title gradient-text">{activeName}</h2>

          {/* View Mode Toggle */}
          <div className="view-toggle">
            <button
              className={`view-toggle-btn ${viewMode === 'mesh' ? 'active' : ''}`}
              onClick={() => setViewMode('mesh')}
            >
              🔺 Mesh
            </button>
            <button
              className={`view-toggle-btn ${viewMode === 'pointcloud' ? 'active' : ''}`}
              onClick={() => setViewMode('pointcloud')}
            >
              ☁️ Point Cloud
            </button>
          </div>
        </div>

        <div className="viewer-toolbar-right">
          {jobId && !activeDemo && (
            <ExportPanel jobId={jobId} />
          )}
          <button
            className={`btn btn-ghost ${galleryOpen ? 'active' : ''}`}
            style={{ fontSize: '0.8rem' }}
            onClick={() => setGalleryOpen(!galleryOpen)}
          >
            📂 Gallery {demos.length > 0 && `(${demos.length})`}
          </button>
          <Link to="/upload" className="btn btn-ghost" style={{ fontSize: '0.8rem' }}>
            + New Upload
          </Link>
        </div>
      </div>

      {/* Main Content Area */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden', position: 'relative' }}>

        {/* Demo Gallery Sidebar */}
        <div className={`demo-gallery ${galleryOpen ? 'open' : ''}`}>
          <div className="demo-gallery-header">
            <h3 className="demo-gallery-title">Demo Gallery</h3>
            <button
              className="btn btn-ghost btn-icon"
              onClick={() => setGalleryOpen(false)}
              style={{ fontSize: '1rem' }}
            >
              ✕
            </button>
          </div>

          <div className="demo-gallery-list">
            {demos.length === 0 && (
              <div className="demo-gallery-empty">
                <p>No demos available yet.</p>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  Complete a reconstruction and save it as a demo.
                </p>
              </div>
            )}
            {demos.map(demo => (
              <button
                key={demo.id}
                className={`demo-card ${activeDemo === demo.id ? 'active' : ''}`}
                onClick={() => selectDemo(demo.id)}
              >
                <div className="demo-card-icon">
                  {demo.source === 'builtin' ? '🏔️' : '🎯'}
                </div>
                <div className="demo-card-info">
                  <span className="demo-card-name">{demo.name}</span>
                  <span className="demo-card-meta">
                    {demo.source === 'builtin' ? 'Built-in' : new Date(demo.created_at).toLocaleDateString()}
                    {demo.has_model && ' · GLB'}
                    {demo.has_pointcloud && ' · PLY'}
                  </span>
                </div>
                {activeDemo === demo.id && (
                  <span style={{ color: 'var(--accent-cyan)', fontSize: '0.8rem' }}>●</span>
                )}
              </button>
            ))}
          </div>
        </div>

        {/* Canvas */}
        <div className="viewer-canvas" style={{ flex: 1 }}>
          {viewMode === 'mesh' ? (
            <ModelViewer
              url={modelUrl}
              showGrid={showGrid}
              autoRotate={autoRotate}
              bgColor={darkBg ? '#06060a' : '#f0f0f0'}
            />
          ) : (
            <PointCloudViewer
              url={pointCloudUrl}
              pointSize={pointSize}
              bgColor={darkBg ? '#06060a' : '#f0f0f0'}
            />
          )}

          {/* Loading Overlay (visible by default, hidden by CSS once canvas renders) */}
          <div className="viewer-loading-overlay" id="viewer-loading">
            <div style={{
              display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px',
              background: 'rgba(6,6,10,0.85)', padding: '32px 48px', borderRadius: '12px',
              border: '1px solid rgba(255,255,255,0.06)',
            }}>
              <div style={{
                width: 40, height: 40,
                border: '3px solid rgba(148,163,184,0.2)',
                borderTopColor: 'rgba(255,255,255,0.6)',
                borderRadius: '50%',
                animation: 'spin 1s linear infinite',
              }} />
              <span style={{ color: '#e2e8f0', fontSize: '0.9rem', fontWeight: 600 }}>Loading 3D model…</span>
              <span style={{ color: '#64748b', fontSize: '0.72rem' }}>Large models may take a moment</span>
            </div>
          </div>

          {/* Floating Controls */}
          <div className="viewer-controls">
            <button
              className={`viewer-control-btn ${showGrid ? 'active' : ''}`}
              onClick={() => setShowGrid(!showGrid)}
              title="Toggle Grid"
            >
              #
            </button>
            <button
              className={`viewer-control-btn ${autoRotate ? 'active' : ''}`}
              onClick={() => setAutoRotate(!autoRotate)}
              title="Auto Rotate"
            >
              ↻
            </button>
            <button
              className={`viewer-control-btn ${!darkBg ? 'active' : ''}`}
              onClick={() => setDarkBg(!darkBg)}
              title="Toggle Background"
            >
              {darkBg ? '☀️' : '🌙'}
            </button>
            {gpsData && (
              <button
                className={`viewer-control-btn ${showMap ? 'active' : ''}`}
                onClick={() => setShowMap(!showMap)}
                title="Toggle Flight Map"
              >
                🗺️
              </button>
            )}
          </div>

          {/* Point Size Control */}
          {viewMode === 'pointcloud' && (
            <div className="point-size-control">
              <span className="point-size-label">Size</span>
              <input
                type="range"
                className="point-size-slider"
                min={0.005}
                max={0.1}
                step={0.005}
                value={pointSize}
                onChange={(e) => setPointSize(parseFloat(e.target.value))}
              />
            </div>
          )}

          {/* Info Panel */}
          <div className="viewer-info">
            <div className="viewer-info-row">
              <span className="viewer-info-label">Mode</span>
              <span className="viewer-info-value">
                {viewMode === 'mesh' ? 'Mesh (GLB)' : 'Point Cloud (PLY)'}
              </span>
            </div>
            <div className="viewer-info-row">
              <span className="viewer-info-label">Source</span>
              <span className="viewer-info-value">{activeName}</span>
            </div>
            <div className="viewer-info-row">
              <span className="viewer-info-label">Controls</span>
              <span className="viewer-info-value" style={{ fontSize: '0.7rem' }}>
                Orbit · Pan · Zoom
              </span>
            </div>
          </div>

          {/* Mini-Map Overlay */}
          {showMap && gpsData && (
            <div className="viewer-minimap">
              <FlightMap gpsData={gpsData} compact />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
