import { useState, useEffect, useRef } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { useWebSocket } from '../hooks/useWebSocket';
import { useApi } from '../hooks/useApi';
import { PIPELINE_STAGES } from '../utils/constants';
import FlightMap from '../components/FlightMap';
import FrameGallery from '../components/FrameGallery';
import './Pipeline.css';

export default function Pipeline() {
  const { jobId } = useParams();
  const navigate = useNavigate();
  const { getJobStatus, getJobMetadata, getJobGps, getJobFrames, saveJobAsDemo } = useApi();
  const { messages, isConnected } = useWebSocket(jobId);

  const [stages, setStages] = useState({});
  const [currentStage, setCurrentStage] = useState(null);
  const [pipelineStatus, setPipelineStatus] = useState('running');
  const [metadata, setMetadata] = useState(null);
  const [logLines, setLogLines] = useState([]);
  const [demoName, setDemoName] = useState('');
  const [demoSaved, setDemoSaved] = useState(false);
  const [demoSaving, setDemoSaving] = useState(false);
  const [gpsData, setGpsData] = useState(null);
  const [frames, setFrames] = useState({ frames: [], base_url: '' });
  const [logFilter, setLogFilter] = useState('all'); // 'all' | 'errors' | 'stages'
  const [startTime] = useState(Date.now());
  const [elapsed, setElapsed] = useState(0);
  const logEndRef = useRef(null);

  // Fetch initial job state
  useEffect(() => {
    if (!jobId) return;
    getJobStatus(jobId)
      .then((data) => {
        setStages(data.stages || {});
        setCurrentStage(data.stage);
        setPipelineStatus(data.status);
        if (data.metadata?.gps) {
          setMetadata(data.metadata);
        }
      })
      .catch(() => {});

    // Fetch GPS data
    getJobGps(jobId)
      .then(data => {
        if (data.points && data.points.length > 0) setGpsData(data);
      })
      .catch(() => {});
  }, [jobId]);

  // Process WebSocket messages
  useEffect(() => {
    if (messages.length === 0) return;
    const msg = messages[messages.length - 1];

    if (msg.type === 'stage') {
      setStages(prev => ({
        ...prev,
        [msg.stage]: { status: msg.status, result: msg.result, error: msg.error },
      }));
      if (msg.status === 'running') {
        setCurrentStage(msg.stage);
      }
    }

    if (msg.type === 'pipeline') {
      setPipelineStatus(msg.status === 'complete' ? 'complete' : msg.status);
    }

    if (msg.type === 'log') {
      setLogLines(prev => [...prev.slice(-300), {
        text: msg.message,
        stage: msg.stage,
        type: msg.type,
      }]);
    }
  }, [messages]);

  // Auto-scroll log
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logLines]);

  // Elapsed timer
  useEffect(() => {
    if (pipelineStatus === 'running') {
      const timer = setInterval(() => setElapsed(Math.floor((Date.now() - startTime) / 1000)), 1000);
      return () => clearInterval(timer);
    }
  }, [pipelineStatus, startTime]);

  // Auto-redirect on completion
  useEffect(() => {
    if (pipelineStatus === 'complete') {
      const timer = setTimeout(() => navigate(`/viewer?job=${jobId}`), 5000);
      return () => clearTimeout(timer);
    }
  }, [pipelineStatus, jobId, navigate]);

  // Fetch metadata when available
  useEffect(() => {
    if (stages.metadata_extraction?.status === 'complete') {
      getJobMetadata(jobId).then(setMetadata).catch(() => {});
    }
  }, [stages.metadata_extraction?.status]);

  // Fetch frames when extraction is complete
  useEffect(() => {
    if (stages.frame_extraction?.status === 'complete') {
      getJobFrames(jobId).then(setFrames).catch(() => {});
    }
  }, [stages.frame_extraction?.status]);

  const formatElapsed = (s) => {
    const m = Math.floor(s / 60);
    return m > 0 ? `${m}m ${s % 60}s` : `${s}s`;
  };

  const filteredLogs = logLines.filter(line => {
    if (logFilter === 'all') return true;
    if (logFilter === 'errors') return line.text?.toLowerCase().includes('error') || line.text?.toLowerCase().includes('fail');
    if (logFilter === 'stages') return line.text?.toLowerCase().includes('stage') || line.text?.toLowerCase().includes('starting') || line.text?.toLowerCase().includes('complete');
    return true;
  });

  const getStageStatus = (stageId) => {
    if (stages[stageId]) return stages[stageId].status;
    // Determine pending based on stage order
    const currentIndex = PIPELINE_STAGES.findIndex(s => s.id === currentStage);
    const stageIndex = PIPELINE_STAGES.findIndex(s => s.id === stageId);
    return stageIndex > currentIndex ? 'pending' : 'pending';
  };

  const getLogLineClass = (line) => {
    if (line.text?.includes('Error') || line.text?.includes('error')) return 'log-line error-event';
    if (line.text?.includes('complete') || line.text?.includes('Success') || line.text?.includes('Saved')) return 'log-line success-event';
    if (line.text?.includes('Running') || line.text?.includes('Loading') || line.text?.includes('Starting')) return 'log-line stage-event';
    return 'log-line';
  };

  return (
    <div className="pipeline-page page">
      <div className="pipeline-layout animate-fade-in">

        {/* Header */}
        <div className="pipeline-header">
          <h1 className="gradient-text">Pipeline Monitor</h1>
          <span className={`badge ${
            pipelineStatus === 'complete' ? 'badge-success' :
            pipelineStatus === 'failed' || pipelineStatus === 'interrupted' ? 'badge-error' :
            pipelineStatus === 'uploaded' ? 'badge-warning' :
            'badge-info'
          }`}>
            {pipelineStatus === 'complete' ? '✓ Complete' :
             pipelineStatus === 'failed' ? '✗ Failed' :
             pipelineStatus === 'interrupted' ? '⚡ Interrupted' :
             pipelineStatus === 'uploaded' ? '📤 Uploaded' :
             '⟳ Running'}
          </span>
        </div>

        {/* Stepper */}
        <div className="stepper">
          <ul className="stepper-list">
            {PIPELINE_STAGES.map((stage) => {
              const status = getStageStatus(stage.id);
              return (
                <li key={stage.id} className={`stepper-item ${status}`}>
                  <div className="stepper-icon">
                    {status === 'complete' ? '✓' :
                     status === 'running' ? <span className="spinner" style={{ width: 16, height: 16 }} /> :
                     status === 'error' ? '✗' :
                     status === 'warning' ? '⚠' :
                     stage.icon}
                  </div>
                  <div className="stepper-info">
                    <div className="stepper-label">{stage.label}</div>
                    <div className="stepper-desc">{stage.description}</div>
                    {status !== 'pending' && (
                      <div className="stepper-status-text">{status}</div>
                    )}
                  </div>
                </li>
              );
            })}
          </ul>
        </div>

        {/* Right Panel */}
        <div className="pipeline-panel">

          {/* Completion Banner */}
          {pipelineStatus === 'complete' && (
            <div className="complete-banner animate-slide-up">
              <h3>🎉 Reconstruction Complete!</h3>
              <p>Your 3D model is ready for interactive viewing.</p>
              <div style={{ display: 'flex', gap: 'var(--space-md)', justifyContent: 'center', flexWrap: 'wrap' }}>
                <Link to={`/viewer?job=${jobId}`} className="btn btn-primary">
                  Open 3D Viewer →
                </Link>
                {!demoSaved ? (
                  <div style={{ display: 'flex', gap: 'var(--space-sm)', alignItems: 'center' }}>
                    <input
                      type="text"
                      placeholder="Demo name (optional)"
                      value={demoName}
                      onChange={(e) => setDemoName(e.target.value)}
                      style={{
                        padding: '10px 14px',
                        borderRadius: 'var(--radius-md)',
                        border: '1px solid var(--glass-border)',
                        background: 'var(--glass-bg)',
                        color: 'var(--text-primary)',
                        fontSize: '0.85rem',
                        fontFamily: 'var(--font-sans)',
                        width: '180px',
                      }}
                    />
                    <button
                      className="btn btn-secondary"
                      disabled={demoSaving}
                      onClick={async () => {
                        setDemoSaving(true);
                        try {
                          await saveJobAsDemo(jobId, demoName || null);
                          setDemoSaved(true);
                        } catch { setDemoSaving(false); }
                      }}
                    >
                      {demoSaving ? 'Saving…' : '⭐ Save to Demos'}
                    </button>
                  </div>
                ) : (
                  <span className="badge badge-success" style={{ padding: '10px 16px', fontSize: '0.82rem' }}>
                    ✓ Saved to Demo Gallery
                  </span>
                )}
              </div>
            </div>
          )}

          {/* Metadata Panel */}
          {metadata?.gps && metadata.gps.length > 0 && (
            <div className="metadata-panel glass">
              <h3 className="metadata-title">📡 Extracted Metadata</h3>
              <div className="metadata-grid">
                <div className="metadata-item">
                  <span className="metadata-label">GPS Points</span>
                  <span className="metadata-value">{metadata.gps.length}</span>
                </div>
                <div className="metadata-item">
                  <span className="metadata-label">Latitude Range</span>
                  <span className="metadata-value">
                    {Math.min(...metadata.gps.map(g => g.latitude)).toFixed(4)}° — {Math.max(...metadata.gps.map(g => g.latitude)).toFixed(4)}°
                  </span>
                </div>
                <div className="metadata-item">
                  <span className="metadata-label">Longitude Range</span>
                  <span className="metadata-value">
                    {Math.min(...metadata.gps.map(g => g.longitude)).toFixed(4)}° — {Math.max(...metadata.gps.map(g => g.longitude)).toFixed(4)}°
                  </span>
                </div>
                <div className="metadata-item">
                  <span className="metadata-label">Altitude Range</span>
                  <span className="metadata-value">
                    {Math.min(...metadata.gps.map(g => g.altitude)).toFixed(1)}m — {Math.max(...metadata.gps.map(g => g.altitude)).toFixed(1)}m
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Flight Path Map */}
          {gpsData && (
            <FlightMap gpsData={gpsData} />
          )}

          {/* Frame Gallery */}
          {frames.frames.length > 0 && (
            <FrameGallery
              jobId={jobId}
              frames={frames.frames}
              baseUrl={frames.base_url}
            />
          )}

          {/* Log Terminal */}
          <div className="log-terminal glass">
            <div className="log-header">
              <div className="log-header-title">
                <span className={`log-header-dot ${isConnected ? 'connected' : 'disconnected'}`} />
                Live Output
                {pipelineStatus === 'running' && (
                  <span style={{ fontSize: '0.7rem', color: 'var(--accent-cyan)', marginLeft: 8, fontFamily: 'var(--font-mono)' }}>
                    ⏱ {formatElapsed(elapsed)}
                  </span>
                )}
              </div>
              <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
                {['all', 'errors', 'stages'].map(f => (
                  <button
                    key={f}
                    className={`btn btn-ghost ${logFilter === f ? 'active' : ''}`}
                    style={{ fontSize: '0.65rem', padding: '2px 8px' }}
                    onClick={() => setLogFilter(f)}
                  >
                    {f.charAt(0).toUpperCase() + f.slice(1)}
                  </button>
                ))}
                <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginLeft: 8 }}>
                  {filteredLogs.length} lines
                </span>
              </div>
            </div>
            <div className="log-body">
              {filteredLogs.length === 0 && (
                <div className="log-line" style={{ color: 'var(--text-muted)' }}>
                  {logFilter === 'all' ? 'Waiting for pipeline output…' : `No ${logFilter} to show`}
                </div>
              )}
              {filteredLogs.map((line, i) => (
                <div key={i} className={getLogLineClass(line)}>
                  {line.stage && <span style={{ color: 'var(--text-muted)' }}>[{line.stage}] </span>}
                  {line.text}
                </div>
              ))}
              <div ref={logEndRef} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
