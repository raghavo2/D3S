import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useApi } from '../hooks/useApi';
import { API_BASE } from '../utils/constants';
import './History.css';

export default function History() {
  const navigate = useNavigate();
  const { listJobs } = useApi();
  const [jobsList, setJobsList] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listJobs()
      .then(data => {
        setJobsList(data.jobs || []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const deleteJob = async (jobId, e) => {
    e.stopPropagation();
    if (!confirm('Delete this job and all its files?')) return;
    try {
      await fetch(`${API_BASE}/api/jobs/${jobId}`, { method: 'DELETE' });
      setJobsList(prev => prev.filter(j => j.id !== jobId));
    } catch { /* ignore */ }
  };

  const formatDate = (iso) => {
    if (!iso) return '—';
    const d = new Date(iso);
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  };

  const getStatusBadge = (status) => {
    const cls = status === 'complete' ? 'badge-success' :
                status === 'running' ? 'badge-info' :
                status === 'failed' || status === 'interrupted' ? 'badge-error' :
                'badge-warning';
    const label = status === 'interrupted' ? 'Interrupted' : status.charAt(0).toUpperCase() + status.slice(1);
    return <span className={`badge ${cls}`}><span className={`status-dot ${status}`} />{label}</span>;
  };

  return (
    <div className="history-page page">
      <div className="history-container animate-fade-in">
        <div className="history-header">
          <h1 className="gradient-text">Job History</h1>
          <span className="history-count">{jobsList.length} jobs</span>
        </div>

        {loading ? (
          <div className="history-empty">
            <div className="spinner" style={{ margin: '0 auto', width: 32, height: 32 }} />
          </div>
        ) : jobsList.length === 0 ? (
          <div className="history-empty glass">
            <span className="history-empty-icon">📂</span>
            <h2>No reconstructions yet</h2>
            <p>Upload a drone video to create your first 3D reconstruction.</p>
            <Link to="/upload" className="btn btn-primary">Upload Video →</Link>
          </div>
        ) : (
          <div className="job-list stagger">
            {jobsList.map(job => (
              <div
                key={job.id}
                className="job-row glass"
                onClick={() => {
                  if (job.status === 'complete') navigate(`/viewer?job=${job.id}`);
                  else navigate(`/pipeline/${job.id}`);
                }}
                style={{ cursor: 'pointer' }}
              >
                <div className="job-row-info">
                  <div className="job-row-id">
                    {job.video_filename || `Job ${job.id.slice(0, 8)}`}
                  </div>
                  <div className="job-row-file">
                    {job.id.slice(0, 8)}… · {job.fps || 2} FPS
                  </div>
                </div>

                {getStatusBadge(job.status)}

                <div className="job-row-date">
                  {formatDate(job.created_at)}
                </div>

                <div className="job-row-size">
                  {job.video_size_mb ? `${job.video_size_mb} MB` : '—'}
                </div>

                <div className="job-row-actions">
                  {job.status === 'complete' && (
                    <button
                      className="job-action-btn"
                      title="Open in Viewer"
                      onClick={(e) => { e.stopPropagation(); navigate(`/viewer?job=${job.id}`); }}
                    >
                      🔍
                    </button>
                  )}
                  <button
                    className="job-action-btn danger"
                    title="Delete Job"
                    onClick={(e) => deleteJob(job.id, e)}
                  >
                    🗑️
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
