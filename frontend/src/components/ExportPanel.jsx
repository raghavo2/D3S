import { useState, useEffect } from 'react';
import { API_BASE } from '../utils/constants';
import './ExportPanel.css';

/**
 * Collapsible export panel listing all output files for a job.
 */
export default function ExportPanel({ jobId }) {
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!jobId) return;
    setLoading(true);
    fetch(`${API_BASE}/api/jobs/${jobId}/outputs`)
      .then(r => r.json())
      .then(data => {
        setFiles(data.files || []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [jobId]);

  const formatSize = (mb) => {
    if (mb < 1) return `${(mb * 1024).toFixed(0)} KB`;
    return `${mb.toFixed(1)} MB`;
  };

  const getFileIcon = (filename) => {
    const ext = filename.split('.').pop().toLowerCase();
    switch (ext) {
      case 'glb': return '🔺';
      case 'ply': return '☁️';
      case 'las': return '📊';
      case 'obj': return '🧊';
      case 'csv': return '📋';
      default: return '📄';
    }
  };

  if (!jobId || files.length === 0) return null;

  return (
    <div className={`export-panel ${open ? 'open' : ''}`}>
      <button className="export-toggle" onClick={() => setOpen(!open)}>
        📦 Exports ({files.length})
        <span className={`export-chevron ${open ? 'up' : ''}`}>▾</span>
      </button>

      {open && (
        <div className="export-list">
          {loading ? (
            <div className="export-loading">Loading files…</div>
          ) : (
            files.map((file, i) => (
              <a
                key={i}
                href={`${API_BASE}${file.download_url}`}
                download
                className="export-item"
              >
                <span className="export-item-icon">{getFileIcon(file.filename)}</span>
                <span className="export-item-name">{file.filename}</span>
                <span className="export-item-size">{formatSize(file.size_mb)}</span>
                <span className="export-item-dl">⬇</span>
              </a>
            ))
          )}
        </div>
      )}
    </div>
  );
}
