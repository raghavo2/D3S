import { useState } from 'react';
import { API_BASE } from '../utils/constants';
import './FrameGallery.css';

/**
 * Thumbnail grid showing extracted frames with lightbox overlay.
 */
export default function FrameGallery({ jobId, frames, baseUrl }) {
  const [lightboxIdx, setLightboxIdx] = useState(null);
  const [expanded, setExpanded] = useState(false);

  if (!frames || frames.length === 0) return null;

  const displayFrames = expanded ? frames : frames.slice(0, 12);

  const openLightbox = (idx) => setLightboxIdx(idx);
  const closeLightbox = () => setLightboxIdx(null);
  const prevFrame = () => setLightboxIdx((prev) => (prev > 0 ? prev - 1 : frames.length - 1));
  const nextFrame = () => setLightboxIdx((prev) => (prev < frames.length - 1 ? prev + 1 : 0));

  return (
    <div className="frame-gallery glass">
      <div className="frame-gallery-header">
        <h3 className="frame-gallery-title">🎞️ Extracted Frames</h3>
        <span className="frame-gallery-count">{frames.length} frames</span>
      </div>

      <div className="frame-grid">
        {displayFrames.map((fname, i) => (
          <button
            key={fname}
            className="frame-thumb"
            onClick={() => openLightbox(i)}
          >
            <img
              src={`${API_BASE}${baseUrl}/${fname}`}
              alt={fname}
              loading="lazy"
            />
            <span className="frame-thumb-label">{fname}</span>
          </button>
        ))}
      </div>

      {frames.length > 12 && (
        <button
          className="frame-gallery-toggle"
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? 'Show less ▲' : `Show all ${frames.length} frames ▼`}
        </button>
      )}

      {/* Lightbox */}
      {lightboxIdx !== null && (
        <div className="lightbox-overlay" onClick={closeLightbox}>
          <div className="lightbox-content" onClick={(e) => e.stopPropagation()}>
            <button className="lightbox-close" onClick={closeLightbox}>✕</button>
            <button className="lightbox-nav lightbox-prev" onClick={prevFrame}>‹</button>

            <img
              src={`${API_BASE}${baseUrl}/${frames[lightboxIdx]}`}
              alt={frames[lightboxIdx]}
              className="lightbox-image"
            />

            <button className="lightbox-nav lightbox-next" onClick={nextFrame}>›</button>

            <div className="lightbox-info">
              <span>{frames[lightboxIdx]}</span>
              <span className="lightbox-counter">{lightboxIdx + 1} / {frames.length}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
