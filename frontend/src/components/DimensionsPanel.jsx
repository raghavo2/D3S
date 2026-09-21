import { useState, useEffect } from 'react';
import './DimensionsPanel.css';

/**
 * Floating panel that displays real-world structure dimensions.
 * 
 * Shows Width × Height × Depth in meters based on the GPS-derived
 * scale factor from the pipeline. Supports manual scale override
 * when GPS data is unavailable.
 */
export default function DimensionsPanel({
  scaleInfo,
  boundingBox,
  onScaleOverride,
  visible = true,
}) {
  const [collapsed, setCollapsed] = useState(false);
  const [manualScale, setManualScale] = useState('');
  const [showOverride, setShowOverride] = useState(false);

  if (!visible) return null;

  const scaleFactor = scaleInfo?.scale_factor || 1.0;
  const isMetric = scaleInfo?.reliable || scaleInfo?.method === 'manual_override';
  const method = scaleInfo?.method || 'none';

  // Compute dimensions from bounding box
  let width = 0, height = 0, depth = 0;
  if (boundingBox) {
    width = (boundingBox.max[0] - boundingBox.min[0]) * scaleFactor;
    height = (boundingBox.max[1] - boundingBox.min[1]) * scaleFactor;
    depth = (boundingBox.max[2] - boundingBox.min[2]) * scaleFactor;
  } else if (scaleInfo?.dimensions_m) {
    width = scaleInfo.dimensions_m.width_m || 0;
    height = scaleInfo.dimensions_m.height_m || 0;
    depth = scaleInfo.dimensions_m.depth_m || 0;
  }

  const volume = width * height * depth;
  const unit = isMetric ? 'm' : 'units';

  const formatDim = (val) => {
    if (val >= 1000) return `${(val / 1000).toFixed(2)} k${unit}`;
    if (val >= 1) return `${val.toFixed(2)} ${unit}`;
    if (val >= 0.01) return `${(val * 100).toFixed(1)} c${unit === 'm' ? 'm' : unit}`;
    return `${val.toFixed(4)} ${unit}`;
  };

  const handleScaleSubmit = () => {
    const val = parseFloat(manualScale);
    if (!isNaN(val) && val > 0 && onScaleOverride) {
      onScaleOverride(val);
      setShowOverride(false);
      setManualScale('');
    }
  };

  return (
    <div className={`dimensions-panel ${collapsed ? 'collapsed' : ''}`}>
      <div className="dim-header" onClick={() => setCollapsed(!collapsed)}>
        <span className="dim-icon">📏</span>
        <span className="dim-title">Dimensions</span>
        <span className="dim-toggle">{collapsed ? '▶' : '▼'}</span>
      </div>

      {!collapsed && (
        <div className="dim-body">
          {/* Scale Method Indicator */}
          <div className="dim-method">
            <span className={`dim-method-dot ${isMetric ? 'metric' : 'relative'}`} />
            <span className="dim-method-label">
              {method === 'gps_haversine' && 'GPS-scaled (meters)'}
              {method === 'manual_override' && 'Manual scale (meters)'}
              {method === 'none' && 'Relative (no GPS)'}
              {method === 'fallback_none' && 'Relative (no GPS)'}
            </span>
          </div>

          {/* Dimension Values */}
          <div className="dim-grid">
            <div className="dim-row">
              <span className="dim-label">Width</span>
              <span className="dim-value">{formatDim(width)}</span>
            </div>
            <div className="dim-row">
              <span className="dim-label">Height</span>
              <span className="dim-value">{formatDim(height)}</span>
            </div>
            <div className="dim-row">
              <span className="dim-label">Depth</span>
              <span className="dim-value">{formatDim(depth)}</span>
            </div>
            {isMetric && volume > 0 && (
              <div className="dim-row dim-row-volume">
                <span className="dim-label">Volume</span>
                <span className="dim-value">{volume.toFixed(2)} m³</span>
              </div>
            )}
          </div>

          {/* Scale Factor Display */}
          {isMetric && (
            <div className="dim-scale-info">
              <span>Scale: 1 unit = {scaleFactor.toFixed(4)} m</span>
              {scaleInfo?.median_residual_m !== undefined && (
                <span className="dim-residual">
                  ±{scaleInfo.median_residual_m.toFixed(3)} m
                </span>
              )}
            </div>
          )}

          {/* Manual Scale Override */}
          {!isMetric && (
            <div className="dim-override">
              {!showOverride ? (
                <button
                  className="dim-override-btn"
                  onClick={() => setShowOverride(true)}
                >
                  Set scale manually
                </button>
              ) : (
                <div className="dim-override-form">
                  <input
                    type="number"
                    className="dim-override-input"
                    placeholder="Scale factor"
                    value={manualScale}
                    onChange={(e) => setManualScale(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleScaleSubmit()}
                    step="0.001"
                    min="0.001"
                    autoFocus
                  />
                  <button className="dim-override-apply" onClick={handleScaleSubmit}>
                    Apply
                  </button>
                  <button
                    className="dim-override-cancel"
                    onClick={() => { setShowOverride(false); setManualScale(''); }}
                  >
                    ✕
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
