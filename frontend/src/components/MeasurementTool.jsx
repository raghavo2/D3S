import { useState, useRef, useCallback } from 'react';
import * as THREE from 'three';

/**
 * Interactive click-to-measure tool for 3D viewers.
 * 
 * Raycasts to the model/point cloud surface and places measurement markers.
 * Shows distance labels between clicked points. Supports multiple measurements.
 * 
 * This component renders inside a Three.js Canvas as a scene overlay.
 */

function MeasurementLine({ start, end, scaleFactor = 1.0, isMetric = false }) {
  const midpoint = new THREE.Vector3().lerpVectors(start, end, 0.5);
  const rawDist = start.distanceTo(end);
  const displayDist = rawDist * scaleFactor;
  const unit = isMetric ? 'm' : 'units';

  const formatDist = (d) => {
    if (d >= 1000) return `${(d / 1000).toFixed(2)} k${unit}`;
    if (d >= 1) return `${d.toFixed(2)} ${unit}`;
    if (d >= 0.01) return `${(d * 100).toFixed(1)} c${unit === 'm' ? 'm' : unit}`;
    return `${d.toFixed(4)} ${unit}`;
  };

  // Create a geometry for the line
  const points = [start, end];
  const lineGeo = new THREE.BufferGeometry().setFromPoints(points);

  return (
    <group>
      {/* Measurement line */}
      <line geometry={lineGeo}>
        <lineBasicMaterial color="#10b981" linewidth={2} transparent opacity={0.8} />
      </line>

      {/* Start sphere */}
      <mesh position={start}>
        <sphereGeometry args={[0.06, 16, 16]} />
        <meshBasicMaterial color="#10b981" transparent opacity={0.9} />
      </mesh>

      {/* End sphere */}
      <mesh position={end}>
        <sphereGeometry args={[0.06, 16, 16]} />
        <meshBasicMaterial color="#10b981" transparent opacity={0.9} />
      </mesh>
    </group>
  );
}

function MeasurementMarker({ position, index }) {
  return (
    <mesh position={position}>
      <sphereGeometry args={[0.08, 16, 16]} />
      <meshBasicMaterial color="#f59e0b" transparent opacity={0.9} />
    </mesh>
  );
}

/**
 * Hook for measurement tool state management.
 * Use this in the parent Viewer component.
 */
export function useMeasurementTool() {
  const [measurements, setMeasurements] = useState([]);
  const [pendingPoint, setPendingPoint] = useState(null);
  const [measuring, setMeasuring] = useState(false);

  const startMeasuring = useCallback(() => {
    setMeasuring(true);
    setPendingPoint(null);
  }, []);

  const stopMeasuring = useCallback(() => {
    setMeasuring(false);
    setPendingPoint(null);
  }, []);

  const toggleMeasuring = useCallback(() => {
    if (measuring) {
      stopMeasuring();
    } else {
      startMeasuring();
    }
  }, [measuring]);

  const addPoint = useCallback((point) => {
    if (!measuring) return;

    if (!pendingPoint) {
      // First click — set start point
      setPendingPoint(point.clone());
    } else {
      // Second click — complete measurement
      setMeasurements(prev => [...prev, {
        id: Date.now(),
        start: pendingPoint.clone(),
        end: point.clone(),
      }]);
      setPendingPoint(null);
    }
  }, [measuring, pendingPoint]);

  const clearMeasurements = useCallback(() => {
    setMeasurements([]);
    setPendingPoint(null);
  }, []);

  const removeLast = useCallback(() => {
    setMeasurements(prev => prev.slice(0, -1));
  }, []);

  return {
    measurements,
    pendingPoint,
    measuring,
    startMeasuring,
    stopMeasuring,
    toggleMeasuring,
    addPoint,
    clearMeasurements,
    removeLast,
  };
}

/**
 * Renders all measurement lines and markers inside a Three.js scene.
 */
export function MeasurementOverlay({
  measurements = [],
  pendingPoint = null,
  scaleFactor = 1.0,
  isMetric = false,
}) {
  return (
    <group>
      {measurements.map((m) => (
        <MeasurementLine
          key={m.id}
          start={m.start}
          end={m.end}
          scaleFactor={scaleFactor}
          isMetric={isMetric}
        />
      ))}
      {pendingPoint && (
        <MeasurementMarker position={pendingPoint} index={0} />
      )}
    </group>
  );
}

/**
 * Floating HUD showing measurement results and controls.
 * Renders as a regular DOM element (outside Canvas).
 */
export function MeasurementHUD({
  measurements = [],
  measuring = false,
  pendingPoint = null,
  scaleFactor = 1.0,
  isMetric = false,
  onClear,
  onRemoveLast,
  onToggle,
}) {
  if (!measuring && measurements.length === 0) return null;

  const unit = isMetric ? 'm' : 'units';
  const formatDist = (d) => {
    const scaled = d * scaleFactor;
    if (scaled >= 1) return `${scaled.toFixed(2)} ${unit}`;
    return `${(scaled * 100).toFixed(1)} c${unit === 'm' ? 'm' : unit}`;
  };

  return (
    <div style={{
      position: 'absolute',
      top: 16,
      right: 16,
      zIndex: 20,
      background: 'rgba(10, 10, 18, 0.92)',
      backdropFilter: 'blur(16px)',
      border: '1px solid rgba(16, 185, 129, 0.2)',
      borderRadius: '12px',
      padding: '12px 16px',
      color: '#e2e8f0',
      fontSize: '0.78rem',
      minWidth: '200px',
      maxWidth: '260px',
      boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
    }}>
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        marginBottom: '8px', paddingBottom: '6px',
        borderBottom: '1px solid rgba(255,255,255,0.06)',
      }}>
        <span style={{ fontWeight: 600, fontSize: '0.8rem' }}>
          📏 Measure {measuring && '(active)'}
        </span>
        <div style={{ display: 'flex', gap: '4px' }}>
          {measurements.length > 0 && (
            <>
              <button onClick={onRemoveLast} style={{
                background: 'none', border: 'none', color: '#64748b',
                fontSize: '0.7rem', cursor: 'pointer', padding: '2px 4px',
              }}>Undo</button>
              <button onClick={onClear} style={{
                background: 'none', border: 'none', color: '#ef4444',
                fontSize: '0.7rem', cursor: 'pointer', padding: '2px 4px',
              }}>Clear</button>
            </>
          )}
        </div>
      </div>

      {measuring && (
        <div style={{
          fontSize: '0.7rem', color: '#94a3b8', marginBottom: '6px',
          padding: '4px 6px', background: 'rgba(16,185,129,0.08)',
          borderRadius: '4px',
        }}>
          {pendingPoint
            ? '🟡 Click second point to measure'
            : '🟢 Click on the model to place first point'}
        </div>
      )}

      {measurements.map((m, i) => (
        <div key={m.id} style={{
          display: 'flex', justifyContent: 'space-between',
          padding: '3px 6px', fontSize: '0.72rem',
          borderRadius: '4px',
        }}>
          <span style={{ color: '#94a3b8' }}>#{i + 1}</span>
          <span style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontWeight: 600, color: '#10b981',
          }}>
            {formatDist(m.start.distanceTo(m.end))}
          </span>
        </div>
      ))}
    </div>
  );
}

export default MeasurementLine;
