import { useRef, useState, useEffect, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Html } from '@react-three/drei';
import { Suspense } from 'react';
import * as THREE from 'three';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
import { MeasurementOverlay } from './MeasurementTool';

/**
 * PLY Point Cloud Viewer using Three.js.
 * 
 * V3 upgrades:
 *   - Reports bounding box after loading
 *   - Scale factor support for metric dimensions
 *   - Bounding box wireframe with dimension labels
 *   - Measurement overlay support
 */

function BoundingBoxWireframe({ boundingBox, scaleFactor = 1.0, isMetric = false }) {
  if (!boundingBox) return null;

  const { min, max } = boundingBox;
  const center = [
    (min[0] + max[0]) / 2,
    (min[1] + max[1]) / 2,
    (min[2] + max[2]) / 2,
  ];
  const size = [
    max[0] - min[0],
    max[1] - min[1],
    max[2] - min[2],
  ];

  const formatDim = (v) => {
    const scaled = v * scaleFactor;
    if (isMetric) {
      if (scaled >= 1) return `${scaled.toFixed(1)}m`;
      return `${(scaled * 100).toFixed(0)}cm`;
    }
    return scaled.toFixed(2);
  };

  return (
    <group>
      <mesh position={center}>
        <boxGeometry args={size} />
        <meshBasicMaterial color="#10b981" wireframe transparent opacity={0.2} />
      </mesh>

      <Html position={[center[0], min[1] - 0.3, center[2]]} center>
        <div style={{
          background: 'rgba(10,10,18,0.85)', color: '#10b981',
          padding: '2px 8px', borderRadius: '4px', fontSize: '0.65rem',
          fontFamily: "'JetBrains Mono', monospace", fontWeight: 600,
          border: '1px solid rgba(16,185,129,0.3)', whiteSpace: 'nowrap',
          pointerEvents: 'none',
        }}>
          W: {formatDim(size[0])}
        </div>
      </Html>

      <Html position={[min[0] - 0.3, center[1], center[2]]} center>
        <div style={{
          background: 'rgba(10,10,18,0.85)', color: '#38bdf8',
          padding: '2px 8px', borderRadius: '4px', fontSize: '0.65rem',
          fontFamily: "'JetBrains Mono', monospace", fontWeight: 600,
          border: '1px solid rgba(56,189,248,0.3)', whiteSpace: 'nowrap',
          pointerEvents: 'none',
        }}>
          H: {formatDim(size[1])}
        </div>
      </Html>

      <Html position={[center[0], min[1] - 0.3, max[2] + 0.3]} center>
        <div style={{
          background: 'rgba(10,10,18,0.85)', color: '#a78bfa',
          padding: '2px 8px', borderRadius: '4px', fontSize: '0.65rem',
          fontFamily: "'JetBrains Mono', monospace", fontWeight: 600,
          border: '1px solid rgba(167,139,250,0.3)', whiteSpace: 'nowrap',
          pointerEvents: 'none',
        }}>
          D: {formatDim(size[2])}
        </div>
      </Html>
    </group>
  );
}

function PointCloud({ url, pointSize = 0.02, onBoundingBox }) {
  const ref = useRef();
  const [geometry, setGeometry] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [pointCount, setPointCount] = useState(0);

  useEffect(() => {
    if (!url) return;
    setLoading(true);
    setError(null);

    const loader = new PLYLoader();
    loader.load(
      url,
      (geo) => {
        geo.computeBoundingBox();
        const box = geo.boundingBox;
        const center = new THREE.Vector3();
        box.getCenter(center);
        geo.translate(-center.x, -center.y, -center.z);

        // Rotate from Z-up (reconstruction) to Y-up (Three.js)
        geo.rotateX(-Math.PI / 2);

        const size = new THREE.Vector3();
        box.getSize(size);
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 8 / maxDim;
        geo.scale(scale, scale, scale);

        // Report bounding box after normalization
        geo.computeBoundingBox();
        if (onBoundingBox) {
          onBoundingBox({
            min: geo.boundingBox.min.toArray(),
            max: geo.boundingBox.max.toArray(),
            viewScale: scale,
          });
        }

        setGeometry(geo);
        setPointCount(geo.attributes.position.count);
        setLoading(false);
      },
      undefined,
      (err) => {
        setError(err.message || 'Failed to load point cloud');
        setLoading(false);
      }
    );
  }, [url]);

  // Gentle rotation
  useFrame((_, delta) => {
    if (ref.current) {
      ref.current.rotation.y += delta * 0.05;
    }
  });

  if (loading) {
    return (
      <Html center>
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '12px',
          color: '#94a3b8',
        }}>
          <div className="spinner" style={{ width: 32, height: 32 }} />
          <span style={{ fontSize: '0.85rem' }}>Loading point cloud…</span>
        </div>
      </Html>
    );
  }

  if (error) {
    return (
      <Html center>
        <div style={{ color: '#ef4444', fontSize: '0.85rem' }}>
          Error: {error}
        </div>
      </Html>
    );
  }

  if (!geometry) return null;

  return (
    <group ref={ref}>
      <points>
        <bufferGeometry attach="geometry" {...geometry} />
        <pointsMaterial
          attach="material"
          size={pointSize}
          vertexColors
          sizeAttenuation
          transparent
          opacity={0.9}
        />
      </points>
    </group>
  );
}

export default function PointCloudViewer({
  url,
  pointSize = 0.02,
  bgColor = '#06060a',
  showBoundingBox = false,
  scaleFactor = 1.0,
  isMetric = false,
  onBoundingBox,
  // Measurement tool props
  measuring = false,
  measurements = [],
  pendingPoint = null,
}) {
  const [bbox, setBbox] = useState(null);

  const handleBbox = useCallback((bb) => {
    setBbox(bb);
    if (onBoundingBox) onBoundingBox(bb);
  }, [onBoundingBox]);

  if (!url) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        color: 'var(--text-muted)',
        fontSize: '0.9rem',
      }}>
        No point cloud URL provided
      </div>
    );
  }

  return (
    <Canvas
      camera={{ position: [8, 5, 8], fov: 50, near: 0.01, far: 500 }}
      style={{
        background: bgColor,
        transition: 'background 0.3s ease',
        cursor: measuring ? 'crosshair' : 'grab',
      }}
    >
      <ambientLight intensity={0.6} />

      <Suspense fallback={null}>
        <PointCloud url={url} pointSize={pointSize} onBoundingBox={handleBbox} />
      </Suspense>

      <gridHelper args={[20, 40, '#1a1a2e', '#111118']} position={[0, -4, 0]} />

      {/* V3: Bounding box wireframe */}
      {showBoundingBox && bbox && (
        <BoundingBoxWireframe
          boundingBox={bbox}
          scaleFactor={scaleFactor}
          isMetric={isMetric}
        />
      )}

      {/* V3: Measurement overlay */}
      <MeasurementOverlay
        measurements={measurements}
        pendingPoint={pendingPoint}
        scaleFactor={scaleFactor}
        isMetric={isMetric}
      />

      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        minDistance={1}
        maxDistance={80}
        enabled={!measuring}
      />
    </Canvas>
  );
}
