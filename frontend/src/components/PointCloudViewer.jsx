import { useRef, useState, useEffect, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Html } from '@react-three/drei';
import { Suspense } from 'react';
import * as THREE from 'three';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';

/**
 * PLY Point Cloud Viewer using Three.js.
 * Loads a .ply file and renders it as a colored point cloud.
 */

function PointCloud({ url, pointSize = 0.02 }) {
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

        const size = new THREE.Vector3();
        box.getSize(size);
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 8 / maxDim;
        geo.scale(scale, scale, scale);

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

export default function PointCloudViewer({ url, pointSize = 0.02, bgColor = '#06060a' }) {
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
      style={{ background: bgColor, transition: 'background 0.3s ease' }}
    >
      <ambientLight intensity={0.6} />

      <Suspense fallback={null}>
        <PointCloud url={url} pointSize={pointSize} />
      </Suspense>

      <gridHelper args={[20, 40, '#1a1a2e', '#111118']} position={[0, -4, 0]} />

      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        minDistance={1}
        maxDistance={80}
      />
    </Canvas>
  );
}
