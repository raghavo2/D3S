import { useRef, useMemo, useState, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Center, useGLTF, Html, Environment } from '@react-three/drei';
import { Suspense } from 'react';
import * as THREE from 'three';
import { MeasurementOverlay } from './MeasurementTool';

/**
 * GLB/GLTF Model Viewer using Three.js via react-three-fiber.
 * 
 * V3 upgrades:
 *   - Reports bounding box dimensions after loading
 *   - Supports scaleFactor for metric display
 *   - Shows optional bounding box wireframe with dimension labels
 *   - Raycasting support for measurement tool
 */

function Model({ url, onLoaded, onBoundingBox }) {
  const { scene } = useGLTF(url);
  const ref = useRef();

  // Auto-center, scale, and fix orientation (Z-up → Y-up)
  useMemo(() => {
    // First, rotate from Z-up to Y-up
    scene.rotation.x = -Math.PI / 2;
    scene.updateMatrixWorld(true);

    const box = new THREE.Box3().setFromObject(scene);
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const scale = 5 / maxDim;
    scene.scale.setScalar(scale);

    const center = box.getCenter(new THREE.Vector3());
    scene.position.sub(center.multiplyScalar(scale));

    // Report bounding box (in normalized units, before scale factor)
    if (onBoundingBox) {
      const newBox = new THREE.Box3().setFromObject(scene);
      const min = newBox.min.toArray();
      const max = newBox.max.toArray();
      onBoundingBox({ min, max, viewScale: scale });
    }

    if (onLoaded) onLoaded();
  }, [scene]);

  return <primitive ref={ref} object={scene} />;
}

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

  const unit = isMetric ? 'm' : '';
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
      {/* Wireframe box */}
      <mesh position={center}>
        <boxGeometry args={size} />
        <meshBasicMaterial
          color="#10b981"
          wireframe
          transparent
          opacity={0.25}
        />
      </mesh>

      {/* Width label (X axis) */}
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

      {/* Height label (Y axis) */}
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

      {/* Depth label (Z axis) */}
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

/** Raycaster helper for measurement tool clicks */
function ClickHandler({ onSurfaceClick, measuring }) {
  const { camera, raycaster, scene } = useThree();

  const handleClick = useCallback((event) => {
    if (!measuring || !onSurfaceClick) return;

    // Compute normalized device coordinates
    const rect = event.target.getBoundingClientRect();
    const mouse = new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1
    );

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(scene.children, true);

    if (intersects.length > 0) {
      onSurfaceClick(intersects[0].point);
    }
  }, [measuring, onSurfaceClick, camera, raycaster, scene]);

  return (
    <mesh
      visible={false}
      onPointerDown={measuring ? handleClick : undefined}
    >
      <planeGeometry args={[1000, 1000]} />
      <meshBasicMaterial transparent opacity={0} />
    </mesh>
  );
}

function LoadingFallback() {
  return (
    <Html center>
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '16px',
        color: '#94a3b8',
        background: 'rgba(6, 6, 10, 0.8)',
        padding: '32px 48px',
        borderRadius: '12px',
        border: '1px solid rgba(255,255,255,0.06)',
      }}>
        <div style={{
          width: 40, height: 40,
          border: '3px solid rgba(148,163,184,0.2)',
          borderTopColor: 'rgba(255,255,255,0.6)',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite',
        }} />
        <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>Loading 3D model…</span>
        <span style={{ fontSize: '0.72rem', color: '#64748b' }}>Large models may take a moment</span>
      </div>
    </Html>
  );
}

function GridFloor() {
  return (
    <gridHelper args={[20, 40, '#1a1a2e', '#111118']} position={[0, -2.5, 0]} />
  );
}

export default function ModelViewer({
  url,
  showGrid = true,
  autoRotate = false,
  bgColor = '#06060a',
  showBoundingBox = false,
  scaleFactor = 1.0,
  isMetric = false,
  onBoundingBox,
  // Measurement tool props
  measuring = false,
  measurements = [],
  pendingPoint = null,
  onSurfaceClick,
}) {
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
        No model URL provided
      </div>
    );
  }

  const [bbox, setBbox] = useState(null);

  const handleBbox = useCallback((bb) => {
    setBbox(bb);
    if (onBoundingBox) onBoundingBox(bb);
  }, [onBoundingBox]);

  return (
    <Canvas
      camera={{ position: [8, 5, 8], fov: 50, near: 0.1, far: 1000 }}
      style={{
        background: bgColor,
        transition: 'background 0.3s ease',
        cursor: measuring ? 'crosshair' : 'grab',
      }}
      gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping }}
    >
      <ambientLight intensity={0.4} />
      <directionalLight position={[10, 10, 5]} intensity={1} />
      <directionalLight position={[-5, 5, -5]} intensity={0.3} />

      <Suspense fallback={<LoadingFallback />}>
        <Center>
          <Model url={url} onBoundingBox={handleBbox} />
        </Center>
        <Environment preset="night" />
      </Suspense>

      {showGrid && <GridFloor />}

      {/* V3: Bounding box wireframe with dimension labels */}
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
        autoRotate={autoRotate}
        autoRotateSpeed={0.5}
        enableDamping
        dampingFactor={0.05}
        minDistance={1}
        maxDistance={50}
        enabled={!measuring}
      />
    </Canvas>
  );
}
