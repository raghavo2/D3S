import { useRef, useMemo, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Center, useGLTF, Html, Environment } from '@react-three/drei';
import { Suspense } from 'react';
import * as THREE from 'three';

/**
 * GLB/GLTF Model Viewer using Three.js via react-three-fiber.
 * Loads a .glb file from a URL and displays it with orbit controls.
 */

function Model({ url, onLoaded }) {
  const { scene } = useGLTF(url);
  const ref = useRef();

  // Auto-center and scale model
  useMemo(() => {
    const box = new THREE.Box3().setFromObject(scene);
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const scale = 5 / maxDim;
    scene.scale.setScalar(scale);

    const center = box.getCenter(new THREE.Vector3());
    scene.position.sub(center.multiplyScalar(scale));
    if (onLoaded) onLoaded();
  }, [scene]);

  return <primitive ref={ref} object={scene} />;
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

export default function ModelViewer({ url, showGrid = true, autoRotate = false, bgColor = '#06060a' }) {
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

  return (
    <Canvas
      camera={{ position: [8, 5, 8], fov: 50, near: 0.1, far: 1000 }}
      style={{ background: bgColor, transition: 'background 0.3s ease' }}
      gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping }}
    >
      <ambientLight intensity={0.4} />
      <directionalLight position={[10, 10, 5]} intensity={1} />
      <directionalLight position={[-5, 5, -5]} intensity={0.3} />

      <Suspense fallback={<LoadingFallback />}>
        <Center>
          <Model url={url} />
        </Center>
        <Environment preset="night" />
      </Suspense>

      {showGrid && <GridFloor />}

      <OrbitControls
        autoRotate={autoRotate}
        autoRotateSpeed={0.5}
        enableDamping
        dampingFactor={0.05}
        minDistance={1}
        maxDistance={50}
      />
    </Canvas>
  );
}
