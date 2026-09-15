import { useRef, useMemo, useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import './Landing.css';

/* ─── Subtle Floating Grid (Three.js background) ─── */
function FloatingGrid() {
  const ref = useRef();
  const count = 800;

  const positions = useMemo(() => {
    const pos = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 40;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 40;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 20;
    }
    return pos;
  }, []);

  useFrame((state) => {
    if (ref.current) {
      ref.current.rotation.y = state.clock.elapsedTime * 0.008;
      ref.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.005) * 0.05;
    }
  });

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          array={positions}
          count={count}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.03}
        color="#ffffff"
        transparent
        opacity={0.12}
        sizeAttenuation
        depthWrite={false}
      />
    </points>
  );
}

/* ─── Animated Counter ─── */
function AnimatedStat({ value, label, suffix = '' }) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    const duration = 2000;
    const steps = 60;
    const increment = value / steps;
    let current = 0;
    const timer = setInterval(() => {
      current += increment;
      if (current >= value) {
        setCount(value);
        clearInterval(timer);
      } else {
        setCount(Math.floor(current));
      }
    }, duration / steps);
    return () => clearInterval(timer);
  }, [value]);

  return (
    <div className="stat-item">
      <span className="stat-value">{count}{suffix}</span>
      <span className="stat-label">{label}</span>
    </div>
  );
}

/* ─── Landing Page ─── */
export default function Landing() {
  return (
    <div className="landing">
      {/* Hero Section */}
      <section className="hero">
        <div className="hero-canvas">
          <Canvas camera={{ position: [0, 0, 15], fov: 50 }}>
            <FloatingGrid />
          </Canvas>
        </div>

        <div className="hero-inner">
          {/* Left — Copy */}
          <div className="hero-left animate-slide-up">
            <p className="hero-eyebrow">3D Reconstruction Platform</p>

            <h1 className="hero-title">
              Drone Video<br />
              <span className="hero-title-light">to 3D Model</span>
            </h1>

            <p className="hero-subtitle">
              Transform a single drone flight into georeferenced, 
              metrically accurate 3D models. No GCPs, no multi-pass overlap.
            </p>

            <div className="hero-actions">
              <Link to="/upload" className="hero-btn-primary">
                Start Reconstruction
                <span className="hero-btn-arrow">→</span>
              </Link>
              <Link to="/viewer" className="hero-btn-secondary">
                View Demo
              </Link>
            </div>
          </div>

          {/* Right — Stats Panel */}
          <div className="hero-right animate-slide-up" style={{ animationDelay: '0.15s' }}>
            <div className="stats-panel">
              <div className="stats-grid">
                <AnimatedStat value={5} label="Pipeline Stages" />
                <AnimatedStat value={4} label="Export Formats" />
                <AnimatedStat value={178} label="Model Output" suffix=" MB" />
                <AnimatedStat value={38} label="GPS Points" />
              </div>
              <div className="stats-divider" />
              <div className="stats-stack">
                <div className="stack-row">
                  <span className="stack-dot stack-dot-1" />
                  <span>DUSt3R Multi-View Stereo</span>
                </div>
                <div className="stack-row">
                  <span className="stack-dot stack-dot-2" />
                  <span>Poisson Surface Reconstruction</span>
                </div>
                <div className="stack-row">
                  <span className="stack-dot stack-dot-3" />
                  <span>GLB / PLY / LAS / OBJ</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="features container">
        <div className="features-header">
          <p className="features-label">Capabilities</p>
          <h2 className="features-title">Built for Precision</h2>
        </div>

        <div className="features-grid stagger">
          <div className="feature-card">
            <div className="feature-number">01</div>
            <h3 className="feature-heading">AI-Driven Reconstruction</h3>
            <p className="feature-text">
              DUSt3R multi-view stereo infers depth and constructs dense point
              clouds directly from single-pass imagery.
            </p>
          </div>

          <div className="feature-card">
            <div className="feature-number">02</div>
            <h3 className="feature-heading">High-Fidelity Meshing</h3>
            <p className="feature-text">
              Screened Poisson Surface Reconstruction with Taubin smoothing
              produces crisp, continuous surfaces.
            </p>
          </div>

          <div className="feature-card">
            <div className="feature-number">03</div>
            <h3 className="feature-heading">Multi-Format Export</h3>
            <p className="feature-text">
              Outputs compile into industry-standard formats: GLB for web,
              LAS for GIS, OBJ for CAD workflows.
            </p>
          </div>
        </div>
      </section>

      {/* Pipeline */}
      <section className="pipeline-section container">
        <p className="features-label">Pipeline</p>
        <h2 className="features-title">Five Steps to 3D</h2>

        <div className="pipeline-flow">
          {['Frame Extraction', 'Metadata Parse', 'Point Cloud', 'Surface Mesh', 'Export'].map((step, i) => (
            <div key={step} className="pipeline-node">
              <span className="pipeline-num">{String(i + 1).padStart(2, '0')}</span>
              <span className="pipeline-label">{step}</span>
              {i < 4 && <span className="pipeline-connector" />}
            </div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="landing-footer">
        <div className="container">
          D3S — Single-Pass Drone Video to 3D Model Generation
        </div>
      </footer>
    </div>
  );
}
