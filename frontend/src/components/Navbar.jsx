import { Link, useLocation } from 'react-router-dom';
import { useState, useEffect } from 'react';
import { useApi } from '../hooks/useApi';
import './Navbar.css';

export default function Navbar() {
  const location = useLocation();
  const { getHealth } = useApi();
  const [serverOnline, setServerOnline] = useState(false);

  useEffect(() => {
    let interval;
    const check = async () => {
      try {
        await getHealth();
        setServerOnline(true);
      } catch {
        setServerOnline(false);
      }
    };
    check();
    interval = setInterval(check, 15000);
    return () => clearInterval(interval);
  }, []);

  const isActive = (path) => location.pathname === path ? 'navbar-link active' : 'navbar-link';

  return (
    <nav className="navbar">
      <Link to="/" className="navbar-brand">
        <span className="navbar-logo gradient-text">D3S</span>
        <span className="navbar-tag">v1.0</span>
      </Link>

      <div className="navbar-links">
        <Link to="/" className={isActive('/')}>Home</Link>
        <Link to="/upload" className={isActive('/upload')}>Upload</Link>
        <Link to="/viewer" className={isActive('/viewer')}>3D Viewer</Link>
        <Link to="/history" className={isActive('/history')}>History</Link>
        <div className="navbar-status">
          <span className={`navbar-status-dot ${serverOnline ? 'online' : ''}`} />
          <span>{serverOnline ? 'Online' : 'Offline'}</span>
        </div>
      </div>
    </nav>
  );
}
