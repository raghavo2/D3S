import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Landing from './pages/Landing';
import Upload from './pages/Upload';
import Pipeline from './pages/Pipeline';
import Viewer from './pages/Viewer';
import History from './pages/History';
import './App.css';

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/pipeline/:jobId" element={<Pipeline />} />
        <Route path="/viewer" element={<Viewer />} />
        <Route path="/history" element={<History />} />
      </Routes>
    </BrowserRouter>
  );
}
