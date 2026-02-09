import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE = 'http://localhost:8000';

function App() {
  const [image, setImage] = useState(null);
  const [audio, setAudio] = useState(null);
  const [format, setFormat] = useState('16:9');
  const [style, setStyle] = useState('surreal');
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    setImage(file);
  };

  const handleAudioChange = (e) => {
    const file = e.target.files[0];
    setAudio(file);
  };

  const uploadFiles = async () => {
    if (!image || !audio) {
      alert('Please select both image and audio files');
      return;
    }

    setIsProcessing(true);
    const formData = new FormData();
    formData.append('image', image);
    formData.append('audio', audio);
    formData.append('format', format);
    formData.append('style', style);

    try {
      const response = await axios.post(`${API_BASE}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setJobId(response.data.job_id);
      startPolling(response.data.job_id);
    } catch (error) {
      console.error('Upload failed:', error);
      alert('Upload failed: ' + (error.response?.data?.error || error.message));
      setIsProcessing(false);
    }
  };

  const startPolling = (jobId) => {
    const interval = setInterval(async () => {
      try {
        const response = await axios.get(`${API_BASE}/status/${jobId}`);
        setStatus(response.data);

        if (response.data.status === 'completed') {
          clearInterval(interval);
          setIsProcessing(false);
        } else if (response.data.status === 'error') {
          clearInterval(interval);
          setIsProcessing(false);
          alert('Processing failed: ' + response.data.message);
        }
      } catch (error) {
        console.error('Status check failed:', error);
      }
    }, 2000);
  };

  const downloadVideo = () => {
    if (jobId) {
      window.open(`${API_BASE}/download/${jobId}`);
    }
  };
return
(
    <div className="App">
      <header className="App-header">
        <h1>🎵 Pixel-Echo 🎬</h1>
        <p>AI MusicVideo Generator</p>
        <div className="tagline">Wo Beats zu visuellen Echos werden</div>
      </header>

      <main className="main-content">
        <div className="upload-section">
          <h2>Create Your Video</h2>
          
          <div className="file-inputs">
            <div className="input-group">
              <label htmlFor="image">📸 Select Image:</label>
              <input
                id="image"
                type="file"
                accept="image/*"
                onChange={handleImageChange}
                disabled={isProcessing}
              />
              {image && <span className="file-name">✓ {image.name}</span>}
            </div>

            <div className="input-group">
              <label htmlFor="audio">🎵 Select Audio:</label>
              <input
                id="audio"
                type="file"
                accept="audio/*"
                onChange={handleAudioChange}
                disabled={isProcessing}
              />
              {audio && <span className="file-name">✓ {audio.name}</span>}
            </div>
          </div>

          <div className="options">
            <div className="input-group">
              <label htmlFor="format">📐 Format:</label>
              <select
                id="format"
                value={format}
                onChange={(e) => setFormat(e.target.value)}
                disabled={isProcessing}
              >
                <option value="16:9">16:9 (YouTube/Landscape)</option>
                <option value="9:16">9:16 (TikTok/Portrait)</option>
              </select>
            </div>

            <div className="input-group">
              <label htmlFor="style">🎨 Style:</label>
              <select
                id="style"
                value={style}
                onChange={(e) => setStyle(e.target.value)}
                disabled={isProcessing}
              >
                <option value="surreal">Surreal Echo</option>
                <option value="glitch">Digital Glitch</option>
                <option value="abstract">Abstract Flow</option>
              </select>
            </div>
          </div>

          <button
            className="upload-btn"
            onClick={uploadFiles}
            disabled={isProcessing || !image || !audio}
          >
            {isProcessing ? '🔄 Generating Echo...' : '🚀 Generate Video'}
          </button>
        </div>

        {status && (
          <div className="status-section">
            <h2>Processing Status</h2>
            <div className="status-card">
              <div className="status-info">
                <p><strong>Job ID:</strong> {jobId}</p>
                <p><strong>Step:</strong> {status.step}</p>
                <p><strong>Progress:</strong> {status.progress}%</p>
                <p><strong>Message:</strong> {status.message}</p>
              </div>
              
              <div className="progress-bar">
                <div 
                  className="progress-fill"
                  style={{ width: `${status.progress}%` }}
                />
              </div>

              {status.status === 'completed' && (
                <button className="download-btn" onClick={downloadVideo}>
                  📥 Download Your Pixel-Echo Video
                </button>
              )}
            </div>
          </div>
        )}

        <div className="info-section">
          <h2>Features</h2>
          <div className="features-grid">
            <div className="feature-card">
              <h3>🎵 Beat-Detection</h3>
              <p>AI analysiert deine Musik für perfekte Synchronisation</p>
            </div>
            <div className="feature-card">
              <h3>✨ Glitch-Effects</h3>
              <p>Beat-synchrone surreale Pixel-Echos</p>
            </div>
            <div className="feature-card">
<h3>🎬 Ready-to-Upload</h3>
              <p>YouTube, TikTok & Instagram optimiert</p>
            </div>
            <div className="feature-card">
              <h3>🚀 3-Minute Magic</h3>
              <p>Von Upload zu fertigem Video</p>
            </div>
          </div>
        </div>
      </main>

      <footer className="App-footer">
        <p>🎨 Pixel-Echo.de | Stuttgart • AI Music Experience</p>
      </footer>
    </div>
  );
}

export default App;
