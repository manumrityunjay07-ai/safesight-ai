import { useState, useEffect, useRef } from 'react'
import './index.css'

function App() {
  const [activeTab, setActiveTab] = useState('live')
  const [isRunning, setIsRunning] = useState(false)
  const [events, setEvents] = useState([])
  const [config, setConfig] = useState({
    video_source: 'Upload a video',
    rtsp_url: 'rtsp://admin:pass@192.168.1.100:554/stream',
    webhook_url: '',
    conf_thresh: 0.4,
    model_name: 'yolov8n.pt'
  })

  const API_URL = 'https://automobiles-buddy-distance-istanbul.trycloudflare.com'

  // Start the video feed
  const startStream = async () => {
    try {
      const res = await fetch(`${API_URL}/api/start`, { method: 'POST' })
      if (res.ok) setIsRunning(true)
    } catch (err) {
      console.error(err)
    }
  }

  const stopStream = async () => {
    try {
      const res = await fetch(`${API_URL}/api/stop`, { method: 'POST' })
      if (res.ok) setIsRunning(false)
    } catch (err) {
      console.error(err)
    }
  }

  // Fetch events periodically
  useEffect(() => {
    let interval;
    if (activeTab === 'events' || isRunning) {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`${API_URL}/api/events`)
          const data = await res.json()
          if (data.events) setEvents(data.events)
        } catch (err) {
          console.error(err)
        }
      }, 2000)
    }
    return () => clearInterval(interval)
  }, [activeTab, isRunning])

  const handleConfigChange = async (key, value) => {
    const newConfig = { ...config, [key]: value }
    setConfig(newConfig)
    try {
      await fetch(`${API_URL}/api/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ [key]: value })
      })
    } catch (err) {
      console.error(err)
    }
  }

  return (
    <div className="app-container">
      {/* Sidebar */}
      <div className="sidebar glass">
        <div className="brand">
          <div className="brand-title">🔶 SafeSight AI</div>
          <div className="brand-sub">Industrial Near-Miss Detection</div>
        </div>
        
        {/* Status Indicator */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{
            width: '12px', height: '12px', borderRadius: '50%',
            backgroundColor: isRunning ? 'var(--success-color)' : 'var(--text-secondary)'
          }}></div>
          <span style={{ fontWeight: 500, color: isRunning ? 'var(--success-color)' : 'var(--text-secondary)' }}>
            {isRunning ? 'LIVE' : 'IDLE'}
          </span>
        </div>
        
        <div className="control-group">
          <label>Video Source</label>
          <select 
            className="input-field" 
            value={config.video_source} 
            onChange={(e) => handleConfigChange('video_source', e.target.value)}
            disabled={isRunning}
          >
            <option>Upload a video</option>
            <option>Webcam (Live)</option>
            <option>IP Camera (RTSP)</option>
          </select>
        </div>

        {config.video_source === 'IP Camera (RTSP)' && (
          <div className="control-group">
            <label>RTSP Stream URL</label>
            <input 
              type="text" 
              className="input-field" 
              value={config.rtsp_url} 
              onChange={(e) => handleConfigChange('rtsp_url', e.target.value)}
              disabled={isRunning}
            />
          </div>
        )}

        <div className="control-group">
          <label>YOLO Model (Fast / Edge)</label>
          <select 
            className="input-field" 
            value={config.model_name} 
            onChange={(e) => handleConfigChange('model_name', e.target.value)}
            disabled={isRunning}
          >
            <option value="yolov8n.pt">yolov8n.pt</option>
            <option value="yolov10n.pt">yolov10n.pt</option>
          </select>
        </div>

        <div className="control-group">
          <label>Slack/Teams Webhook URL</label>
          <input 
            type="text" 
            className="input-field" 
            placeholder="https://hooks.slack.com/..." 
            value={config.webhook_url} 
            onChange={(e) => handleConfigChange('webhook_url', e.target.value)}
          />
        </div>

        <div style={{ display: 'flex', gap: '8px', marginTop: '16px' }}>
          <button 
            className="btn btn-primary" 
            style={{ flex: 1 }} 
            onClick={startStream} 
            disabled={isRunning}
          >
            ▶ Start
          </button>
          <button 
            className="btn btn-danger" 
            style={{ flex: 1 }} 
            onClick={stopStream} 
            disabled={!isRunning}
          >
            ⏹ Stop
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        <div className="tabs">
          <button 
            className={`tab ${activeTab === 'live' ? 'active' : ''}`} 
            onClick={() => setActiveTab('live')}
          >
            📺 Live Feed
          </button>
          <button 
            className={`tab ${activeTab === 'events' ? 'active' : ''}`} 
            onClick={() => setActiveTab('events')}
          >
            📋 Event Log
          </button>
        </div>

        {activeTab === 'live' && (
          <div className="video-container glass">
            {isRunning ? (
              <img 
                src={`${API_URL}/api/video_feed?t=${Date.now()}`} 
                alt="Live AI Feed" 
                className="video-feed" 
                onError={(e) => console.error("Error loading video feed", e)}
              />
            ) : (
              <div className="video-placeholder">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M23 7l-7 5 7 5V7z"></path>
                  <rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect>
                </svg>
                <p>Stream Offline. Click Start in the sidebar.</p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'events' && (
          <div className="events-table-wrapper glass">
            <table className="events-table">
              <thead>
                <tr>
                  <th>Time (UTC)</th>
                  <th>Event Type</th>
                  <th>Severity</th>
                  <th>Details</th>
                </tr>
              </thead>
              <tbody>
                {events.length === 0 ? (
                  <tr>
                    <td colSpan="4" style={{ textAlign: 'center', color: 'var(--text-secondary)' }}>No events logged yet.</td>
                  </tr>
                ) : (
                  events.slice().reverse().map((evt, idx) => (
                    <tr key={idx}>
                      <td>{evt.timestamp}</td>
                      <td>{evt.event_type}</td>
                      <td>
                        <span className={`badge ${evt.severity === 'HIGH' ? 'badge-high' : 'badge-medium'}`}>
                          {evt.severity}
                        </span>
                      </td>
                      <td>{evt.details}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
