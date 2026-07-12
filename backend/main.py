import asyncio
import cv2
import json
import threading
import time
from typing import Optional
from fastapi import FastAPI, BackgroundTasks, Response, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import StreamingResponse
import shutil
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import existing logic (ensure paths are correct since this is in backend/)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import SafeSightDetector
from events import process_events, ProximityTracker, load_events
from zones import HazardZone, check_zone_violations, draw_zones, default_demo_zones
from notifications import send_webhook_alert
from events import (
    draw_proximity_alert, draw_zone_violation_alert, draw_trajectory_prediction,
    predict_positions, _compute_velocity
)

app = FastAPI(title="SafeSight AI Backend")

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
class AppState:
    def __init__(self):
        self.running = False
        self.video_source = "Upload a video"
        self.rtsp_url = ""
        self.video_path = "_uploaded_video.mp4"
        self.cap: Optional[cv2.VideoCapture] = None
        self.detector: Optional[SafeSightDetector] = None
        self.proximity_tracker = ProximityTracker()
        self.zones = []
        self.current_frame_jpg = None
        self.webhook_url = ""
        self.conf_thresh = 0.4
        self.model_name = "yolov8n.pt"
        self.lock = threading.Lock()

state = AppState()

# Config Model
class ConfigUpdate(BaseModel):
    video_source: Optional[str] = None
    rtsp_url: Optional[str] = None
    webhook_url: Optional[str] = None
    conf_thresh: Optional[float] = None
    model_name: Optional[str] = None

def processing_loop():
    while True:
        if not state.running:
            time.sleep(0.1)
            continue
            
        with state.lock:
            if state.cap is None or not state.cap.isOpened():
                state.running = False
                continue
                
            ret, frame = state.cap.read()
            if not ret:
                # End of video or stream drop
                state.running = False
                continue

            # Run Detection
            if state.detector is None:
                state.detector = SafeSightDetector(model_name=state.model_name, conf_threshold=state.conf_thresh)
            
            # Predict and draw
            # Note: For simplicity we draw directly on frame
            tracked, annotated = state.detector.process_frame(frame)

            fh, fw = annotated.shape[:2]
            
            # Set up zones if empty
            if not state.zones:
                state.zones = default_demo_zones(fw, fh)

            active_zones = check_zone_violations(tracked, state.zones)
            new_events, current_close_pairs = process_events(tracked, active_zones, state.proximity_tracker)

            # Webhook
            for evt in new_events:
                if state.webhook_url:
                    send_webhook_alert(evt.event_type, evt.details, state.webhook_url)

            # Draw Zones & Alerts
            annotated = draw_zones(annotated, state.zones)
            for pair in current_close_pairs:
                annotated = draw_proximity_alert(annotated, pair.obj1, pair.obj2)
            
            by_id = {obj.track_id: obj for obj in tracked}
            for evt in new_events:
                if evt.event_type == "ZONE_INTRUSION" and evt.track_id in by_id:
                    obj = by_id[evt.track_id]
                    annotated = draw_zone_violation_alert(annotated, obj)

            # Convert to JPEG
            ret, buffer = cv2.imencode('.jpg', annotated)
            if ret:
                state.current_frame_jpg = buffer.tobytes()

        # Simple pacing to not melt CPU
        time.sleep(0.03)

# Start background thread
thread = threading.Thread(target=processing_loop, daemon=True)
thread.start()

@app.post("/api/start")
def start_processing():
    with state.lock:
        if state.cap is not None:
            state.cap.release()
        
        if state.video_source == "Webcam (Live)":
            import os
            if os.name == 'nt':
                state.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            else:
                state.cap = cv2.VideoCapture(0)
        elif state.video_source == "IP Camera (RTSP)":
            state.cap = cv2.VideoCapture(state.rtsp_url)
        else:
            state.cap = cv2.VideoCapture(state.video_path)
            
        if not state.cap.isOpened():
            return {"status": "error", "message": "Failed to open video source"}
        
        state.running = True
        return {"status": "success"}

@app.post("/api/stop")
def stop_processing():
    with state.lock:
        state.running = False
        if state.cap is not None:
            state.cap.release()
            state.cap = None
        return {"status": "success"}

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    with open(state.video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"status": "success", "filename": file.filename}

@app.post("/api/config")
def update_config(config: ConfigUpdate):
    with state.lock:
        if config.video_source is not None: state.video_source = config.video_source
        if config.rtsp_url is not None: state.rtsp_url = config.rtsp_url
        if config.webhook_url is not None: state.webhook_url = config.webhook_url
        if config.conf_thresh is not None: state.conf_thresh = config.conf_thresh
        if config.model_name is not None: 
            state.model_name = config.model_name
            state.detector = None # Force reload
    return {"status": "success"}

@app.get("/api/events")
def get_events():
    events = load_events()
    return {"events": events}

def generate_video_stream():
    while True:
        if state.current_frame_jpg is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + state.current_frame_jpg + b'\r\n')
        time.sleep(0.03)

@app.get("/api/video_feed")
def video_feed():
    return StreamingResponse(generate_video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")
