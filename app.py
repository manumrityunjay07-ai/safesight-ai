"""
app.py — SafeSight AI Streamlit Dashboard
==========================================

Entry point for the SafeSight AI Industrial Near-Miss Detection System.

Layout
------
  Sidebar   : video source, confidence slider, zone config, start/stop controls.
  Tab 1     : Live video feed with all overlays (bounding boxes, tracking IDs,
              hazard zones, near-miss alerts).
  Tab 2     : Event log table — filterable by event type.
  Tab 3     : Near-miss heatmap (scatter plot of incident locations).
  Tab 4     : System statistics and tuning reference.

KEY DESIGN — one frame per Streamlit rerun
------------------------------------------
The video loop processes exactly ONE frame per script execution then calls
st.rerun() to hand control back to Streamlit.  This keeps every sidebar widget
(Stop button, sliders, etc.) fully responsive throughout a run.  The
VideoCapture object is persisted in st.session_state so it isn't reopened on
every rerun, which would restart the video from frame 0 each time.

Run
---
  streamlit run app.py
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import PIL.Image
import plotly.graph_objects as go
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import supervision as svt

# Local modules.
from detector import SafeSightDetector
from events import (
    CONSECUTIVE_FRAMES_REQUIRED,
    PREDICTION_HORIZON_FRAMES,
    PROXIMITY_THRESHOLD_PX,
    ProximityTracker,
    clear_events,
    draw_proximity_alert,
    draw_trajectory_prediction,
    draw_zone_violation_alert,
    load_events,
    process_events,
    _compute_velocity,
    predict_positions,
)
from zones import HazardZone, check_zone_violations, default_demo_zones, draw_zones

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SafeSight AI — Near-Miss Detection",
    page_icon="🔶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark industrial aesthetic.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Hide Streamlit Toolbar (Fork/GitHub) and Footer */
    header[data-testid="stHeader"] {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
    footer {visibility: hidden;}
    .viewerBadge_container__1QSob {display: none !important;}
    .viewerBadge_link__1S137 {display: none !important;}
    
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1117 50%, #111827 100%);
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #334155;
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 { color: #f97316; }

    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="stMetricValue"] { color: #f97316; font-weight: 700; font-size: 2rem; }
    [data-testid="stMetricLabel"] { color: #94a3b8; font-size: 0.8rem; }

    .alert-high {
        background: linear-gradient(90deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05));
        border-left: 4px solid #ef4444; border-radius: 8px;
        padding: 10px 14px; margin: 6px 0; font-size: 0.88rem;
    }
    .alert-medium {
        background: linear-gradient(90deg, rgba(249,115,22,0.15), rgba(249,115,22,0.05));
        border-left: 4px solid #f97316; border-radius: 8px;
        padding: 10px 14px; margin: 6px 0; font-size: 0.88rem;
    }
    .section-header {
        font-size: 0.75rem; font-weight: 600; letter-spacing: 0.12em;
        text-transform: uppercase; color: #64748b;
        margin: 20px 0 8px; padding-bottom: 4px; border-bottom: 1px solid #1e293b;
    }
    [data-testid="stTabs"] [role="tab"] { color: #94a3b8; font-weight: 600; }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: #f97316; border-bottom: 2px solid #f97316;
    }
    [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

    .status-live {
        display: inline-block; background: #16a34a; color: white;
        font-size: 0.75rem; font-weight: 700; padding: 2px 10px;
        border-radius: 999px; letter-spacing: 0.08em;
        animation: pulse-green 1.5s infinite;
    }
    @keyframes pulse-green { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }
    .status-idle {
        display: inline-block; background: #475569; color: #cbd5e1;
        font-size: 0.75rem; font-weight: 700; padding: 2px 10px;
        border-radius: 999px; letter-spacing: 0.08em;
    }
    .brand-header { text-align: center; padding: 20px 0 10px; }
    .brand-title {
        font-size: 1.6rem; font-weight: 700;
        background: linear-gradient(90deg, #f97316, #ef4444);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .brand-sub {
        font-size: 0.78rem; color: #64748b; letter-spacing: 0.08em;
        text-transform: uppercase; margin-top: 2px;
    }
    hr { border-color: #1e293b; margin: 16px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

def _init_state() -> None:
    defaults = {
        "dark_mode": True,
        "running": False,
        "frame_count": 0,
        "event_counts": {"ZONE_INTRUSION": 0, "PROXIMITY_NEAR_MISS": 0, "PREDICTED_NEAR_MISS": 0},
        "detector": None,
        "proximity_tracker": None,
        "video_path": None,
        "zones": [],
        "custom_zones": [],
        "last_frame_rgb": None,
        "recent_alerts": [],
        "fps_display": 0.0,
        # NEW: persisted VideoCapture so we don't reopen the file every rerun.
        "cap": None,
        "video_fps": 30.0,
        "total_frames": 0,
        "frame_w": 640,
        "frame_h": 480,
        # NEW: playback mode flag.
        "realtime_playback": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ---------------------------------------------------------------------------
# Helpers: release the captured video cleanly
# ---------------------------------------------------------------------------
def _release_cap() -> None:
    """Release the OpenCV VideoCapture if one is open."""
    if st.session_state.cap is not None:
        try:
            st.session_state.cap.release()
        except Exception:
            pass
        st.session_state.cap = None


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
# Streamlit Native OIDC Authentication
try:
    logged_in = st.user.is_logged_in
except AttributeError:
    # This happens when [auth] secrets are not configured in Streamlit Cloud
    st.markdown('<div style="text-align:center;margin-top:100px;font-size:3rem;">🔒</div>', unsafe_allow_html=True)
    st.markdown('<h1 style="text-align:center;">Auth Not Configured</h1>', unsafe_allow_html=True)
    st.error("Google Authentication is not yet configured in Streamlit Secrets. Please follow the instructions in the walkthrough to add your Google Client ID and Secret to your Streamlit Cloud dashboard.")
    st.stop()

if not logged_in:
    st.markdown('<div style="text-align:center;margin-top:100px;font-size:3rem;">🔶</div>', unsafe_allow_html=True)
    st.markdown('<h1 style="text-align:center;">SafeSight AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;">Please sign in with your Google account to access the dashboard.</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Sign in with Google", use_container_width=True, type="primary"):
            st.login()
    st.stop()
else:
    # Optional: Display who is logged in on the sidebar
    with st.sidebar:
        st.caption(f"Logged in as: {st.user.email}")
        if st.button("Log out"):
            st.logout()


# ---------------------------------------------------------------------------
# Theme styling
# ---------------------------------------------------------------------------
if not st.session_state.dark_mode:
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background-color: #ffffff; color: #0f172a; }
    [data-testid="stSidebar"] { background-color: #f1f5f9; color: #0f172a; }
    .section-header { color: #334155 !important; border-bottom: 2px solid #cbd5e1 !important; }
    hr { border-color: #cbd5e1 !important; }
    .status-idle { background: #e2e8f0 !important; color: #475569 !important; }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand-header">
            <div class="brand-title">🔶 SafeSight AI</div>
            <div class="brand-sub">Industrial Near-Miss Detection</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.session_state.dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    st.divider()

    # ── Status indicator ──
    if st.session_state.running:
        st.markdown('<span class="status-live">● LIVE</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-idle">◉ IDLE</span>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Video Source</div>', unsafe_allow_html=True)
    video_mode = st.radio(
        "Input mode",
        ["Upload a video", "Webcam (index 0)"],
        label_visibility="collapsed",
        disabled=st.session_state.running,   # lock while running
    )
    uploaded_file = None
    if video_mode == "Upload a video":
        uploaded_file = st.file_uploader(
            "Upload MP4 / AVI", type=["mp4", "avi", "mov"], label_visibility="collapsed",
            disabled=st.session_state.running,
        )

    st.markdown('<div class="section-header">Detection Settings</div>', unsafe_allow_html=True)
    conf_thresh = st.slider("Confidence threshold", 0.20, 0.90, 0.40, 0.05,
                             help="Minimum model confidence to accept a detection.",
                             disabled=st.session_state.running)

    # ── Model selector (Permissive BSD-3) ──
    _available_models = ["fasterrcnn_mobilenet_v3"]
    model_size = st.selectbox(
        "Torchvision Model (Free / BSD-3)",
        _available_models,
        help="This model is 100% free for commercial use (BSD-3 licensed). Weights auto-download on first use.",
        disabled=st.session_state.running,
    )

    prox_thresh = st.slider("Proximity threshold (px)", 50, 400, PROXIMITY_THRESHOLD_PX, 10,
                             help="Distance (pixels) that triggers a proximity near-miss.",
                             disabled=st.session_state.running)
    pred_horizon = st.slider("Prediction horizon (frames)", 5, 60, PREDICTION_HORIZON_FRAMES, 5,
                              help="How many frames ahead to project trajectories.",
                              disabled=st.session_state.running)

    st.markdown('<div class="section-header">Hazard Zones</div>', unsafe_allow_html=True)
    zone_mode = st.radio("Zone configuration",
                          ["Use default demo zones", "No zones (proximity only)", "Interactive drawing"],
                          label_visibility="collapsed",
                          disabled=st.session_state.running)

    # Fix #7: playback speed toggle.
    st.markdown('<div class="section-header">Playback Speed</div>', unsafe_allow_html=True)
    playback_mode = st.radio(
        "Speed mode",
        ["Real-time (match source FPS)", "Max speed (no sleep)"],
        index=0 if st.session_state.realtime_playback else 1,
        label_visibility="collapsed",
        help="'Max speed' skips the inter-frame sleep — useful on CPU where inference is slower than the source FPS.",
    )
    st.session_state.realtime_playback = (playback_mode == "Real-time (match source FPS)")

    st.divider()

    # ── Control buttons: Start / Stop / Restart ──
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    start_btn  = btn_col1.button("▶ Start",   type="primary",   width='stretch',
                                  disabled=st.session_state.running)
    stop_btn   = btn_col2.button("⏹ Stop",    width='stretch',
                                  disabled=not st.session_state.running)
    restart_btn = btn_col3.button("↺ Reset",  width='stretch')

    st.markdown('<div class="section-header">Recent Alerts</div>', unsafe_allow_html=True)
    alert_placeholder = st.empty()

    # FPS display (updates every rerun while running).
    fps_placeholder = st.empty()
    if st.session_state.fps_display > 0:
        fps_placeholder.caption(
            f"Processing: **{st.session_state.fps_display:.1f} FPS** "
            f"({'real-time' if st.session_state.realtime_playback else 'max speed'})"
        )


# ---------------------------------------------------------------------------
# Control button logic
# ---------------------------------------------------------------------------

# ── Stop ──
if stop_btn:
    st.session_state.running = False
    _release_cap()
    st.rerun()

# ── Reset ──
if restart_btn:
    st.session_state.running = False
    _release_cap()
    st.session_state.frame_count = 0
    st.session_state.event_counts = {"ZONE_INTRUSION": 0, "PROXIMITY_NEAR_MISS": 0, "PREDICTED_NEAR_MISS": 0}
    st.session_state.detector = None
    st.session_state.proximity_tracker = None
    st.session_state.recent_alerts = []
    st.session_state.last_frame_rgb = None
    st.session_state.fps_display = 0.0
    clear_events()
    st.rerun()

# ── Start ──
if start_btn:
    # Resolve video path.
    if video_mode == "Upload a video":
        if uploaded_file is None:
            st.error("Please upload a video file first.")
            st.stop()
        tmp_path = Path("_uploaded_video.mp4")
        tmp_path.write_bytes(uploaded_file.read())
        st.session_state.video_path = str(tmp_path)
    else:
        st.session_state.video_path = 0  # webcam

    # Torchvision handles its own model downloading and caching in ~/.cache/torch/hub/checkpoints
    # No need to validate if the model file exists in the local directory.

    # Open VideoCapture once here — persisted across reruns.
    _release_cap()
    cap = cv2.VideoCapture(st.session_state.video_path)
    if not cap.isOpened():
        st.error(f"Could not open video source: `{st.session_state.video_path}`")
        st.stop()

    # Probe video dimensions and FPS.
    st.session_state.video_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    st.session_state.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    st.session_state.frame_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    st.session_state.frame_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    st.session_state.cap         = cap

    # Set up hazard zones (need frame dims).
    if zone_mode == "Use default demo zones":
        st.session_state.zones = default_demo_zones(
            st.session_state.frame_w, st.session_state.frame_h
        )
    elif zone_mode == "Interactive drawing":
        st.session_state.zones = st.session_state.get("custom_zones", [])
    else:
        st.session_state.zones = []

    # Initialise detector and trackers.
    st.session_state.detector = SafeSightDetector(
        model_name=model_size,
        conf_threshold=conf_thresh,
    )
    st.session_state.proximity_tracker = ProximityTracker()
    # Fix #3: initialise zone/prediction dedup state inside session_state too.
    st.session_state.zone_fired    = set()   # (track_id, zone_name) pairs that fired
    st.session_state.pred_fired    = set()   # (pid, vid) predicted pairs that fired
    st.session_state.frame_count   = 0
    st.session_state.event_counts  = {"ZONE_INTRUSION": 0, "PROXIMITY_NEAR_MISS": 0, "PREDICTED_NEAR_MISS": 0}
    st.session_state.recent_alerts = []
    clear_events()
    st.session_state.running = True
    st.rerun()


# ---------------------------------------------------------------------------
# Main content area — layout (always rendered so placeholders exist)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
        <div style="font-size:1.5rem;font-weight:700;color:#f97316;">🔶 SafeSight AI</div>
        <div style="font-size:0.8rem;color:#64748b;border:1px solid #334155;padding:2px 10px;
                    border-radius:6px;font-family:'JetBrains Mono',monospace;">
            Industrial Near-Miss Detection System
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.running and zone_mode == "Interactive drawing":
    st.info("Draw polygons on the canvas below to define custom hazard zones. Click 'Start' in the sidebar when done.")
    
    # Try to grab the first frame of the selected video
    bg_image = None
    vp = None
    if video_mode == "Upload a video" and uploaded_file is not None:
        tmp_path = Path("_uploaded_video.mp4")
        tmp_path.write_bytes(uploaded_file.read())
        vp = str(tmp_path)
        
    if vp:
        cap = cv2.VideoCapture(vp)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                bg_image = PIL.Image.fromarray(frame_rgb)
        cap.release()

    if bg_image:
        # Scale down for drawing UI if it's too large
        canvas_w = min(bg_image.width, 800)
        canvas_h = int(bg_image.height * (canvas_w / bg_image.width))
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=2,
            stroke_color="#ff0000",
            background_image=bg_image,
            update_streamlit=True,
            height=canvas_h,
            width=canvas_w,
            drawing_mode="polygon",
            key="canvas",
        )
        
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            custom_zones = []
            for idx, obj in enumerate(objects):
                if obj["type"] == "polygon":
                    # Scale coordinates back to original video dimensions
                    scale_x = bg_image.width / canvas_w
                    scale_y = bg_image.height / canvas_h
                    pts = [(int(p[0] * scale_x), int(p[1] * scale_y)) for p in obj["path"] if len(p) >= 2 and type(p[0]) in (int, float)]
                    if len(pts) >= 3:
                        custom_zones.append(HazardZone(
                            name=f"Custom Zone {idx+1}",
                            points=pts,
                            severity="HIGH"
                        ))
            st.session_state.custom_zones = custom_zones
            if custom_zones:
                st.success(f"Parsed {len(custom_zones)} custom zone(s)!")
    else:
        st.warning("Please upload a video or use the demo video to start drawing.")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("🔴 Zone Intrusions",       st.session_state.event_counts["ZONE_INTRUSION"])
kpi2.metric("⚠️ Proximity Near-Misses", st.session_state.event_counts["PROXIMITY_NEAR_MISS"])
kpi3.metric("🔮 Predicted Near-Misses", st.session_state.event_counts["PREDICTED_NEAR_MISS"])
kpi4.metric("🎞️ Frames Processed",      st.session_state.frame_count)

st.divider()

tab_live, tab_events = st.tabs(
    ["📺 Live Feed", "📋 Event Log"]
)

with tab_live:
    feed_col, sidebar_col = st.columns([3, 1])
    with feed_col:
        frame_display = st.empty()
        status_row    = st.empty()
    with sidebar_col:
        st.markdown('<div class="section-header">Object Tracker</div>', unsafe_allow_html=True)
        tracker_info  = st.empty()
        st.markdown('<div class="section-header">Frame Info</div>', unsafe_allow_html=True)
        frame_info    = st.empty()

with tab_events:
    st.button("🔄 Refresh table", key="refresh_events")
    filter_col, _ = st.columns([2, 3])
    with filter_col:
        event_filter = st.multiselect(
            "Filter by event type",
            ["ZONE_INTRUSION", "PROXIMITY_NEAR_MISS", "PREDICTED_NEAR_MISS"],
            default=["ZONE_INTRUSION", "PROXIMITY_NEAR_MISS", "PREDICTED_NEAR_MISS"],
        )
    events_table_placeholder = st.empty()


# ---------------------------------------------------------------------------
# Render functions
# ---------------------------------------------------------------------------
# Helper: render event log table
# ---------------------------------------------------------------------------
def _render_event_table(filter_types: List[str]) -> None:
    events = load_events()
    if filter_types:
        events = [e for e in events if e.get("event_type") in filter_types]
    if not events:
        events_table_placeholder.info("No events logged yet. Start the analysis to begin.")
        return
    import pandas as pd
    df = pd.DataFrame(events)
    col_map = {
        "event_type": "Type", "timestamp": "Timestamp", "frame_number": "Frame",
        "severity": "Severity", "zone_name": "Zone", "details": "Details",
    }
    show_cols = [c for c in col_map if c in df.columns]
    df_display = df[show_cols].rename(columns=col_map)

    def _sev_style(val: str) -> str:
        if val == "HIGH":   return "background-color:rgba(239,68,68,0.2);color:#ef4444;font-weight:700"
        if val == "MEDIUM": return "background-color:rgba(249,115,22,0.2);color:#f97316;font-weight:700"
        return ""

    # pandas >= 2.1 uses .map(); .applymap() was removed in 2.2.
    styled = df_display.style.map(_sev_style, subset=["Severity"])
    events_table_placeholder.dataframe(styled, width='stretch', height=420)



# ---------------------------------------------------------------------------
# ONE-FRAME processing — runs when st.session_state.running is True.
# After processing one frame this block calls st.rerun() so Streamlit can
# check button state before the next frame — this is what makes Stop work.
# ---------------------------------------------------------------------------
if st.session_state.running and st.session_state.detector is not None:
    cap: cv2.VideoCapture = st.session_state.cap

    # Safety-check: cap may have been released by a concurrent Stop press.
    if cap is None or not cap.isOpened():
        st.session_state.running = False
        st.rerun()

    t_start = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        # Video ended — loop back to beginning.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        st.session_state.detector.reset()
        st.session_state.proximity_tracker = ProximityTracker()
        st.session_state.zone_fired  = set()
        st.session_state.pred_fired  = set()
        st.rerun()

    detector: SafeSightDetector   = st.session_state.detector
    prox_tracker: ProximityTracker = st.session_state.proximity_tracker
    zones                          = st.session_state.zones
    fw  = st.session_state.frame_w
    fh  = st.session_state.frame_h
    fn  = detector.frame_number + 1   # will be incremented inside process_frame

    # ── Detection + tracking ──
    tracked, annotated = detector.process_frame(frame)
    fn = detector.frame_number
    st.session_state.frame_count = fn

    # ── Zone violation check ──
    zone_violations  = check_zone_violations(tracked, zones)
    active_zone_names = []
    for _tid, violated_zones in zone_violations.items():
        active_zone_names.extend(z.name for z in violated_zones)

    # ── Draw hazard zones ──
    annotated = draw_zones(annotated, zones, active_zone_names=active_zone_names)

    # ── Event logic (with dedup state from session_state) ──
    new_events, current_close_pairs = process_events(
        tracked,
        zone_violations,
        prox_tracker,
        frame_number=fn,
        zone_fired=st.session_state.zone_fired,
        pred_fired=st.session_state.pred_fired,
    )
    # Persist updated fired-sets back to session_state.
    # (process_events mutates the sets in-place, but we reassign to be explicit.)
    # Nothing extra needed — sets are mutable and shared by reference.

    # ── Update KPI counters ──
    for evt in new_events:
        st.session_state.event_counts[evt.event_type] = (
            st.session_state.event_counts.get(evt.event_type, 0) + 1
        )
        icon = "🔴" if evt.severity == "HIGH" else "🟠"
        alert_str = f"{icon} {evt.details[:60]}..." if len(evt.details) > 60 else f"{icon} {evt.details}"
        st.session_state.recent_alerts.insert(0, alert_str)
        st.session_state.recent_alerts = st.session_state.recent_alerts[:8]

    # ── Draw zone violation alerts (flashing box) ──
    by_id = {obj.track_id: obj for obj in tracked}
    for tid, violated_zones in zone_violations.items():
        obj = by_id.get(tid)
        if obj:
            for zone in violated_zones:
                annotated = draw_zone_violation_alert(annotated, obj.bbox, tid, zone.name, fn)

    # ── Draw proximity alerts ──
    centroids = {obj.track_id: obj.centroid for obj in tracked}
    for pid, vid in current_close_pairs:
        if pid in centroids and vid in centroids:
            annotated = draw_proximity_alert(annotated, centroids[pid], centroids[vid], "PROXIMITY", fn)

    # ── Draw predicted trajectories ──
    for obj in tracked:
        vel = _compute_velocity(obj.history)
        if vel is not None:
            future = predict_positions(obj.centroid, vel, 20)
            color  = (0, 200, 255) if obj.is_person else (255, 180, 0)
            annotated = draw_trajectory_prediction(annotated, future, color)

    # ── HUD overlay ──
    elapsed = time.perf_counter() - t_start
    fps = 1.0 / max(elapsed, 1e-6)
    st.session_state.fps_display = fps
    total = st.session_state.total_frames
    speed_label = "RT" if st.session_state.realtime_playback else "MAX"
    hud = (
        f"SafeSight AI  |  Frame {fn}/{total}  |  {fps:.1f} FPS [{speed_label}]  |  "
        f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC"
    )
    cv2.putText(annotated, hud, (10, fh - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Display frame ──
    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    st.session_state.last_frame_rgb = rgb
    frame_display.image(rgb, channels="RGB", use_container_width=True)

    # ── Tracker info panel ──
    persons  = [o for o in tracked if o.is_person]
    vehicles = [o for o in tracked if o.is_vehicle]
    tracker_info.markdown(
        f"**People:** {len(persons)}  \n**Vehicles:** {len(vehicles)}  \n"
        f"**Zone violations:** {len(active_zone_names)}  \n**Close pairs:** {len(current_close_pairs)}"
    )
    frame_info.markdown(f"`Frame: {fn}`  \n`FPS: {fps:.1f}`  \n`Res: {fw}×{fh}`")

    # ── Sidebar alert list ──
    if st.session_state.recent_alerts:
        alerts_html = "".join(
            f'<div class="alert-high">{a}</div>' if "🔴" in a
            else f'<div class="alert-medium">{a}</div>'
            for a in st.session_state.recent_alerts
        )
        alert_placeholder.markdown(alerts_html, unsafe_allow_html=True)

    # ── Render event tabs every frame ──
    with tab_events:
        _render_event_table(event_filter)

    # ── Pacing: only sleep in real-time mode ──
    if st.session_state.realtime_playback:
        frame_delay = 1.0 / st.session_state.video_fps
        sleep_time  = frame_delay - (time.perf_counter() - t_start)
        if sleep_time > 0:
            time.sleep(sleep_time)

    # ── Yield back to Streamlit — this is what makes Stop responsive ──
    st.rerun()

else:
    # ── Idle state — show welcome screen or last frame ──
    with tab_live:
        if st.session_state.last_frame_rgb is not None:
            frame_display.image(st.session_state.last_frame_rgb, channels="RGB",
                                use_container_width=True)
        else:
            frame_display.markdown(
                """
                <div style="
                    height:420px; display:flex; flex-direction:column;
                    align-items:center; justify-content:center;
                    background:linear-gradient(135deg,#0f172a,#1e293b);
                    border:2px dashed #334155; border-radius:16px;
                    color:#475569; text-align:center; gap:16px;
                ">
                    <div style="font-size:4rem;">🔶</div>
                    <div style="font-size:1.4rem;font-weight:700;color:#64748b;">SafeSight AI</div>
                    <div style="font-size:0.9rem;color:#475569;max-width:380px;line-height:1.6;">
                        Select a video source in the sidebar<br>
                        and click <strong style="color:#f97316;">▶ Start</strong> to begin analysis.
                    </div>
                    <div style="font-size:0.75rem;color:#334155;font-family:monospace;">
                        Computer Vision + ByteTrack + Zone Detection
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    fw_ = st.session_state.frame_w
    fh_ = st.session_state.frame_h
    _render_event_table(["ZONE_INTRUSION", "PROXIMITY_NEAR_MISS", "PREDICTED_NEAR_MISS"])
