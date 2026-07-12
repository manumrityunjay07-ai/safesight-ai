# SafeSight AI — Industrial Near-Miss Detection

A computer-vision prototype that detects, tracks, and flags near-miss safety
events in industrial/warehouse footage — restricted-zone intrusions and
person-vehicle proximity events — using pretrained YOLOv8 + ByteTrack, with
a live Streamlit dashboard.

## Architecture

```
 Video Input (file / webcam / RTSP)
          │
          ▼
 detector.py  ──  YOLOv8 (pretrained) detection
          │        + ByteTrack (via `supervision`) for persistent IDs
          │        + Rolling centroid history per track (last 10 positions)
          ▼
 zones.py  ──  Hazard zone polygons + cv2.pointPolygonTest
          │     + Semi-transparent overlay rendering
          ▼
 events.py ──  Zone intrusion / proximity near-miss / predicted near-miss
          │     + ProximityTracker (consecutive-frame debounce)
          │     + Linear trajectory prediction (velocity vectors)
          │     + JSONL event logging
          ▼
 app.py  ──  Streamlit dashboard: live overlay feed, event counters,
              event table, near-miss location heatmap
```

## File Structure

```
SafeSight AI — Industrial Near-Miss Detection System/
├── app.py                  # Streamlit dashboard entry point
├── detector.py             # YOLOv8 + ByteTrack wrapper
├── zones.py                # Hazard zone definition and polygon logic
├── events.py               # Near-miss / proximity / prediction + logging
├── generate_demo_video.py  # Script to create a synthetic demo video
├── requirements.txt        # Python dependencies
├── demo_video.mp4          # Sample footage (add your own or generate)
└── events.jsonl            # Auto-generated event log (runtime)
```

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

First run will auto-download the YOLOv8n pretrained weights (~6 MB).

## Demo Footage — IMPORTANT

> **`generate_demo_video.py` produces synthetic cartoon shapes for
> pipeline/UI testing ONLY.  YOLOv8 is trained on photographic imagery
> and will detect ZERO objects in synthetic output.**
> Use real footage for the live detection demo.

### Sourcing real footage (< 5 minutes)

You need a short clip (~30–60 seconds) that contains **at least one person
and one car, truck, or bus** visible together.  Options:

| Source | Where to get it |
|--------|-----------------|
| Phone recording | Record yourself or a friend walking near a parked car in a parking lot |
| Free stock video | [Pexels.com](https://www.pexels.com/search/videos/warehouse/) — search "warehouse" or "parking lot" |
| YouTube download | Any dashcam/parking lot clip — use `yt-dlp` to download |

Save the file as **`demo_video.mp4`** in the project folder.

### Verify detections before the demo

Always run this before a live presentation:

```bash
python verify_detections.py demo_video.mp4
# Optional: check more frames or lower confidence threshold
python verify_detections.py demo_video.mp4 --frames 60 --conf 0.30
```

Expected output on good footage:
```
  Frame   8: people=[person#1(87%)]  vehicles=[car#2(91%)]
  ...
  [PASS] Detections confirmed — footage is suitable for the live demo.
```

## Generating a Synthetic Test Video (pipeline testing only)

If you just want to test the UI/pipeline logic without real footage:

```bash
python generate_demo_video.py
```

This creates `demo_video.mp4` (25 seconds) that exercises the zone overlay,
zone-breach flash, and proximity-line rendering.  **YOLO will not detect
objects in it** — the KPI counters and event log will remain at zero unless
you use real footage.

## Running the Demo

```bash
streamlit run app.py
```

In the browser:

1. **Sidebar → Video Source**: Select *"Use demo_video.mp4"* (or upload your own).
2. **Sidebar → Detection Settings**: Adjust confidence / proximity thresholds.
3. **Sidebar → Hazard Zones**: Select *"Use default demo zones"*.
4. **Click ▶ Start** — live feed appears with overlays.
5. **Click ⏹ Stop** at any time to pause analysis (sidebar stays fully responsive).

### What you'll see

| Overlay | Meaning |
|---------|---------|
| Green box + trail | Tracked person with centroid history |
| Orange box | Tracked vehicle (forklift stand-in) |
| Semi-transparent red/orange polygon | Hazard zone |
| Flashing red box + "ZONE BREACH!" | Person entered restricted zone |
| Dashed red line + "⚠ NEAR-MISS!" | Person & vehicle too close |
| Dotted cyan/yellow dots | Predicted trajectory (next 20 frames) |

## What to Point Out to Judges

- **Not just detection** — the pipeline chains:
  `detection → tracking (stable IDs) → spatial reasoning (zone polygons) → event prediction (linear trajectory flags a converging path *before* it becomes a near-miss)`.
- **Zero custom training** — pretrained YOLOv8n + ByteTrack means the whole
  system was buildable in a day, but the architecture is model-agnostic:
  swap in a fine-tuned model later for forklifts / PPE / hard-hats.
- **Consecutive-frame debounce** — proximity alerts require N consecutive frames
  of close proximity, eliminating false positives from detector jitter.
- **Trajectory prediction** — velocity vectors estimated by weighted linear
  regression over the last 5 centroid positions; projected paths fire a
  `PREDICTED_NEAR_MISS` 1 second before a collision would occur.
- **Commercial path** — same pipeline works on CCTV/RTSP streams; the event
  log (JSONL/SQLite-ready) is the seed of an analytics dashboard, and hazard
  zones + thresholds are fully configurable per site.

## Tuning

| Parameter | File | Description |
|-----------|------|-------------|
| `PROXIMITY_THRESHOLD_PX` | `events.py` | Pixel distance for near-miss trigger |
| `CONSECUTIVE_FRAMES_REQUIRED` | `events.py` | Frames before proximity alert fires |
| `PREDICTION_HORIZON_FRAMES` | `events.py` | How far ahead to project trajectories |
| `PREDICTION_THRESHOLD_PX` | `events.py` | Predicted collision distance threshold |
| `default_demo_zones()` | `zones.py` | Hardcoded zone fractions of frame size |

- Swap `yolov8n.pt` → `yolov8s.pt` in the **Sidebar → YOLOv8 model** dropdown for better
  accuracy if you have GPU headroom.

## Non-Goals (out of scope for the 24h build)

- No custom model training/fine-tuning.
- No real CCTV/IoT integration (simulated via local video files).
- No user authentication or multi-tenant support.
- No mobile app.
