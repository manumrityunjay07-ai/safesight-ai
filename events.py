"""
events.py — Near-miss detection, trajectory prediction, and event logging.

Three classes of safety events are detected:

1. ZONE_INTRUSION — a person's foot-point enters a hazard zone polygon.
2. PROXIMITY_NEAR_MISS — a person and a vehicle remain within
   PROXIMITY_THRESHOLD_PX pixels of each other for at least
   CONSECUTIVE_FRAMES_REQUIRED frames.  Using consecutive frames rather
   than a single-frame trigger reduces false positives from detection noise.
3. PREDICTED_NEAR_MISS — linear trajectory extrapolation:  if two objects'
   projected paths will bring them within PREDICTION_THRESHOLD_PX pixels
   within PREDICTION_HORIZON_FRAMES frames, we pre-warn now.

All events are written as newline-delimited JSON (JSONL) to EVENTS_LOG_FILE.
The dashboard reads this file to populate the event table and heatmap.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Tunable constants — referenced in README.md "Tuning" section.
# ---------------------------------------------------------------------------

# Pixels between centroids that constitute a dangerous proximity.
# At typical 720p warehouse-cam resolution ~150px ≈ 1–2 metres.
PROXIMITY_THRESHOLD_PX: int = 160

# How many consecutive frames the proximity must hold before we fire an alert.
# This filters out detector jitter / transient misclassifications.
CONSECUTIVE_FRAMES_REQUIRED: int = 4

# Trajectory prediction: how many frames ahead we project paths.
PREDICTION_HORIZON_FRAMES: int = 30

# If predicted positions come within this distance, fire a predicted near-miss.
PREDICTION_THRESHOLD_PX: int = 120

# File where all events are appended (SQLite Database).
EVENTS_DB_FILE: str = "events.db"

# ---------------------------------------------------------------------------
# Event data model
# ---------------------------------------------------------------------------

EVENT_TYPES = {
    "ZONE_INTRUSION": "Zone Intrusion",
    "PROXIMITY_NEAR_MISS": "Proximity Near-Miss",
    "PREDICTED_NEAR_MISS": "Predicted Near-Miss",
}

SEVERITY_MAP = {
    "ZONE_INTRUSION": "HIGH",
    "PROXIMITY_NEAR_MISS": "HIGH",
    "PREDICTED_NEAR_MISS": "MEDIUM",
}


@dataclass
class SafetyEvent:
    event_type: str                  # one of EVENT_TYPES keys
    timestamp: str                   # ISO-8601
    frame_number: int
    track_ids: List[int]             # involved track IDs
    zone_name: Optional[str]         # relevant for ZONE_INTRUSION
    location_x: int                  # pixel x of incident centroid
    location_y: int                  # pixel y of incident centroid
    severity: str                    # HIGH / MEDIUM
    details: str                     # human-readable description


# ---------------------------------------------------------------------------
# Proximity tracker — maintains consecutive-frame counters per object pair
# ---------------------------------------------------------------------------

class ProximityTracker:
    """
    Tracks how many consecutive frames each (person_id, vehicle_id) pair
    has been within PROXIMITY_THRESHOLD_PX of each other.

    Using consecutive frames rather than a single-frame check makes the
    alert far more reliable: it tolerates one or two frames where the
    detector momentarily loses one object.
    """

    def __init__(self) -> None:
        # Maps (person_tid, vehicle_tid) → consecutive close-frame count
        self._counters: Dict[Tuple[int, int], int] = defaultdict(int)
        # Tracks which pairs have already fired to avoid repeated logging
        self._fired: set = set()

    def update(
        self,
        person_ids: List[int],
        vehicle_ids: List[int],
        centroids: Dict[int, Tuple[int, int]],
        threshold_px: int = PROXIMITY_THRESHOLD_PX,
    ) -> List[Tuple[int, int]]:
        """
        Update counters and return (person_id, vehicle_id) pairs that have
        just crossed the consecutive-frame threshold for the first time.
        """
        # Compute distances for all person-vehicle combinations in this frame.
        close_pairs: set = set()
        for pid in person_ids:
            for vid in vehicle_ids:
                if pid not in centroids or vid not in centroids:
                    continue
                dist = _euclidean(centroids[pid], centroids[vid])
                if dist < threshold_px:
                    close_pairs.add((pid, vid))

        # Increment counters for currently-close pairs; reset others.
        all_known = set(self._counters.keys()) | close_pairs
        for pair in all_known:
            if pair in close_pairs:
                self._counters[pair] += 1
            else:
                # Reset counter — the pair separated.
                self._counters[pair] = 0
                self._fired.discard(pair)  # allow re-fire if they get close again

        # Collect pairs that just crossed the threshold and haven't fired yet.
        newly_triggered = []
        for pair in close_pairs:
            if (
                self._counters[pair] >= CONSECUTIVE_FRAMES_REQUIRED
                and pair not in self._fired
            ):
                newly_triggered.append(pair)
                self._fired.add(pair)

        return newly_triggered

    def get_current_close_pairs(self) -> List[Tuple[int, int]]:
        """Return all pairs currently in a close state (counter > 0)."""
        return [p for p, c in self._counters.items() if c > 0]


# ---------------------------------------------------------------------------
# Trajectory prediction
# ---------------------------------------------------------------------------

def _compute_velocity(history: List[Tuple[int, int]]) -> Optional[Tuple[float, float]]:
    """
    Estimate velocity (vx, vy) in pixels/frame from the last few positions.

    We weight later position deltas more heavily so that recent movement
    dominates over older, potentially stale positions.

    Fix: the denominator is now sum(weights[1:]) — exactly the weights used
    in the numerator.  The original code divided by sum(weights) which
    included weight[0] (unused), systematically underestimating velocity.

    Sanity check: for a history of [(0,0),(5,0),(10,0),(15,0),(20,0)]
    (constant +5 px/frame), this returns (5.0, 0.0).
    """
    if len(history) < 3:
        return None
    recent = history[-5:]   # at most last 5 positions
    n = len(recent)
    xs = [p[0] for p in recent]
    ys = [p[1] for p in recent]
    # Weights for each consecutive delta (higher index = more recent = higher weight).
    weights = list(range(1, n + 1))   # [1, 2, 3, 4, 5] for n=5
    delta_weights = weights[1:]        # [2, 3, 4, 5] — one per consecutive pair
    denom = sum(delta_weights)         # FIX: was sum(weights) — now matches numerator
    vx = sum(w * (xs[i] - xs[i-1]) for i, w in zip(range(1, n), delta_weights)) / denom
    vy = sum(w * (ys[i] - ys[i-1]) for i, w in zip(range(1, n), delta_weights)) / denom
    return (vx, vy)


def predict_positions(
    centroid: Tuple[int, int],
    velocity: Tuple[float, float],
    horizon: int = PREDICTION_HORIZON_FRAMES,
) -> List[Tuple[int, int]]:
    """
    Project the object's future positions assuming constant velocity.

    Returns a list of (x, y) positions for each of the next `horizon` frames.
    This is a deliberate simplification — constant-velocity works well for
    slow-moving warehouse vehicles and people over short horizons.
    """
    cx, cy = centroid
    vx, vy = velocity
    return [
        (int(cx + vx * t), int(cy + vy * t))
        for t in range(1, horizon + 1)
    ]


def check_predicted_near_miss(
    obj_a,   # TrackedObject
    obj_b,   # TrackedObject
    threshold_px: int = PREDICTION_THRESHOLD_PX,
    horizon: int = PREDICTION_HORIZON_FRAMES,
) -> Optional[int]:
    """
    Check whether two objects' projected paths will converge within `horizon`
    frames.

    Returns the frame offset (1..horizon) at which the closest approach occurs
    if it is below threshold_px, else None.

    The key insight: we don't need to wait for a near-miss to happen; we can
    warn operators *before* it occurs.  A 30-frame horizon at 30fps = 1 second
    of warning time — enough for a human operator to hit an emergency stop.
    """
    vel_a = _compute_velocity(obj_a.history)
    vel_b = _compute_velocity(obj_b.history)
    if vel_a is None or vel_b is None:
        return None

    future_a = predict_positions(obj_a.centroid, vel_a, horizon)
    future_b = predict_positions(obj_b.centroid, vel_b, horizon)

    min_dist = float("inf")
    closest_frame = None
    for t, (pa, pb) in enumerate(zip(future_a, future_b), start=1):
        d = _euclidean(pa, pb)
        if d < min_dist:
            min_dist = d
            closest_frame = t

    if min_dist < threshold_px:
        return closest_frame
    return None


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------

def draw_proximity_alert(
    frame: np.ndarray,
    centroid_a: Tuple[int, int],
    centroid_b: Tuple[int, int],
    alert_type: str = "PROXIMITY",
    frame_number: int = 0,
) -> np.ndarray:
    """
    Draw a connecting line and alert labels between two objects in danger.

    The line pulses between two colours based on frame parity to create a
    visual "alarm" effect without requiring any external animation state.
    """
    # Pulsing color: alternate between bright red and orange every 15 frames.
    pulse = (frame_number // 15) % 2 == 0
    line_color = (0, 0, 255) if pulse else (0, 100, 255)

    # Thick dashed line connecting the two centroids.
    _draw_dashed_line(frame, centroid_a, centroid_b, line_color, thickness=3, dash_len=15)

    # Distance label at the midpoint.
    mid = ((centroid_a[0] + centroid_b[0]) // 2, (centroid_a[1] + centroid_b[1]) // 2)
    dist = int(_euclidean(centroid_a, centroid_b))
    if alert_type == "PROXIMITY":
        label = f"[!] NEAR-MISS! {dist}px"
    else:
        label = f"[!] PRED. NEAR-MISS {dist}px"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(frame,
                  (mid[0] - tw // 2 - 4, mid[1] - th - 6),
                  (mid[0] + tw // 2 + 4, mid[1] + 4),
                  line_color, -1)
    cv2.putText(frame, label, (mid[0] - tw // 2, mid[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def draw_zone_violation_alert(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    track_id: int,
    zone_name: str,
    frame_number: int = 0,
) -> np.ndarray:
    """
    Draw a flashing red bounding box around an object that has entered a zone.
    """
    pulse = (frame_number // 10) % 2 == 0
    color = (0, 0, 255) if pulse else (0, 50, 200)
    x1, y1, x2, y2 = bbox
    # Double-thickness alert box.
    cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), color, 4)
    label = f"ZONE BREACH! #{track_id} -> {zone_name}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def draw_trajectory_prediction(
    frame: np.ndarray,
    future_positions: List[Tuple[int, int]],
    color: Tuple[int, int, int] = (255, 200, 0),
) -> np.ndarray:
    """Draw a dotted line showing predicted future path of an object."""
    for i, pt in enumerate(future_positions[::3]):  # every 3rd point for clarity
        radius = max(2, 5 - i)
        cv2.circle(frame, pt, radius, color, -1)
    return frame


# ---------------------------------------------------------------------------
# Event logging
# ---------------------------------------------------------------------------

def init_db(db_file: str = EVENTS_DB_FILE) -> None:
    with sqlite3.connect(db_file) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT,
                timestamp TEXT,
                frame_number INTEGER,
                track_ids TEXT,
                zone_name TEXT,
                location_x INTEGER,
                location_y INTEGER,
                severity TEXT,
                details TEXT
            )
        ''')


def log_event(event: SafetyEvent, db_file: str = EVENTS_DB_FILE) -> None:
    """
    Append a SafetyEvent to the SQLite event log.
    """
    init_db(db_file)
    with sqlite3.connect(db_file) as conn:
        conn.execute('''
            INSERT INTO events (event_type, timestamp, frame_number, track_ids, zone_name, location_x, location_y, severity, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event.event_type,
            event.timestamp,
            event.frame_number,
            json.dumps(event.track_ids),
            event.zone_name,
            event.location_x,
            event.location_y,
            event.severity,
            event.details
        ))


def load_events(db_file: str = EVENTS_DB_FILE) -> List[dict]:
    """Load all logged events from the SQLite database, newest first."""
    init_db(db_file)
    events = []
    with sqlite3.connect(db_file) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute('SELECT * FROM events ORDER BY id DESC')
        for row in cur:
            d = dict(row)
            d['track_ids'] = json.loads(d['track_ids'])
            events.append(d)
    return events


def clear_events(db_file: str = EVENTS_DB_FILE) -> None:
    """Truncate the event log (called on Restart)."""
    init_db(db_file)
    with sqlite3.connect(db_file) as conn:
        conn.execute('DELETE FROM events')


# ---------------------------------------------------------------------------
# Main per-frame event processing function
# ---------------------------------------------------------------------------

def process_events(
    tracked_objects,            # List[TrackedObject]
    zone_violations: dict,      # {track_id: [HazardZone, ...]}
    proximity_tracker: ProximityTracker,
    frame_number: int,
    db_file: str = EVENTS_DB_FILE,
    # Fix #3 — dedup sets, mutated in-place and persisted in session_state.
    # Keys: (track_id, zone_name) for zone intrusions;
    #       (person_id, vehicle_id) for predicted near-misses.
    # A pair is removed from the set when the condition clears, allowing
    # re-logging if the same object re-enters a zone or paths reconverge.
    zone_fired: Optional[set] = None,
    pred_fired: Optional[set] = None,
) -> Tuple[List[SafetyEvent], List[Tuple[int, int]]]:
    """
    Run all event checks for the current frame and log any new events.

    Returns
    -------
    new_events : list of SafetyEvent fired this frame.
    current_close_pairs : list of (person_tid, vehicle_tid) currently close
        (for drawing proximity lines every frame, even between logged events).
    """
    new_events: List[SafetyEvent] = []
    now = datetime.now(timezone.utc).isoformat() + "Z"

    # Initialise dedup sets if caller didn't provide them (backward compat).
    if zone_fired is None:
        zone_fired = set()
    if pred_fired is None:
        pred_fired = set()

    # Build lookup dicts for O(1) access.
    by_id = {obj.track_id: obj for obj in tracked_objects}
    centroids = {obj.track_id: obj.centroid for obj in tracked_objects}
    person_ids = [obj.track_id for obj in tracked_objects if obj.is_person]
    vehicle_ids = [obj.track_id for obj in tracked_objects if obj.is_vehicle]

    # 1. Zone intrusion events — debounced: one log entry per continuous stay.
    # Collect the (tid, zone_name) pairs that are CURRENTLY active.
    currently_in_zone: set = set()
    for tid, zones in zone_violations.items():
        obj = by_id.get(tid)
        if obj is None:
            continue
        for zone in zones:
            key = (tid, zone.name)
            currently_in_zone.add(key)
            if key in zone_fired:
                continue   # already logged this continuous intrusion — skip
            # First frame this object is inside this zone: log it.
            evt = SafetyEvent(
                event_type="ZONE_INTRUSION",
                timestamp=now,
                frame_number=frame_number,
                track_ids=[tid],
                zone_name=zone.name,
                location_x=obj.centroid[0],
                location_y=obj.centroid[1],
                severity=zone.severity,
                details=f"Object #{tid} ({obj.class_name}) entered '{zone.name}'",
            )
            new_events.append(evt)
            log_event(evt, db_file)
            zone_fired.add(key)

    # Clear fired state for any (tid, zone) pair that is no longer active,
    # so re-entry after leaving will log a new event.
    for key in list(zone_fired):
        if key not in currently_in_zone:
            zone_fired.discard(key)

    # 2. Proximity near-miss events (consecutive-frame threshold).
    newly_triggered = proximity_tracker.update(person_ids, vehicle_ids, centroids)
    for (pid, vid) in newly_triggered:
        p_obj = by_id.get(pid)
        v_obj = by_id.get(vid)
        if p_obj is None or v_obj is None:
            continue
        loc = (
            (p_obj.centroid[0] + v_obj.centroid[0]) // 2,
            (p_obj.centroid[1] + v_obj.centroid[1]) // 2,
        )
        dist = int(_euclidean(p_obj.centroid, v_obj.centroid))
        evt = SafetyEvent(
            event_type="PROXIMITY_NEAR_MISS",
            timestamp=now,
            frame_number=frame_number,
            track_ids=[pid, vid],
            zone_name=None,
            location_x=loc[0],
            location_y=loc[1],
            severity="HIGH",
            details=(
                f"Person #{pid} within {dist}px of {v_obj.class_name} #{vid} "
                f"for {CONSECUTIVE_FRAMES_REQUIRED}+ consecutive frames"
            ),
        )
        new_events.append(evt)
        log_event(evt, db_file)

    # 3. Predicted near-miss — debounced: one log entry per convergence episode.
    current_close = {(p, v) for p, v in proximity_tracker.get_current_close_pairs()}
    # Clear pred_fired for pairs that have separated AND are no longer converging
    # (will be re-evaluated below — we track which keys remain active this frame).
    pred_active_this_frame: set = set()
    for pid in person_ids:
        for vid in vehicle_ids:
            if (pid, vid) in current_close:
                continue   # already in a live near-miss; no need to predict
            p_obj = by_id.get(pid)
            v_obj = by_id.get(vid)
            if p_obj is None or v_obj is None:
                continue
            frames_until = check_predicted_near_miss(p_obj, v_obj)
            if frames_until is not None:
                key = (pid, vid)
                pred_active_this_frame.add(key)
                if key in pred_fired:
                    continue   # already logged this convergence episode — skip
                loc = (
                    (p_obj.centroid[0] + v_obj.centroid[0]) // 2,
                    (p_obj.centroid[1] + v_obj.centroid[1]) // 2,
                )
                evt = SafetyEvent(
                    event_type="PREDICTED_NEAR_MISS",
                    timestamp=now,
                    frame_number=frame_number,
                    track_ids=[pid, vid],
                    zone_name=None,
                    location_x=loc[0],
                    location_y=loc[1],
                    severity="MEDIUM",
                    details=(
                        f"Predicted collision: Person #{pid} & {v_obj.class_name} #{vid} "
                        f"in ~{frames_until} frames"
                    ),
                )
                new_events.append(evt)
                log_event(evt, db_file)
                pred_fired.add(key)

    # Clear fired state for pairs no longer converging so they can re-trigger.
    for key in list(pred_fired):
        if key not in pred_active_this_frame:
            pred_fired.discard(key)

    return new_events, proximity_tracker.get_current_close_pairs()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _euclidean(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _draw_dashed_line(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    dash_len: int = 10,
) -> None:
    """Draw a dashed line between two points."""
    dist = _euclidean(pt1, pt2)
    if dist == 0:
        return
    dx = (pt2[0] - pt1[0]) / dist
    dy = (pt2[1] - pt1[1]) / dist
    n_dashes = int(dist // (dash_len * 2))
    for i in range(n_dashes + 1):
        start = (
            int(pt1[0] + dx * i * dash_len * 2),
            int(pt1[1] + dy * i * dash_len * 2),
        )
        end = (
            int(pt1[0] + dx * (i * dash_len * 2 + dash_len)),
            int(pt1[1] + dy * (i * dash_len * 2 + dash_len)),
        )
        cv2.line(img, start, end, color, thickness)
