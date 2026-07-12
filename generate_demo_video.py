"""
generate_demo_video.py — Creates a synthetic demo video for SafeSight AI.

This script generates a ~20-second MP4 file that simulates an industrial
warehouse scene with:
  - A "person" (green rectangle) walking across the frame and entering a zone.
  - A "vehicle" (orange rectangle) moving toward the person (forklift proxy).

This is enough to reliably trigger both detection types WITHOUT needing actual
YOLOv8 detections — it's purely for pipeline/UI testing.

For the live demo with REAL detection, replace demo_video.mp4 with actual
warehouse/street footage containing people and vehicles.

Usage
-----
  python generate_demo_video.py
"""

import cv2
import numpy as np
import math

FRAME_W, FRAME_H = 1280, 720
FPS = 30
DURATION_S = 25
TOTAL_FRAMES = FPS * DURATION_S
OUT_FILE = "demo_video.mp4"


def draw_background(frame: np.ndarray, fn: int) -> None:
    """Draw a simple industrial floor grid background."""
    frame[:] = (25, 30, 35)  # dark concrete grey (BGR)

    # Grid lines — floor tiles.
    for x in range(0, FRAME_W, 120):
        cv2.line(frame, (x, 0), (x, FRAME_H), (40, 45, 50), 1)
    for y in range(0, FRAME_H, 120):
        cv2.line(frame, (0, y), (FRAME_W, y), (40, 45, 50), 1)

    # Perspective vanishing lines toward top-centre.
    vp = (FRAME_W // 2, FRAME_H // 3)
    for x in range(0, FRAME_W + 1, 160):
        cv2.line(frame, (x, FRAME_H), vp, (50, 55, 60), 1)


def draw_hazard_zone(frame: np.ndarray, pts: np.ndarray, name: str, active: bool) -> None:
    """Draw a semi-transparent hazard zone."""
    overlay = frame.copy()
    color = (0, 0, 180) if not active else (0, 0, 255)
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    cv2.polylines(frame, [pts], True, (0, 0, 220) if not active else (255, 255, 255), 2)
    label_pt = (pts[0][0] + 8, pts[0][1] + 26)
    cv2.putText(frame, f"⛔ {name}", label_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)


def draw_person(frame: np.ndarray, cx: int, cy: int, tid: int = 1) -> tuple:
    """Draw a simplified person silhouette (box + head circle)."""
    w, h = 50, 120
    x1, y1, x2, y2 = cx - w // 2, cy - h, cx + w // 2, cy
    # Body.
    cv2.rectangle(frame, (x1, y1 + 30), (x2, y2), (0, 200, 80), -1)
    cv2.rectangle(frame, (x1, y1 + 30), (x2, y2), (0, 255, 100), 2)
    # Head.
    cv2.circle(frame, (cx, y1 + 18), 18, (0, 200, 80), -1)
    cv2.circle(frame, (cx, y1 + 18), 18, (0, 255, 100), 2)
    # Label.
    cv2.putText(frame, f"#P{tid} person", (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1, cv2.LINE_AA)
    return (x1, y1, x2, y2)


def draw_vehicle(frame: np.ndarray, cx: int, cy: int, tid: int = 2) -> tuple:
    """Draw a forklift-like box."""
    w, h = 140, 90
    x1, y1, x2, y2 = cx - w // 2, cy - h, cx + w // 2, cy
    # Body.
    cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 100, 220), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 140, 255), 3)
    # Wheels.
    for wx in [x1 + 20, x2 - 20]:
        cv2.circle(frame, (wx, y2), 14, (60, 60, 60), -1)
        cv2.circle(frame, (wx, y2), 14, (100, 100, 100), 2)
    # Fork prongs.
    cv2.rectangle(frame, (x2, y1 + 20), (x2 + 40, y1 + 32), (200, 200, 200), -1)
    cv2.rectangle(frame, (x2, y1 + 50), (x2 + 40, y1 + 62), (200, 200, 200), -1)
    # Label.
    cv2.putText(frame, f"#V{tid} vehicle", (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 140, 255), 1, cv2.LINE_AA)
    return (x1, y1, x2, y2)


def draw_alert(frame: np.ndarray, msg: str, fn: int) -> None:
    pulse = (fn // 10) % 2 == 0
    color = (0, 0, 255) if pulse else (0, 80, 200)
    cv2.rectangle(frame, (0, 0), (FRAME_W, FRAME_H), color, 8)
    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    x = (FRAME_W - tw) // 2
    cv2.rectangle(frame, (x - 10, 40), (x + tw + 10, 40 + th + 14), color, -1)
    cv2.putText(frame, msg, (x, 40 + th + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)


def main():
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_FILE, fourcc, FPS, (FRAME_W, FRAME_H))

    # Zone polygon: left third of frame, middle half.
    zone_pts = np.array([
        [40, FRAME_H // 3],
        [FRAME_W // 3, FRAME_H // 3],
        [FRAME_W // 3, FRAME_H - 80],
        [40, FRAME_H - 80],
    ], dtype=np.int32)

    print(f"Generating {TOTAL_FRAMES} frames -> {OUT_FILE}")

    for fn in range(TOTAL_FRAMES):
        frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        draw_background(frame, fn)

        t = fn / TOTAL_FRAMES   # normalised time 0→1
        t_s = fn / FPS           # seconds elapsed

        # ── Person path ──
        # Phase 1 (0–8s): walks right-to-left, entering zone.
        # Phase 2 (8–16s): lingers in zone area.
        # Phase 3 (16–25s): walks toward vehicle (proximity near-miss).
        if t_s < 8:
            px = int(FRAME_W * 0.85 - (FRAME_W * 0.6) * (t_s / 8))
            py = int(FRAME_H * 0.75)
        elif t_s < 16:
            px = int(FRAME_W * 0.18 + 20 * math.sin(t_s * 1.2))
            py = int(FRAME_H * 0.72 + 10 * math.cos(t_s * 0.8))
        else:
            px = int(FRAME_W * 0.18 + (FRAME_W * 0.55) * ((t_s - 16) / 9))
            py = int(FRAME_H * 0.72)

        # ── Vehicle path ──
        # Starts at right, moves left slowly.
        vx = int(FRAME_W * 0.92 - (FRAME_W * 0.45) * min(t_s / 20, 1.0))
        vy = int(FRAME_H * 0.72)

        # Check zone intrusion (person's foot at py).
        person_in_zone = cv2.pointPolygonTest(zone_pts, (float(px), float(py)), False) >= 0
        dist_pv = math.sqrt((px - vx) ** 2 + (py - vy) ** 2)
        proximity_alert = dist_pv < 200

        # ── Draw ──
        draw_hazard_zone(frame, zone_pts, "Restricted Zone A", person_in_zone)

        # Second zone (loading dock, lower right).
        zone2_pts = np.array([
            [int(FRAME_W * 0.60), int(FRAME_H * 0.50)],
            [int(FRAME_W * 0.95), int(FRAME_H * 0.40)],
            [int(FRAME_W * 0.95), int(FRAME_H * 0.95)],
            [int(FRAME_W * 0.55), int(FRAME_H * 0.95)],
        ], dtype=np.int32)
        draw_hazard_zone(frame, zone2_pts, "Loading Dock", False)

        # Draw objects.
        pbbox = draw_person(frame, px, py, tid=1)
        vbbox = draw_vehicle(frame, vx, vy, tid=2)

        # Proximity connecting line.
        if proximity_alert:
            pulse = (fn // 15) % 2 == 0
            lc = (0, 0, 255) if pulse else (0, 100, 255)
            cv2.line(frame, (px, py), (vx, vy), lc, 3)
            mid = ((px + vx) // 2, (py + vy) // 2)
            cv2.putText(frame, f"⚠ NEAR-MISS {int(dist_pv)}px", (mid[0] - 80, mid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Zone breach box.
        if person_in_zone:
            draw_alert(frame, "⛔  ZONE BREACH DETECTED", fn)

        # HUD.
        cv2.putText(frame, f"SafeSight AI Demo  |  Frame {fn}/{TOTAL_FRAMES}  |  t={t_s:.1f}s",
                    (10, FRAME_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        writer.write(frame)

        if fn % 60 == 0:
            print(f"  {fn}/{TOTAL_FRAMES} frames written...")

    writer.release()
    print(f"[OK] Demo video saved: {OUT_FILE}  ({TOTAL_FRAMES} frames @ {FPS}fps)")


if __name__ == "__main__":
    main()
