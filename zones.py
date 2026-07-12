"""
zones.py — Hazard zone definitions and point-in-polygon logic for SafeSight AI.

A HazardZone is simply a named polygon defined in pixel coordinates.
We use OpenCV's cv2.pointPolygonTest to check whether a tracked object's
foot-point falls inside the polygon — this is more accurate than centroid
for tall objects like people (their feet are on the ground, not their centre).

Zone colours and opacity are chosen to be visible on dark warehouse footage
without fully obscuring the underlying image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class HazardZone:
    """
    A named polygonal hazard zone in pixel coordinates.

    Attributes
    ----------
    name : str
        Human-readable label (e.g. "Restricted Zone A").
    points : list of (x, y) tuples
        Polygon vertices in pixel space, ordered (winding doesn't matter for
        cv2.pointPolygonTest).
    color : (B, G, R) tuple
        OpenCV colour used for the filled overlay and label.
    severity : str
        "HIGH" or "MEDIUM" — used to triage event log entries.
    """

    name: str
    points: List[Tuple[int, int]]
    color: Tuple[int, int, int] = (0, 0, 220)   # default red
    severity: str = "HIGH"

    @property
    def np_points(self) -> np.ndarray:
        """Polygon vertices as an (N,1,2) int32 array for OpenCV."""
        return np.array(self.points, dtype=np.int32).reshape((-1, 1, 2))

    def contains_point(self, pt: Tuple[int, int]) -> bool:
        """
        Return True if (x, y) is inside (or on the boundary of) this zone.

        cv2.pointPolygonTest returns:
          +1  → inside
           0  → on the edge
          -1  → outside
        We treat edge as inside so that objects at the exact boundary are
        flagged (conservative safety decision).
        """
        result = cv2.pointPolygonTest(self.np_points, (float(pt[0]), float(pt[1])), False)
        return result >= 0


def default_demo_zones(frame_w: int, frame_h: int) -> List[HazardZone]:
    """
    Return two hardcoded demo zones sized as fractions of the frame dimensions.

    These are calibrated for typical warehouse-footage aspect ratios.
    Adjust the fractions (0.0–1.0) to match your specific demo video.

    Zone A — a rectangular "Restricted Zone" on the left side.
    Zone B — a trapezoidal "Loading Dock" in the bottom-right corner.
    """
    w, h = frame_w, frame_h

    zone_a = HazardZone(
        name="Restricted Zone A",
        points=[
            (int(w * 0.03), int(h * 0.30)),
            (int(w * 0.30), int(h * 0.30)),
            (int(w * 0.30), int(h * 0.85)),
            (int(w * 0.03), int(h * 0.85)),
        ],
        color=(0, 0, 200),   # red
        severity="HIGH",
    )

    zone_b = HazardZone(
        name="Loading Dock",
        points=[
            (int(w * 0.60), int(h * 0.50)),
            (int(w * 0.95), int(h * 0.40)),
            (int(w * 0.95), int(h * 0.95)),
            (int(w * 0.55), int(h * 0.95)),
        ],
        color=(0, 120, 255),  # orange
        severity="MEDIUM",
    )

    return [zone_a, zone_b]


def draw_zones(
    frame: np.ndarray,
    zones: List[HazardZone],
    alpha: float = 0.25,
    active_zone_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Draw semi-transparent filled polygons for each zone onto the frame.

    Parameters
    ----------
    frame : BGR image (will be modified in-place on a copy).
    zones : list of HazardZone objects.
    alpha : fill opacity (0=invisible, 1=opaque). Default 0.25.
    active_zone_names : list of zone names currently being violated —
        these are drawn with higher opacity and a pulsing border to grab
        attention.

    Returns
    -------
    Annotated copy of the frame.
    """
    active_zone_names = active_zone_names or []
    overlay = frame.copy()
    result = frame.copy()

    for zone in zones:
        pts = zone.np_points
        is_active = zone.name in active_zone_names

        # Fill the zone polygon.
        fill_alpha = 0.45 if is_active else alpha
        cv2.fillPoly(overlay, [pts], zone.color)
        cv2.addWeighted(overlay, fill_alpha, result, 1 - fill_alpha, 0, result)
        overlay = result.copy()   # reset overlay for next zone

        # Draw border — thicker / brighter when active.
        border_color = (255, 255, 255) if is_active else zone.color
        border_thickness = 3 if is_active else 2
        cv2.polylines(result, [pts], isClosed=True, color=border_color, thickness=border_thickness)

        # Label the zone near its top-left corner.
        label_pt = (zone.points[0][0] + 6, zone.points[0][1] + 22)
        # Use ASCII prefixes — cv2.putText cannot render Unicode/emoji characters.
        severity_prefix = "[!!]" if zone.severity == "HIGH" else "[!]"
        label = f"{severity_prefix} {zone.name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(result,
                      (label_pt[0] - 2, label_pt[1] - th - 4),
                      (label_pt[0] + tw + 2, label_pt[1] + 4),
                      zone.color, -1)
        cv2.putText(result, label, label_pt,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    return result


def check_zone_violations(
    tracked_objects,   # List[TrackedObject] — avoid circular import
    zones: List[HazardZone],
) -> Dict[int, List[HazardZone]]:
    """
    For every tracked object, return the list of zones it currently violates.

    We use the *foot-point* (bottom-centre of bbox) for people, and the
    centroid for vehicles — a vehicle's body spans a large area so centroid
    is a reasonable proxy for "is this machine in the danger zone".

    Returns
    -------
    dict mapping track_id → list of violated HazardZone objects.
    """
    violations: Dict[int, List[HazardZone]] = {}

    for obj in tracked_objects:
        # Choose the check point: foot for people, centroid for vehicles.
        check_pt = obj.foot_point if obj.is_person else obj.centroid
        violated = [z for z in zones if z.contains_point(check_pt)]
        if violated:
            violations[obj.track_id] = violated

    return violations
