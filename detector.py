"""
detector.py — Torchvision + ByteTrack wrapper for SafeSight AI.

Responsibilities:
  - Load the pretrained Torchvision Faster R-CNN model (BSD-3 licensed).
  - Run inference on each frame and filter to safety-relevant COCO classes.
  - Hand detections off to supervision's ByteTrack for stable, persistent IDs
    across occlusion and re-entry.
  - Maintain a rolling centroid history (last N positions) per track ID so that
    events.py can compute velocity vectors for trajectory prediction.

IMPORTANT — Windows file-lock fix:
  _MODEL_CACHE below is a module-level dict that keeps exactly one
  model instance alive for the lifetime of the Python process.
  Subsequent calls reuse the cached object — no file I/O, no lock contention.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Module-level model cache — one model instance per model name per process.
# This is the key fix for the Windows WinError 32 file-lock issue.
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict = {}

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import torchvision
import numpy as np
import supervision as sv
import warnings

# Suppress supervision ByteTrack deprecation warning to keep logs clean
warnings.filterwarnings("ignore", category=FutureWarning, module="supervision")
import cv2
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)

# ---------------------------------------------------------------------------
# COCO class IDs that are relevant to industrial safety
# Note: Torchvision COCO classes are 1-indexed (0 is background).
# 1=person, 3=car, 6=bus, 8=truck
# ---------------------------------------------------------------------------
PERSON_CLASS_ID = 1
VEHICLE_CLASS_IDS = {3, 6, 8}
RELEVANT_CLASS_IDS = {PERSON_CLASS_ID} | VEHICLE_CLASS_IDS

CLASS_NAMES = {
    1: "person",
    3: "car",
    6: "bus",
    8: "truck"
}

# How many historical centroid positions we keep per tracked object.
HISTORY_LEN = 10


@dataclass
class TrackedObject:
    """Represents a single tracked object in one frame."""

    track_id: int
    class_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2) in pixels
    centroid: Tuple[int, int]          # (cx, cy) — foot-point would be (cx, y2)
    confidence: float
    # Rolling history of centroids — oldest first, newest last.
    history: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def is_person(self) -> bool:
        return self.class_id == PERSON_CLASS_ID

    @property
    def is_vehicle(self) -> bool:
        return self.class_id in VEHICLE_CLASS_IDS

    @property
    def foot_point(self) -> Tuple[int, int]:
        """Bottom-centre of the bounding box — better proxy for ground position."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, y2)


class SafeSightDetector:
    """
    Wraps Torchvision detection + ByteTrack to produce a list of TrackedObject
    instances for every frame passed through `process_frame`.
    """

    def __init__(
        self,
        model_name: str = "fasterrcnn_mobilenet_v3_large_fpn",
        conf_threshold: float = 0.4,
        device: str = "",          # "" → auto (CUDA if available, else CPU)
    ) -> None:
        
        if device == "":
            self.device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        else:
            self.device = device
        print(f"[SafeSight] Inference device: {self.device}")
            
        # Reuse cached model if already loaded
        if model_name not in _MODEL_CACHE:
            print(f"[SafeSight] Loading free BSD-licensed model: {model_name}")
            self.model = torchvision.models.detection.__dict__[model_name](weights="DEFAULT")
            self.model.to(self.device)
            self.model.eval()
            _MODEL_CACHE[model_name] = self.model
        else:
            print(f"[SafeSight] Reusing cached model: {model_name}")
            
        self.model = _MODEL_CACHE[model_name]
        self.conf_threshold = conf_threshold

        # ByteTrack tracker — supervision's implementation is stable across
        # a wide range of frame rates and occlusion durations.
        self.tracker = sv.ByteTrack()

        # Centroid history keyed by track_id → deque of (cx, cy) tuples.
        self._history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=HISTORY_LEN)
        )

        # Frame counter — used for timing / logging.
        self.frame_number: int = 0

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[List[TrackedObject], np.ndarray]:
        """
        Run detection + tracking on a single BGR frame.

        Returns
        -------
        tracked : list of TrackedObject
            All currently tracked objects after filtering.
        annotated_frame : np.ndarray
            A copy of the frame with raw detection boxes drawn (zones and
            near-miss overlays are added later by app.py / events.py).
        """
        self.frame_number += 1

        # --- Torchvision inference --------------------------------------------
        # Torchvision expects RGB tensor [C, H, W] normalized to [0, 1]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        input_tensor = input_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
        
        with torch.no_grad():
            preds = self.model(input_tensor)[0]

        # Convert to supervision Detections so we can feed them straight into ByteTrack.
        boxes = preds['boxes'].cpu().numpy()
        scores = preds['scores'].cpu().numpy()
        labels = preds['labels'].cpu().numpy()
        
        if len(boxes) > 0:
            detections = sv.Detections(
                xyxy=boxes,
                confidence=scores,
                class_id=labels
            )
            # Filter to our classes and confidence threshold
            mask = np.isin(detections.class_id, list(RELEVANT_CLASS_IDS)) & (detections.confidence >= self.conf_threshold)
            detections = detections[mask]
        else:
            detections = sv.Detections.empty()

        # --- ByteTrack update -------------------------------------------------
        # ByteTrack assigns / maintains persistent track IDs.
        tracked_sv = self.tracker.update_with_detections(detections)

        # --- Build TrackedObject list -----------------------------------------
        tracked: List[TrackedObject] = []

        for i in range(len(tracked_sv)):
            bbox = tracked_sv.xyxy[i].astype(int)
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            tid = int(tracked_sv.tracker_id[i])
            cid = int(tracked_sv.class_id[i])
            conf = float(tracked_sv.confidence[i]) if tracked_sv.confidence is not None else 0.0

            # Update centroid history for this track.
            self._history[tid].append((cx, cy))

            obj = TrackedObject(
                track_id=tid,
                class_id=cid,
                class_name=CLASS_NAMES.get(cid, str(cid)),
                bbox=(x1, y1, x2, y2),
                centroid=(cx, cy),
                confidence=conf,
                history=list(self._history[tid]),
            )
            tracked.append(obj)

        # --- Annotate frame (lightweight, just boxes + IDs) -------------------
        annotated = frame.copy()
        for obj in tracked:
            color = (0, 255, 0) if obj.is_person else (255, 165, 0)
            x1, y1, x2, y2 = obj.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"#{obj.track_id} {obj.class_name} {obj.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                annotated, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA
            )
            # Draw centroid trail.
            pts = obj.history
            for j in range(1, len(pts)):
                alpha = j / len(pts)
                trail_color = tuple(int(c * alpha) for c in color)
                cv2.line(annotated, pts[j - 1], pts[j], trail_color, 2)

        return tracked, annotated

    def reset(self) -> None:
        """Reset tracker state between video clips."""
        self.tracker = sv.ByteTrack()
        self._history.clear()
        self.frame_number = 0
