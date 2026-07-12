"""
verify_detections.py — Pre-demo footage validator for SafeSight AI.

Run this against your demo video BEFORE the live presentation to confirm
that the model actually detects people and vehicles in it.

Usage
-----
  python verify_detections.py demo_video.mp4
  python verify_detections.py demo_video.mp4 --frames 60 --conf 0.35

Why this matters
----------------
The generate_demo_video.py script produces synthetic cartoon shapes that
the model will NOT detect (it's trained on photographic imagery).  Real demo
footage — a person walking near a car in a parking lot, or any warehouse/
factory clip — is required for live detection.

This script samples the first N frames of any video and prints a frame-by-
frame summary, so you know immediately whether detections are firing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

# Ensure the project root is on sys.path (works from any working directory).
sys.path.insert(0, str(Path(__file__).parent))

from detector import SafeSightDetector, PERSON_CLASS_ID, VEHICLE_CLASS_IDS


def verify(video_path: str, num_frames: int = 30, conf: float = 0.35) -> bool:
    """
    Run SafeSightDetector on the first `num_frames` of the video.

    Prints a per-frame summary and a final pass/fail verdict.

    Returns True if at least one person OR vehicle was detected, else False.
    """
    path = Path(video_path)
    if not path.exists():
        print(f"[ERROR] File not found: {video_path}")
        return False

    print(f"\nSafeSight Detection Verifier")
    print(f"={'='*50}")
    print(f"  Video  : {video_path}")
    print(f"  Frames : first {num_frames}")
    print(f"  Conf   : >= {conf}")
    print(f"{'='*52}\n")

    detector = SafeSightDetector(model_name="fasterrcnn_mobilenet_v3", conf_threshold=conf)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return False

    total_people   = 0
    total_vehicles = 0
    frames_checked = 0

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"  [End of video at frame {i}]")
            break

        tracked, _ = detector.process_frame(frame)
        frames_checked += 1

        people   = [o for o in tracked if o.is_person]
        vehicles = [o for o in tracked if o.is_vehicle]
        total_people   += len(people)
        total_vehicles += len(vehicles)

        # Print frame summary (only frames with detections, or every 10th frame).
        if people or vehicles or (i % 10 == 0):
            p_str = ", ".join(f"person#{o.track_id}({o.confidence:.0%})" for o in people) or "-"
            v_str = ", ".join(f"{o.class_name}#{o.track_id}({o.confidence:.0%})" for o in vehicles) or "-"
            print(f"  Frame {i+1:>4}: people=[{p_str}]  vehicles=[{v_str}]")

    cap.release()

    print(f"\n{'='*52}")
    print(f"  Frames checked : {frames_checked}")
    print(f"  Total people   : {total_people}")
    print(f"  Total vehicles : {total_vehicles}")
    print()

    if total_people == 0 and total_vehicles == 0:
        print("  [FAIL] No detections found.")
        print("  This footage is likely synthetic/cartoon, or the camera angle/")
        print("  lighting makes detection unreliable.  Try:")
        print("    - Lowering --conf (e.g. --conf 0.2)")
        print("    - Using real photographic footage (person + car in a parking lot)")
        print("    - Checking that the video actually contains people/vehicles")
        return False
    else:
        print("  [PASS] Detections confirmed — footage is suitable for the live demo.")
        if total_people == 0:
            print("  WARNING: No people detected. Zone intrusion events won't fire.")
        if total_vehicles == 0:
            print("  WARNING: No vehicles detected. Proximity near-miss events won't fire.")
        return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that the model detects people/vehicles in a video."
    )
    parser.add_argument("video", help="Path to the video file to verify.")
    parser.add_argument(
        "--frames", type=int, default=30,
        help="Number of frames to sample (default: 30).",
    )
    parser.add_argument(
        "--conf", type=float, default=0.35,
        help="Confidence threshold (default: 0.35).",
    )
    args = parser.parse_args()

    ok = verify(args.video, num_frames=args.frames, conf=args.conf)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
