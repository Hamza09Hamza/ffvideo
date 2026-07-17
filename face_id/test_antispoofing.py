#!/usr/bin/env python3
"""
Test MiniFASNet anti-spoofing on live camera feed.

Shows real faces in GREEN, spoofed faces in RED.
Displays liveness score and confidence for each face.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from collections import deque
import sys

# Add parent directory to path so we can import ffvideo
sys.path.insert(0, str(Path(__file__).parent.parent))

from ffvideo import FaceDetector
from face_id.models.antispoofing import MiniFASNetAntiSpoof

# Initialize detector
print("Loading face detector...")
model_path = Path(__file__).parent.parent / "models" / "shape_predictor_68_face_landmarks.dat"
detector = FaceDetector(str(model_path), face_model=0)

# Initialize anti-spoofing
print("Loading anti-spoofing model...")
antispoof = MiniFASNetAntiSpoof()

# Camera setup
print("Opening camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("✓ Camera opened")

print("Warming up camera...")
for i in range(20):
    ret, _ = cap.read()
    time.sleep(0.1)
print("✓ Camera ready\n")

print("Controls:")
print("  'q' - quit")
print("  's' - toggle confidence display")
print("  't' - toggle threshold (0.3, 0.5, 0.7)\n")

# Thresholds for real/spoof decision
threshold = 0.5
show_confidence = True
available_thresholds = [0.3, 0.5, 0.7]
threshold_idx = 1

# Temporal aggregation: collect scores over a sliding window of frames
# Decision is based on the AVERAGE score, not a single frame
# This prevents a single bad frame from flipping the result
WINDOW_SIZE = 15  # number of frames to average over
POSITION_THRESHOLD = 50  # pixels; if face moves more than this, reset history

# Per-face tracking: face_id → (score_history, last_position)
face_histories = {}
face_id_counter = [0]  # use list for mutable counter in nested scope

frame_count = 0
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        frame_count += 1
        height, width = frame.shape[:2]

        try:
            # Detect faces
            detector.set_image(frame, width, height, detection_scale=0.35)
            num_faces = detector.detect_faces()

            frame_display = frame.copy()

            if num_faces == 0:
                # No faces detected — clear old histories
                face_histories.clear()

            if num_faces > 0:
                # Match detected faces to tracked faces by proximity
                matched_face_ids = set()

                for i in range(num_faces):
                    # Get bounding box
                    x1, y1, x2, y2 = detector.get_face_box(i)
                    face_center = ((x1 + x2) // 2, (y1 + y2) // 2)

                    # Find closest tracked face
                    closest_face_id = None
                    closest_distance = float('inf')

                    for fid, (history, last_pos) in face_histories.items():
                        dist = ((face_center[0] - last_pos[0])**2 + (face_center[1] - last_pos[1])**2)**0.5
                        if dist < POSITION_THRESHOLD and dist < closest_distance:
                            closest_face_id = fid
                            closest_distance = dist

                    # Assign or create face ID
                    if closest_face_id is not None:
                        face_id = closest_face_id
                        matched_face_ids.add(face_id)
                    else:
                        # New face
                        face_id = face_id_counter[0]
                        face_id_counter[0] += 1
                        face_histories[face_id] = (deque(maxlen=WINDOW_SIZE), face_center)
                        matched_face_ids.add(face_id)

                    # Run anti-spoofing on raw frame + bbox
                    result = antispoof.get_liveness_score(frame, x1, y1, x2, y2)

                    # Update this face's history
                    history, _ = face_histories[face_id]
                    history.append(result['score'])
                    face_histories[face_id] = (history, face_center)

                    # Decision based on AVERAGE over the window, not single frame
                    avg_score = sum(history) / len(history) if len(history) > 0 else 0
                    is_real = avg_score > threshold
                    warming_up = len(history) < WINDOW_SIZE

                    # Color: green = real, red = spoof, yellow = warming up
                    if warming_up:
                        color = (0, 255, 255)
                        decision = f"WARMING UP ({len(history)}/{WINDOW_SIZE})"
                    else:
                        color = (0, 255, 0) if is_real else (0, 0, 255)
                        decision = "REAL ✓" if is_real else "SPOOF ✗"

                    # Draw bounding box
                    cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 2)

                    # Draw decision and averaged score
                    score_text = f"{decision} (avg: {avg_score:.3f})"
                    cv2.putText(
                        frame_display, score_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                    )

                    # Draw per-frame and window scores if enabled
                    if show_confidence:
                        conf_text = f"Frame: {result['score']:.3f} | V2: {result['v2_score']:.3f} V1: {result['v1_score']:.3f}"
                        cv2.putText(
                            frame_display, conf_text, (x1, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                        )

                # Remove face histories that weren't matched (face disappeared)
                for fid in list(face_histories.keys()):
                    if fid not in matched_face_ids:
                        del face_histories[fid]

            # Display stats
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            info = f"Frame: {frame_count} | Faces: {num_faces} | FPS: {fps:.1f} | Threshold: {threshold:.1f}"
            cv2.putText(
                frame_display, info, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

            cv2.imshow("Anti-Spoofing Test", frame_display)

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"\nQuitting... (processed {frame_count} frames)")
                break
            elif key == ord('s'):
                show_confidence = not show_confidence
                print(f"Confidence display: {'ON' if show_confidence else 'OFF'}")
            elif key == ord('t'):
                threshold_idx = (threshold_idx + 1) % len(available_thresholds)
                threshold = available_thresholds[threshold_idx]
                print(f"Threshold changed to: {threshold:.1f}")

        except Exception as e:
            print(f"Error in frame {frame_count}: {e}")
            import traceback
            traceback.print_exc()
            break

except KeyboardInterrupt:
    print("\nInterrupted")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Done")
