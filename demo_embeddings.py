#!/usr/bin/env python3
"""
Demo: Face Embeddings & Person Re-ID

This demonstrates:
1. Extracting 128×128 face chips from detected faces
2. Converting chips to 512-dimensional embedding vectors
3. Comparing embeddings to identify the same person across frames
4. Simple person tracking: "This is person #1, seen again 2 seconds later"
"""

import cv2
import numpy as np
import time
from pathlib import Path
from collections import defaultdict
from ffvideo import FaceDetector, FaceEmbedder

# Initialize detector and embedder
model_path = Path(__file__).parent / "models" / "shape_predictor_68_face_landmarks.dat"
detector = FaceDetector(str(model_path), face_model=0)

print("Loading embedder (downloading model on first run)...")
embedder = FaceEmbedder(model_name='buffalo_s')  # Small model for speed
print()

# Camera
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
print("✓ Camera ready")

print("\nControls:")
print("  'q' - quit")
print("  't' - toggle threshold display")
print("  '+'/'-' - adjust matching threshold\n")

# Person tracking state
person_embeddings = {}  # person_id → last seen embedding
person_last_seen = {}   # person_id → timestamp
next_person_id = 0
matching_threshold = 0.35  # Allows head rotation up/down without re-identifying
show_threshold = True

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
            # Run detection (0.35 = good range + speed balance)
            detector.set_image(frame, width, height, detection_scale=0.35)
            num_faces = detector.detect_faces()

            frame_display = frame.copy()

            if num_faces > 0:
                current_time = time.time()

                for i in range(num_faces):
                    # Get bounding box
                    x1, y1, x2, y2 = detector.get_face_box(i)

                    # Extract face chip
                    chip = detector.get_face_chip(i)
                    if chip is None:
                        continue

                    # Get embedding for this face
                    embedding = embedder.get_embedding(chip)

                    # Try to match with known people
                    matched_person_id = None
                    min_distance = float('inf')

                    for person_id, stored_embedding in person_embeddings.items():
                        dist = embedder.distance(embedding, stored_embedding)
                        if dist < min_distance:
                            min_distance = dist
                            if dist < matching_threshold:
                                matched_person_id = person_id

                    # Assign to person
                    if matched_person_id is not None:
                        # Seen this person before
                        person_id = matched_person_id
                        # Blend in new embedding (learn their angles/distances)
                        # Higher weight on history (0.8) makes it harder to be fooled by pose variations
                        old_emb = person_embeddings[person_id]
                        person_embeddings[person_id] = 0.8 * old_emb + 0.2 * embedding
                        person_last_seen[person_id] = current_time
                        match_text = f"Person #{person_id} (match: {min_distance:.3f})"
                        color = (0, 255, 0)  # Green = known person
                    else:
                        # New person
                        person_id = next_person_id
                        next_person_id += 1
                        person_embeddings[person_id] = embedding
                        person_last_seen[person_id] = current_time
                        match_text = f"Person #{person_id} (NEW)"
                        color = (0, 165, 255)  # Orange = new person

                    # Draw bounding box
                    cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame_display, match_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )

                # Prune old people (not seen for 30 seconds)
                for person_id in list(person_last_seen.keys()):
                    if current_time - person_last_seen[person_id] > 30:
                        del person_embeddings[person_id]
                        del person_last_seen[person_id]

            # Display info
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            info = f"Frame: {frame_count} | Faces: {num_faces} | People tracked: {len(person_embeddings)} | FPS: {fps:.1f}"
            cv2.putText(
                frame_display, info, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
            )

            if show_threshold:
                threshold_text = f"Matching threshold: {matching_threshold:.2f}"
                cv2.putText(
                    frame_display, threshold_text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
                )

            cv2.imshow("Face Embeddings & Person Re-ID", frame_display)

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"\nQuitting... (processed {frame_count} frames, tracked {len(person_embeddings)} unique people)")
                break
            elif key == ord('t'):
                show_threshold = not show_threshold
            elif key == ord('+'):
                matching_threshold = min(1.0, matching_threshold + 0.05)
                print(f"Threshold: {matching_threshold:.2f}")
            elif key == ord('-'):
                matching_threshold = max(0.0, matching_threshold - 0.05)
                print(f"Threshold: {matching_threshold:.2f}")

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
