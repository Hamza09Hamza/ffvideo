"""
Per-connection verification state: face tracking + temporal score aggregation.

This is the same sliding-window logic proven in test_antispoofing.py,
refactored so a WebSocket handler can drive it frame by frame instead of a
local `while True` camera loop.

Deepfake detection (XceptionNet) is intentionally NOT wired in here: for this
app's threat model (an admin logging in from their own device/camera), the
realistic attack is a presentation attack — photo, video replay, mask —
which anti-spoofing already covers. XceptionNet also currently has random,
untrained weights (see face_id/models/deepfake.py), so including it would
add noise, not signal. The code lives on in face_id/models/deepfake.py and
_deepfake_architecture.py if it's revisited later with real trained weights.
"""

import time
from collections import deque

WINDOW_SIZE = 15          # frames to average scores over
# Pixels; if a face moves further than this between frames, treat it as a new
# face and reset its score history. Needs to be generous because detection
# currently runs at ~5fps (CPU-bound dlib HOG, see the `detect` timing) —
# ordinary head movement between two frames 200ms apart can easily be
# 100+ px, so a tight threshold here causes constant, spurious resets.
POSITION_THRESHOLD = 150


def _ms(t0, t1):
    return round((t1 - t0) * 1000, 1)


class FaceTracker:
    """
    Matches detected face boxes across frames to stable face_ids by spatial
    proximity, and keeps a bounded score history per face_id per metric
    (e.g. "antispoof", "deepfake").
    """

    def __init__(self, window_size=WINDOW_SIZE, position_threshold=POSITION_THRESHOLD):
        self.window_size = window_size
        self.position_threshold = position_threshold
        self._next_id = 0
        # face_id -> {"position": (cx, cy), "histories": {metric: deque}}
        self._faces = {}

    def match(self, box):
        """Return the face_id for this box, creating a new one if no tracked face is close enough."""
        x1, y1, x2, y2 = box
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        closest_id = None
        closest_dist = float("inf")
        for fid, state in self._faces.items():
            px, py = state["position"]
            dist = ((center[0] - px) ** 2 + (center[1] - py) ** 2) ** 0.5
            if dist < self.position_threshold and dist < closest_dist:
                closest_id = fid
                closest_dist = dist

        if closest_id is None:
            closest_id = self._next_id
            self._next_id += 1
            self._faces[closest_id] = {"position": center, "histories": {}}

        self._faces[closest_id]["position"] = center
        return closest_id

    def record(self, face_id, metric, score):
        """Append a score for `metric` on `face_id`, return the running average."""
        histories = self._faces[face_id]["histories"]
        if metric not in histories:
            histories[metric] = deque(maxlen=self.window_size)
        histories[metric].append(score)
        history = histories[metric]
        return sum(history) / len(history)

    def frame_count(self, face_id, metric):
        histories = self._faces[face_id]["histories"]
        return len(histories.get(metric, ()))

    def is_warmed_up(self, face_id, metric):
        return self.frame_count(face_id, metric) >= self.window_size

    def prune(self, matched_ids):
        """Drop tracked faces that weren't seen in the latest frame."""
        for fid in list(self._faces.keys()):
            if fid not in matched_ids:
                del self._faces[fid]

    def clear(self):
        self._faces.clear()


class VerificationSession:
    """
    Owns one FaceDetector (stateful, not shareable) for the lifetime of a
    WebSocket connection, and drives the shared antispoof model (stateless,
    GPU-resident, safe to reuse across sessions) per frame.
    """

    def __init__(self, face_detector, antispoof_model, antispoof_threshold=0.5):
        self.detector = face_detector
        self.antispoof_model = antispoof_model
        self.antispoof_threshold = antispoof_threshold
        self.tracker = FaceTracker()

    def process_frame(self, bgr_frame):
        """
        Run detection + anti-spoofing on one frame.

        Returns a JSON-serializable dict describing every face currently
        tracked in this session.
        """
        t_start = time.perf_counter()

        height, width = bgr_frame.shape[:2]
        self.detector.set_image(bgr_frame, width, height, detection_scale=0.35)
        num_faces = self.detector.detect_faces()
        t_detect = time.perf_counter()

        if num_faces <= 0:
            self.tracker.clear()
            return {
                "faces": [],
                "timing_ms": {"detect": _ms(t_start, t_detect), "antispoof": 0.0, "total": _ms(t_start, t_detect)},
            }

        matched_ids = set()
        faces_out = []
        antispoof_total = 0.0

        for i in range(num_faces):
            x1, y1, x2, y2 = self.detector.get_face_box(i)
            face_id = self.tracker.match((x1, y1, x2, y2))
            matched_ids.add(face_id)

            t0 = time.perf_counter()
            antispoof_result = self.antispoof_model.get_liveness_score(bgr_frame, x1, y1, x2, y2)
            antispoof_total += time.perf_counter() - t0

            avg_liveness = self.tracker.record(face_id, "antispoof", antispoof_result["score"])
            warmed_up = self.tracker.is_warmed_up(face_id, "antispoof")

            faces_out.append({
                "face_id": face_id,
                "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "warming_up": not warmed_up,
                "frames_seen": self.tracker.frame_count(face_id, "antispoof"),
                "window_size": self.tracker.window_size,
                "antispoof": {
                    "frame_score": antispoof_result["score"],
                    "avg_score": avg_liveness,
                    "is_real": avg_liveness > self.antispoof_threshold,
                },
            })

        self.tracker.prune(matched_ids)

        return {
            "faces": faces_out,
            "timing_ms": {
                "detect": _ms(t_start, t_detect),
                "antispoof": round(antispoof_total * 1000, 1),
                "total": round((time.perf_counter() - t_start) * 1000, 1),
            },
        }
