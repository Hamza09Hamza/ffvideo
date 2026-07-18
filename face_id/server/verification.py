"""
Stateless face verification: given a batch of frames captured client-side
(e.g. ~15 frames over ~1.5s), decide who this is (if anyone) and whether
they're a live person.

No per-connection state needed — unlike the old WebSocket version, every
call gets its own frames and returns a complete answer in one shot.
"""

import numpy as np

from face_id.server import db

MIN_GOOD_FRAMES = 8       # of the submitted batch, need at least this many single-face frames
LIVENESS_THRESHOLD = 0.5


def verify_batch(face_analyzer, antispoof_model, frames):
    """
    Args:
        face_analyzer: shared InsightFaceAnalyzer
        antispoof_model: shared MiniFASNetAntiSpoof
        frames: list of BGR numpy frames

    Returns a JSON-serializable dict, always with "success": bool.
    """
    liveness_scores = []
    embeddings = []

    for frame in frames:
        detected = face_analyzer.analyze(frame)
        if len(detected) != 1:
            continue  # skip frames with no face or an ambiguous multi-face shot

        face = detected[0]
        x1, y1, x2, y2 = face["box"]
        result = antispoof_model.get_liveness_score(frame, x1, y1, x2, y2)
        liveness_scores.append(result["score"])
        embeddings.append(face["embedding"])

    good_frames = len(liveness_scores)
    if good_frames < MIN_GOOD_FRAMES:
        return {
            "success": False,
            "reason": "insufficient_faces",
            "good_frames": good_frames,
            "required": MIN_GOOD_FRAMES,
            "frames_submitted": len(frames),
        }

    avg_liveness = float(np.mean(liveness_scores))
    if avg_liveness < LIVENESS_THRESHOLD:
        return {
            "success": False,
            "reason": "spoof_suspected",
            "avg_liveness": avg_liveness,
            "good_frames": good_frames,
            "frames_submitted": len(frames),
        }

    avg_embedding = np.mean(embeddings, axis=0)
    match = db.find_best_match(avg_embedding)

    if match is None:
        return {
            "success": False,
            "reason": "not_recognized",
            "avg_liveness": avg_liveness,
            "good_frames": good_frames,
            "frames_submitted": len(frames),
        }

    return {
        "success": True,
        "employee_id": match["employee_id"],
        "full_name": match["full_name"],
        "similarity": match["similarity"],
        "avg_liveness": avg_liveness,
        "good_frames": good_frames,
        "frames_submitted": len(frames),
    }
