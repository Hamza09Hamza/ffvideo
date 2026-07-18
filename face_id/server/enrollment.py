"""
Stateless enrollment: one image per named pose, submitted all at once by
the client. Every pose must clear single-face + liveness checks; if any
pose fails, nothing gets written and the client learns exactly which
pose(s) to retake.
"""

from face_id.server import db

LIVENESS_THRESHOLD = 0.5

REQUIRED_POSES = ["center", "left", "right", "up", "down"]


def enroll_batch(face_analyzer, antispoof_model, full_name, email, pose_frames):
    """
    Args:
        pose_frames: dict of {pose_name: BGR numpy frame}, one entry per
            name in REQUIRED_POSES.

    Returns {"success": True, "employee_id": ...} or
            {"success": False, "failures": {pose_name: reason, ...}}.
    """
    failures = {}
    embeddings = {}

    for pose in REQUIRED_POSES:
        frame = pose_frames[pose]
        detected = face_analyzer.analyze(frame)

        if len(detected) == 0:
            failures[pose] = "no_face"
            continue
        if len(detected) > 1:
            failures[pose] = "multiple_faces"
            continue

        face = detected[0]
        x1, y1, x2, y2 = face["box"]
        result = antispoof_model.get_liveness_score(frame, x1, y1, x2, y2)
        if result["score"] < LIVENESS_THRESHOLD:
            failures[pose] = "spoof_suspected"
            continue

        embeddings[pose] = face["embedding"]

    if failures:
        return {"success": False, "failures": failures}

    employee_id = db.create_employee(full_name, email)
    for embedding in embeddings.values():
        db.add_embedding(employee_id, embedding)

    return {"success": True, "employee_id": employee_id}
