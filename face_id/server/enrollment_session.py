"""
Per-connection enrollment state: capture N valid (single-face, live) frames
for one new employee, then persist their embeddings to the DB.
"""

REQUIRED_CAPTURES = 5      # how many good embeddings to collect before enrollment is complete
LIVENESS_THRESHOLD = 0.5


class EnrollmentSession:
    """
    Drives one WebSocket connection's enrollment attempt. Like
    VerificationSession, the face analyzer and antispoof model are shared
    GPU-resident singletons passed in from main.py — this class only holds
    the state specific to THIS one enrollment (the embeddings collected so
    far).
    """

    def __init__(self, face_analyzer, antispoof_model, full_name, email=None):
        self.face_analyzer = face_analyzer
        self.antispoof_model = antispoof_model
        self.full_name = full_name
        self.email = email
        self.collected_embeddings = []  # list of np.ndarray(512,), one per good frame

    def process_frame(self, bgr_frame):
        """
        Run one frame through the enrollment gate.

        Returns a JSON-serializable status dict. Doesn't touch the DB —
        that only happens once self.is_complete() is True (see main.py).
        """
        detected = self.face_analyzer.analyze(bgr_frame)

        if len(detected) == 0:
            return {"status": "no_face", "captured": len(self.collected_embeddings), "required": REQUIRED_CAPTURES}
        elif len(detected) > 1:
            return {"status": "multiple_faces", "captured": len(self.collected_embeddings), "required": REQUIRED_CAPTURES}

        face = detected[0]
        x1, y1, x2, y2 = face["box"]

        liveness_score = self.antispoof_model.get_liveness_score(bgr_frame, x1, y1, x2, y2)
        if liveness_score["score"] < LIVENESS_THRESHOLD:
            return {"status": "spoof_suspected", "captured": len(self.collected_embeddings), "required": REQUIRED_CAPTURES}

        self.collected_embeddings.append(face["embedding"])
        return {"status": "capturing", "captured": len(self.collected_embeddings), "required": REQUIRED_CAPTURES}

    def is_complete(self):
        """True once enough good embeddings have been collected."""
        return len(self.collected_embeddings) >= REQUIRED_CAPTURES
