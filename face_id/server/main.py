"""
FastAPI server: streams anti-spoofing (liveness) scores over a WebSocket.

Run with:
    uvicorn face_id.server.main:app --host 0.0.0.0 --port 8000 --reload

Both the face analyzer (GPU SCRFD detection + ArcFace embedding) and
MiniFASNet are loaded ONCE at startup and shared across all connections —
both are stateless at inference time and expensive to put on the GPU. See
insightface_analyzer.py for why this one's safe to share (unlike the old
dlib-based FaceDetector, which needed a fresh instance per connection).

Deepfake detection (XceptionNet) is deliberately not wired in here — see the
docstring in verification_session.py for why.
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from face_id.models.antispoofing import MiniFASNetAntiSpoof
from face_id.models.insightface_analyzer import InsightFaceAnalyzer
from face_id.server import db
from face_id.server.enrollment_session import EnrollmentSession
from face_id.server.verification_session import VerificationSession

# Populated at startup by the lifespan handler below.
models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading shared models (this happens once)...")
    models["analyzer"] = InsightFaceAnalyzer()
    models["antispoof"] = MiniFASNetAntiSpoof()
    print("✓ Models ready")
    yield
    models.clear()


app = FastAPI(lifespan=lifespan)

# Dev-time CORS: the React app runs on a different port (e.g. 5173) than
# this API (8000). Tighten this to your actual frontend origin before
# exposing the server beyond your LAN.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.websocket("/ws/verify")
async def verify(websocket: WebSocket):
    await websocket.accept()

    session = VerificationSession(
        face_analyzer=models["analyzer"],
        antispoof_model=models["antispoof"],
    )

    try:
        while True:
            # Frontend sends one JPEG frame per message as raw bytes.
            jpeg_bytes = await websocket.receive_bytes()

            buffer = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            if frame is None:
                await websocket.send_json({"error": "could not decode frame"})
                continue

            result = session.process_frame(frame)
            await websocket.send_json(result)

    except WebSocketDisconnect:
        pass


@app.websocket("/ws/enroll")
async def enroll(websocket: WebSocket, full_name: str, email: str | None = None):
    await websocket.accept()

    session = EnrollmentSession(
        face_analyzer=models["analyzer"],
        antispoof_model=models["antispoof"],
        full_name=full_name,
        email=email,
    )

    try:
        while True:
            jpeg_bytes = await websocket.receive_bytes()
            buffer = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            if frame is None:
                await websocket.send_json({"error": "could not decode frame"})
                continue


            result = session.process_frame(frame)
            await websocket.send_json(result)

            if not session.is_complete():
                continue
            
            employee_id = db.create_employee(session.full_name, session.email)
            for embedding in session.collected_embeddings:
                db.add_embedding(employee_id, embedding)
            await websocket.send_json({"status": "enrolled", "employee_id": employee_id})
            break
            

    except WebSocketDisconnect:
        pass
