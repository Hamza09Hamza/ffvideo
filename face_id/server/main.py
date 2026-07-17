"""
FastAPI server: streams anti-spoofing (liveness) scores over a WebSocket.

Run with:
    uvicorn face_id.server.main:app --host 0.0.0.0 --port 8000 --reload

MiniFASNet is loaded ONCE at startup and shared across all connections —
it's stateless at inference time and expensive to put on the GPU.
FaceDetector wraps a stateful C++ handle, so each WebSocket connection gets
its own instance (see verification_session.py).

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

from ffvideo import FaceDetector
from face_id.models.antispoofing import MiniFASNetAntiSpoof
from face_id.server.verification_session import VerificationSession

DLIB_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "shape_predictor_68_face_landmarks.dat"

# Populated at startup by the lifespan handler below.
models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading shared models (this happens once)...")
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

    face_detector = FaceDetector(str(DLIB_MODEL_PATH), face_model=0)
    session = VerificationSession(
        face_detector=face_detector,
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
