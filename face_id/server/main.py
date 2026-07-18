"""
FastAPI server: stateless REST API for face enrollment + verification.

Run with:
    uvicorn face_id.server.main:app --host 0.0.0.0 --port 8000 --reload

Any client (mobile, web, anything that can do an HTTP POST with an API key
header) captures its own frames locally and submits them in one request —
the server holds no per-client session state between requests. This also
means every request's GPU/DB work is offloaded via asyncio.to_thread(),
so one slow request doesn't block the event loop from servicing every
other concurrent request's network I/O.

The face analyzer (GPU SCRFD detection + ArcFace embedding) and MiniFASNet
are loaded ONCE at startup and shared across all requests — both are
stateless at inference time and expensive to put on the GPU.
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import jwt
import numpy as np
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

TOKEN_TTL_MINUTES = 5

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

load_dotenv(Path(__file__).parent.parent.parent / ".env")

from face_id.models.antispoofing import MiniFASNetAntiSpoof
from face_id.models.insightface_analyzer import InsightFaceAnalyzer
from face_id.server.enrollment import REQUIRED_POSES, enroll_batch
from face_id.server.verification import verify_batch

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

# Dev-time CORS: tighten allow_origins to your actual app origins before
# exposing this beyond your LAN/tailnet. Native mobile apps ignore CORS
# entirely — it only matters for browser-based clients.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_api_key_header = APIKeyHeader(name="X-API-Key")


def require_api_key(key: str = Depends(_api_key_header)):
    if key != os.environ["API_KEY"]:
        raise HTTPException(status_code=403, detail="invalid API key")


def _decode_frame(upload_bytes: bytes):
    buffer = np.frombuffer(upload_bytes, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def _issue_token(employee_id, full_name):
    """
    A short-lived, signed proof of "this specific employee was just verified
    by face" — any app holding JWT_SECRET can verify this independently
    (no callback to this server needed) and trust employee_id/full_name.
    Short TTL because it's meant to be exchanged immediately for whatever
    longer-lived session the consuming app itself manages, not used as a
    long-lived credential on its own.
    """
    now = datetime.now(timezone.utc)
    payload = {
        "employee_id": employee_id,
        "full_name": full_name,
        "iat": now,
        "exp": now + timedelta(minutes=TOKEN_TTL_MINUTES),
    }
    return jwt.encode(payload, os.environ["JWT_SECRET"], algorithm="HS256")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/verify", dependencies=[Depends(require_api_key)])
async def verify(frames: list[UploadFile] = File(...)):
    decoded = [_decode_frame(await f.read()) for f in frames]
    decoded = [f for f in decoded if f is not None]

    result = await asyncio.to_thread(
        verify_batch, models["analyzer"], models["antispoof"], decoded
    )
    if result["success"]:
        result["token"] = _issue_token(result["employee_id"], result["full_name"])
    return result


@app.post("/api/v1/enroll", dependencies=[Depends(require_api_key)])
async def enroll(
    full_name: str = Form(...),
    email: str | None = Form(None),
    center: UploadFile = File(...),
    left: UploadFile = File(...),
    right: UploadFile = File(...),
    up: UploadFile = File(...),
    down: UploadFile = File(...),
):
    uploads = {"center": center, "left": left, "right": right, "up": up, "down": down}
    pose_frames = {pose: _decode_frame(await uploads[pose].read()) for pose in REQUIRED_POSES}

    result = await asyncio.to_thread(
        enroll_batch, models["analyzer"], models["antispoof"], full_name, email, pose_frames
    )
    return result
