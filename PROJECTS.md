# Project Roadmap

This fork of [bsenftner/ffvideo](https://github.com/bsenftner/ffvideo) is the foundation for two separate AI systems.
Both build on top of the Python-C++ bridge already built (`ffvideo.py`).

---

## Foundation (Already Done)

**Repo:** `Hamza09Hamza/ffvideo` (this repo)  
**Branch:** `master`

What exists:
- C++ face detection + 68-point landmarks via dlib (Blake Senftner's original work)
- Python wrapper (`ffvideo.py`) — `FaceDetector` + `FaceEmbedder`
- Real-time demo (`demo_embeddings.py`) — ~17 FPS, person tracking, embedding blending
- Apple Silicon GPU acceleration via CoreML

This is the shared base. Both projects below import from `ffvideo.py`.

---

## Path 1 — Face ID Verification (Admin 2FA)

**What it does:**  
Admin logs in with password (step 1), then the app silently opens the camera, captures a frame, and verifies their face (step 2). No preview shown. Rejects photos, video replays, deepfakes, and different people.

**Lives in:** `face_id/` folder inside this same repo  
**Branch:** `feature/face-id` → merge to `master` when complete  
**Deployed as:** API endpoint (FastAPI)

### Pipeline

```
[Frame in]
    ↓
[1] Quality Check          — Is the face clear, centered, well-lit?
    ↓
[2] Liveness / Anti-Spoof  — MiniFASNet (Silent-Face): blocks photo & video attacks
    ↓
[3] Deepfake Detection     — XceptionNet (FaceForensics++): blocks AI-generated faces
    ↓
[4] 3D Normalization       — 3DDFA_V2: normalize pose, expression, lighting
    ↓
[5] Embedding Extraction   — ArcFace buffalo_l: 512-dim vector
    ↓
[6] Identity Match         — Cosine distance < 0.3 vs stored canonical embedding
    ↓
[Result: verified / rejected + reason]
```

### Folder Layout

```
face_id/
  pipeline.py         — full 6-step verification pipeline
  register.py         — one-time admin registration flow
  models/
    antispoofing.py   — MiniFASNet wrapper
    deepfake.py       — XceptionNet wrapper
    reconstruct3d.py  — 3DDFA_V2 wrapper
    embedding.py      — ArcFace wrapper (reuses ffvideo.py FaceEmbedder)
```

### What Makes It Robust

| Attack | Blocked By |
|--------|-----------|
| Printed photo | MiniFASNet (step 2) |
| Video replay on screen | MiniFASNet (step 2) |
| 2D face mask | MiniFASNet (step 2) |
| AI-generated deepfake | XceptionNet (step 3) |
| Different person | ArcFace match (step 6) |
| Same person, weird angle | 3DDFA_V2 normalization (step 4) |

### Models Needed

| Model | Purpose | Est. Speed |
|-------|---------|-----------|
| MiniFASNet (Silent-Face) | Liveness detection | ~100ms |
| XceptionNet | Deepfake detection | ~200ms |
| 3DDFA_V2 | 3D reconstruction + normalization | ~300ms |
| buffalo_l w600k_r50.onnx | ArcFace embedding | ~100ms |

**Total: ~700ms** — well within the 3-5 second target.

### Registration Flow (done once per admin)

1. Admin opens registration tool
2. Captures 5 frontal frames (good lighting)
3. Each frame goes through steps 1–5 of the pipeline
4. Average the 5 embeddings → canonical embedding stored in DB

---

## Path 2 — Multi-Camera Person Tracking

**What it does:**  
Track people across multiple camera feeds in real time. Person A exits Camera 1, walks into Camera 2 — system knows it's still Person A.

**Lives in:** a new separate repo (`Hamza09Hamza/multicam-reid`)  
**Why separate:** different architecture (YOLO + ByteTrack + body ReID), different dependencies, different deployment.

> **Status:** planned for after Path 1 is complete.

### Two-Layer Architecture

**Layer 1 — Per-camera tracking (within one feed):**
```
[Camera frame]
    ↓
YOLO — detect all people (full body, not just face)
    ↓
ByteTrack / SORT — assign local IDs within this camera
    → "Person A is still Person A as they walk left"
```

**Layer 2 — Cross-camera re-identification:**
```
[Person crops from all cameras]
    ↓
Face embeddings (ArcFace) + Body embeddings (OSNet/FastReID)
    ↓
Match embeddings across cameras using cosine distance
    → "Person A from Camera 1 = Person B from Camera 2"
```

### Why Body Embeddings Too?

Cameras at a distance often can't see faces clearly.
Body re-ID (clothing, gait, shape) fills the gap when faces aren't usable.

### Where FaceX Fits Here

FaceX (~3ms/face in C) is a candidate for the **face detection layer** within each camera feed — the speed matters at scale:
- 4 cameras × 25 FPS × 8 people = ~800 face ops/second
- FaceX can handle this; our current setup (~60ms/face) cannot

FaceX is only 1 month old — needs accuracy testing before committing to it.

### Models Needed

| Model | Purpose |
|-------|---------|
| YOLOv8n or YOLOv8s | Person detection |
| ByteTrack | Single-camera person tracking |
| ArcFace (from ffvideo.py) | Face re-ID across cameras |
| OSNet or FastReID | Body re-ID (clothing/shape) |
| FaceX (optional) | Fast face detection alternative |

---

## Repo & Branch Strategy

```
Hamza09Hamza/ffvideo          ← this repo
│
├── master                    ← foundation + Path 1 production code
├── feature/face-id           ← Path 1 development branch
│
└── (Path 1 face_id/ folder lives here)

Hamza09Hamza/multicam-reid    ← new repo (create when starting Path 2)
│
└── main                      ← Path 2 lives entirely here
```

**Why not two branches in the same repo for Path 2?**  
Path 2 is architecturally separate — different models, different dependencies (YOLO, ByteTrack, body ReID), different deployment target. A separate repo keeps it clean and avoids polluting this repo's dependency list.

---

## Setup Instructions

### This Repo (Foundation + Path 1)

```bash
# 1. Clone
git clone https://github.com/Hamza09Hamza/ffvideo
cd ffvideo

# 2. Build C++ library
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
cd ..

# 3. Set up Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Download models
mkdir -p models
# Download shape_predictor_68_face_landmarks.dat from:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# Decompress it into models/

# 5. Test foundation works
python3 demo_embeddings.py
```

### Start Working on Path 1

```bash
# Create and switch to feature branch
git checkout -b feature/face-id

# Create the face_id module structure
mkdir -p face_id/models

# Work here, commit to feature/face-id
# When Path 1 is complete: merge back to master
git checkout master
git merge feature/face-id
```

### Path 2 (when ready)

```bash
# Create new separate repo on GitHub: Hamza09Hamza/multicam-reid
# Then:
mkdir multicam-reid
cd multicam-reid
git init
# Set up fresh venv with YOLO, ByteTrack, OSNet dependencies
# ffvideo.py can be imported as a local package or installed via pip
```
