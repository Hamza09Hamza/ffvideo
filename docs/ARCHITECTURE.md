# Face ID Verification System Architecture

## Overview

A robust face verification system for admin dashboard authentication. Silently verifies user identity in 3-5 seconds with multiple anti-spoofing layers.

**Goal:** Resist attacks (printed photos, video replays, deepfakes, different person) while remaining fast and user-friendly.

---

## System Pipeline

```
User requests sensitive data
        ↓
[1] Face Detection (dlib HOG)
    - Locate face in frame
    - Extract bounding box
        ↓
[2] Liveness Detection (MiniFASNet)
    - Is this a real face or a spoof?
    - Blocks: printed photos, videos, masks
        ↓
[3] Anti-Deepfake (XceptionNet) — TODO
    - Is this face AI-generated?
    - Blocks: deepfakes, face-swaps
        ↓
[4] 3D Preprocessing (3DDFA_V2) — TODO
    - Normalize pose → frontal view
    - Normalize lighting → standard conditions
    - Neutralize expression → baseline face
        ↓
[5] Embedding Extraction (ArcFace buffalo_l)
    - Extract 512-dim face vector
    - Captures identity information
        ↓
[6] Identity Matching (cosine distance)
    - Compare against stored embedding
    - Threshold: distance < 0.3
    - Returns: verified or rejected
```

---

## Components & Sources

### [1] Face Detection
**Library:** dlib (C++ HOG detector)  
**Source:** `ffvideo_player_src/ffvideo_python.cpp` (C++ wrapper)  
**Why dlib:**
- Fast, accurate on frontal faces
- Works on CPU (no GPU needed for detection)
- Well-established, production-tested
- Built-in landmark prediction (68 face points)

**Model:** shape_predictor_68_face_landmarks.dat  
**Source:** http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2  
**Why 68 points:** Covers eyes, nose, mouth, jaw—useful for downstream preprocessing

---

### [2] Liveness Detection (Anti-Spoofing)
**Architecture:** MiniFASNet (mobile-optimized)  
**Source:** [minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) (MIT License)  
**Trained on:** SiW (Spoofing in the Wild) dataset  
**Why MiniFASNet:**
- Lightweight (~1.8M parameters)
- Runs on CPU/MPS (fast inference ~100ms)
- Ensemble of two variants (V2 + V1SE) for robustness
- Detects multiple attack types: print, replay, mask

**Models:**
- `2.7_80x80_MiniFASNetV2.pth` — scale factor 2.7x
- `4_0_0_80x80_MiniFASNetV1SE.pth` — scale factor 4.0x

**How it works:**
1. Expand face bbox by scale factor (captures context)
2. Crop and resize to 80×80
3. Run through network → 3-class logit (spoof, real, spoof)
4. Softmax → probability for class "real"
5. Average both models for final score

**Preprocessing critical details:**
- Keep pixel values in [0, 255] (NOT normalized)
- Use BGR color space directly (OpenCV native)
- Each model needs separate crop at its scale

**Temporal aggregation:**
- Collect scores over 15-frame sliding window
- Decision based on AVERAGE score (not per-frame)
- Prevents single-frame false positives from flipping result
- Per-face tracking: each face maintains own history (50px spatial threshold)

---

### [3] Deepfake Detection — TODO
**Architecture:** XceptionNet  
**Source:** [facexlib](https://github.com/chaofengc/Face-Restoration-and-Enhancement) or custom training on FaceForensics++  
**Why XceptionNet:**
- Efficient for real-time inference
- Learned to detect AI artifacts: boundary blurs, unnatural textures, frequency patterns
- Works on single frames (no temporal info needed)

**Plan:** Load pre-trained model, return real/fake probability [0, 1]

---

### [4] 3D Preprocessing — TODO
**Architecture:** 3DDFA_V2 (3D Dense Face Alignment)  
**Source:** [cleardusk/3DDFA_V2](https://github.com/cleardusk/3DDFA_V2)  
**Why 3D normalization:**
- Handles pose variation (face can be tilted)
- Handles lighting variation (different light sources)
- Normalizes expression (different emotions/mouth positions)
- Renders clean "passport" image for embedding extraction

**Process:**
1. Reconstruct 3D face from 2D image
2. Rotate 3D face to frontal pose
3. Neutralize expression (relax muscles)
4. Apply standard lighting (spherical harmonics)
5. Render back to 2D image

**Output:** Clean, normalized 2D face for embedding extraction

---

### [5] Embedding Extraction
**Architecture:** ArcFace (buffalo_l)  
**Source:** [insightface](https://github.com/deepinsight/insightface)  
**Why ArcFace:**
- State-of-the-art face recognition
- 512-dim vector captures face identity
- Large margin loss ensures discrimination
- buffalo_l is most accurate variant

**Model:** Already in ffvideo.py as FaceEmbedder  
**Input:** Normalized face image (from step 4)  
**Output:** 512-dim embedding vector

---

### [6] Identity Matching
**Method:** Cosine distance  
**Threshold:** < 0.3 (strict, high precision)  
**Registration flow:**
1. Capture 5 frames of user (frontal, good lighting)
2. Run through pipeline (steps 1-5)
3. Average the 5 embeddings
4. Store canonical embedding in DB

**Verification flow:**
1. Capture 1 frame
2. Run through pipeline
3. Compute cosine distance to canonical
4. Return: verified (distance < 0.3) or rejected

---

## Why This Architecture?

### Security Layers
- **Layer 1 (Liveness):** Stops static attacks (photos, videos)
- **Layer 2 (Deepfake):** Stops AI-generated attacks
- **Layer 3 (3D Norm):** Stops pose/lighting variations
- **Layer 4 (Embedding):** Stops wrong-person attacks

### Performance
- Face detection: ~50ms (dlib HOG)
- Anti-spoofing: ~100ms (MiniFASNet ensemble)
- Deepfake detection: ~200ms (XceptionNet)
- 3D preprocessing: ~300ms (3DDFA_V2)
- Embedding: ~100ms (ArcFace)
- **Total: ~750ms** + network overhead = well within 3-5s target

### Robustness
- Ensemble models (V2 + V1SE) vote on liveness
- Temporal aggregation prevents flickering
- 3D normalization handles real-world variation
- Strict matching threshold (0.3) prevents false accepts

---

## Files & Code Structure

```
ffvideo/
  ffvideo_player_src/
    ffvideo_python.cpp      ← Face detection (dlib C++)
  ffvideo.py                ← Python wrapper + ArcFace embedding
  
  face_id/
    models/
      antispoofing.py       ← MiniFASNet wrapper
      _minifasnet_architecture.py ← MiniFASNet layers (internal)
      deepfake.py           ← TODO: XceptionNet wrapper
      reconstruct3d.py      ← TODO: 3DDFA_V2 wrapper
    
    test_antispoofing.py    ← Live camera test (temporal aggregation)
    pipeline.py             ← TODO: Full 6-step verification
    register.py             ← TODO: Registration flow
  
  docs/
    ARCHITECTURE.md         ← This file
    ROADMAP.md              ← Pending work
```

---

## Trade-offs & Decisions

### Why NOT Apple's approach?
- Apple uses dedicated hardware (Neural Engine, IR camera, dot projector)
- We're building software-only, single RGB camera
- Trade: slightly lower accuracy for portability

### Why NOT 3D reconstruction first?
- 3D reconstruction is slow (~300ms) and complex
- Liveness detection is fast (~100ms) and catches 80% of attacks
- Pipeline: fast security layers first, expensive normalization later
- Follows OWASP: fail fast, expensive checks last

### Why NOT TensorRT or ONNX?
- PyTorch simpler for prototyping
- Models small enough (no need extreme optimization yet)
- Plan: optimize to ONNX/TensorRT when moving to production

---

## Testing Strategy

**Current:**
- ✅ Anti-spoofing tested on live camera
- ✅ Temporal aggregation validated
- ✅ Per-face tracking working

**TODO:**
- [ ] Deepfake detection test (AI-generated faces)
- [ ] 3D preprocessing validation (pose/lighting robustness)
- [ ] Full pipeline integration test
- [ ] Attack simulation (print, video, deepfake)
- [ ] User study (registration + verification accuracy)

---

## References

- MiniFASNet: [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
- 3DDFA_V2: [3D Face Alignment](https://github.com/cleardusk/3DDFA_V2)
- ArcFace: [InsightFace](https://github.com/deepinsight/insightface)
- FaceForensics++: [Deepfake Detection Dataset](https://github.com/ondyari/FaceForensics)
- dlib: [Face Detection & Landmarks](http://dlib.net)
