# Roadmap & Pending Work

Status: Anti-spoofing complete and working. Next phases in order of priority.

---

## Phase 1: Core Pipeline (Current)

- [x] **Anti-Spoofing (MiniFASNet)**
  - [x] Model loading + inference
  - [x] Temporal aggregation (15-frame window)
  - [x] Per-face tracking (spatial proximity)
  - [x] Live camera testing
  - Status: ✅ Complete and validated

---

## Phase 2: Security Layers (TODO)

### Step 2: Deepfake Detection
**Priority:** High (blocks AI-generated attacks)  
**Complexity:** Medium  
**Estimated time:** 1-2 hours

**What to build:**
- [ ] `face_id/models/deepfake.py` — XceptionNet wrapper
  - Load pre-trained model
  - Crop & preprocess face
  - Run inference → real/fake probability [0, 1]
  - Handle multiple model variants if available
- [ ] `face_id/test_deepfake.py` — Live camera test
  - Show live deepfake scores
  - Test on: real faces, deepfake videos, AI-generated images

**Model source options:**
- Option A: FaceForensics++ dataset (custom training)
- Option B: Pre-trained checkpoint from FaceXlib or similar
- Decision: Start with pre-trained, train custom if needed

**Integration into pipeline:**
```python
# In pipeline.py (TODO)
antispoof_score = antispoof.get_liveness_score(...)
deepfake_score = deepfake.get_real_probability(...)  # NEW
if antispoof_score < 0.5 or deepfake_score < 0.5:
    return {"verified": False, "reason": "spoofing or deepfake"}
```

---

### Step 3: 3D Preprocessing (Face Normalization)
**Priority:** High (handles pose/lighting variation)  
**Complexity:** High  
**Estimated time:** 3-4 hours

**What to build:**
- [ ] `face_id/models/reconstruct3d.py` — 3DDFA_V2 wrapper
  - Load 3DDFA_V2 model
  - Reconstruct 3D face from 2D image
  - Normalize pose (rotate to frontal)
  - Normalize lighting (spherical harmonics)
  - Neutralize expression
  - Render back to clean 2D image
- [ ] `face_id/test_3d_preprocessing.py` — Test on various poses/lighting

**Why we need this:**
- Anti-spoofing only: works if face is frontal
- With 3D norm: works at any angle, any lighting
- Critical for real-world robustness

**Integration into pipeline:**
```python
# In pipeline.py (TODO)
antispoof_score = antispoof.get_liveness_score(...)
deepfake_score = deepfake.get_real_probability(...)
normalized_face = reconstruct3d.normalize(face)  # NEW
embedding = embedder.get_embedding(normalized_face)  # Use normalized face
```

---

## Phase 3: Registration & Verification (TODO)

### Step 4: Registration Flow
**Priority:** High (required for verification)  
**Complexity:** Medium  
**Estimated time:** 2 hours

**What to build:**
- [ ] `face_id/register.py`
  - Open camera
  - Capture 5 frames (user faces camera, good lighting)
  - Validate each frame (anti-spoofing + liveness check)
  - Run through pipeline: detection → spoofing → 3D norm → embedding
  - Average the 5 embeddings
  - Store canonical embedding + user_id in DB
  - Return: registration success or failure reason

**Database:** (TBD - SQLite for demo, proper DB for production)
```python
users_table = {
    user_id: {
        "embedding": [float] * 512,
        "registered_at": timestamp,
        "last_verified": timestamp
    }
}
```

---

### Step 5: Full Verification Pipeline
**Priority:** High (end-to-end system)  
**Complexity:** Medium  
**Estimated time:** 2 hours

**What to build:**
- [ ] `face_id/pipeline.py`
  - Single function: `verify_face(bgr_frame, user_id) → {verified: bool, confidence: float, reason: str}`
  - Runs all 6 steps in sequence
  - Returns structured result

**Logic:**
```
Input: frame + user_id
  ↓
[1] detect_face(frame) → x1,y1,x2,y2
    if no face: return {"verified": false, "reason": "no face"}
  ↓
[2] antispoof.is_real(frame, bbox)
    if score < 0.5: return {"verified": false, "reason": "spoofing detected"}
  ↓
[3] deepfake.is_real(frame, bbox)
    if score < 0.5: return {"verified": false, "reason": "deepfake detected"}
  ↓
[4] reconstruct3d.normalize(frame, bbox) → normalized_face
  ↓
[5] embedder.get_embedding(normalized_face) → embedding [512-dim]
  ↓
[6] match(embedding, user_id) → {verified: bool, distance: float}
    if distance < 0.3: return {"verified": true, "confidence": 1 - distance}
    else: return {"verified": false, "reason": "face doesn't match user"}
```

---

### Step 6: API Endpoint
**Priority:** Medium (for deployment)  
**Complexity:** Low  
**Estimated time:** 1 hour

**What to build:**
- [ ] `api.py` — Flask/FastAPI endpoint
  - POST `/verify` → accepts base64 frame + user_id
  - Runs full pipeline
  - Returns JSON: `{verified: bool, confidence: float, reason: str}`
  - Handles errors gracefully

---

## Phase 4: Testing & Validation (TODO)

- [ ] **Anti-spoofing validation**
  - Test on: real faces, printed photos, phone replays, masks
  - Record: accuracy, FP rate, FN rate

- [ ] **Deepfake detection validation**
  - Test on: real faces, AI-generated faces, deepfake videos
  - Record: detection rate, false positive rate

- [ ] **3D normalization validation**
  - Test on: frontal faces, 45° poses, profile, lighting variations
  - Verify: embeddings closer for same person after normalization

- [ ] **Full pipeline integration test**
  - Simulate registration (5 frames)
  - Simulate verification (1 frame, same user)
  - Test false rejection (different user)
  - Test false accept (spoofing/deepfake attacks)

- [ ] **Stress testing**
  - Multiple faces in frame
  - Very close faces (< 50px apart)
  - Partial face in frame
  - Blurry/low-light frames

---

## Phase 5: Optimization (TODO)

- [ ] Profile inference speed per component
- [ ] Optimize slow components (target: < 3s total)
- [ ] Convert to ONNX/TensorRT for production
- [ ] Optimize model sizes (quantization)
- [ ] GPU acceleration (CUDA/Metal Performance Shaders)

---

## Phase 6: Deployment (TODO)

- [ ] Docker containerization
- [ ] Database setup (production-grade)
- [ ] API authentication
- [ ] Rate limiting
- [ ] Monitoring & logging
- [ ] Load testing

---

## Decision Log

**2026-05-29: Architecture finalized**
- ✅ Chose MiniFASNet for anti-spoofing (lightweight, fast, proven)
- ✅ Chose temporal aggregation + per-face tracking (prevents flickering)
- ✅ Deferred deepfake detection (added to Phase 2 instead of Phase 1)
- ✅ Deferred 3D preprocessing (complex, build after deepfake detection works)

**Rationale:** Security first, complexity later. Fast shallow checks before expensive deep checks.

---

## Questions & Notes

- [ ] Which deepfake detection model to use? (Pre-trained vs. custom training)
- [ ] Database: SQLite for demo, PostgreSQL for production?
- [ ] API framework: Flask (simple) or FastAPI (production-ready)?
- [ ] Deployment: Docker only, or also consider serverless?
- [ ] User study needed? (accuracy on real users)
