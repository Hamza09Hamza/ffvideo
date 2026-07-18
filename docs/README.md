# FFVideo — Python Person Re-Identification & Cross-Platform Port

**Credits:** This is a fork of [bsenftner/ffvideo](https://github.com/bsenftner/ffvideo) by [Blake Senftner](https://github.com/bsenftner).

## What's New in This Fork

This fork adds **Python bindings and real-time person re-identification** on top of the original C++ face detection code:

- **Python wrapper** (`ffvideo.py`): Lightweight ctypes bridge to C++ face detection
  - `FaceDetector`: HOG-based face detection with dlib landmarks
  - `FaceEmbedder`: 512-dimensional face embeddings via InsightFace ArcFace
- **Real-time person tracking** (`demo_embeddings.py`): Identifies and tracks people across video frames
  - Cosine distance-based person matching
  - Embedding blending for pose-invariant tracking
  - ~17 FPS on Apple Silicon with CoreML GPU acceleration
- **Cross-platform support**: Ported from Windows-only to macOS (and Linux compatible)
  - All platform-specific code wrapped with proper guards
  - CMake build system for easier compilation

**Original work by Blake Senftner** (see below) includes all the C++ FFmpeg/Dlib integration and face detection logic.

---

## Original FFVideo Documentation

![FFVideo00](https://user-images.githubusercontent.com/1216815/125710205-65eaf07c-31b3-43e0-b660-867780cbaba5.png)

An FFmpeg library wrapper and wxWidgets Player application with video filters and face detection. This is a no-audio video player intended for video experiments and developers learning how to code media applications.

FFVideo Player supports multiple simultaneous minimum delay playback video windows, seek, scrubbing, basic face detection, plus the surrounding code necessary for a moderately professional application, such as persistence for end-user settings and an embedded web browser providing a 'help window'.

The idea of this application is to provide a basic video app framework for developers wanting to learn and experiment with video filters without the overhead of audio processing or the normal frame delays of ordinary video playback.

### Original Features (by Blake Senftner)

 - **FFmpeg library support**
   - Video files, USB Cameras, IP Cameras, and IP video services
   - Video file seeks and scrubbing
   - Unlimited, chained single-source AVFilterGraph filters
   - Frame exporting
   - Uses Blake Senftner's modified FFmpeg: https://github.com/bsenftner/FFmpeg
   - Uses Blake Senftner's SQLite3 wrapper: https://github.com/bsenftner/kvs
   - Uses Jorge L Rodriguez's stb image scaling library: http://github.com/nothings/stb

 - **Multi-threaded wxWidgets Video Player**
   - Multiple simultaneous video windows
   - Exported video frames re-encoded as H.264 MP4 and elementary streams
   - Easy access to AVFilterGraph video filters
   - Dlib integration for face detection and landmark recovery
   - Embedded web browser as help window
   - Extensive in-code documentation

Performance: ~70 FPS on HD content using minimal CPU resources.

### Face Detection Components (Original)

- Dlib HOG-based face detection
- 68-point or 81-point facial landmarks
- Standardized passport-style face image extraction

**Required:** Download `shape_predictor_68_face_landmarks.dat` from:
```
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
```
Place it in `models/` directory after decompression.

---

## Quick Start

### Build C++ Library
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
cd ..
```

### Run Person Re-ID Demo
```bash
source venv/bin/activate
python3 demo_embeddings.py
```

**Controls:**
- `q`: Quit
- `t`: Toggle threshold display
- `+`/`-`: Adjust matching threshold

---

## Architecture

### `ffvideo.py` — Python-C++ Bridge
- `FaceDetector`: Wraps C++ HOG detector (from original project)
  - `set_image()`: Input camera frame
  - `detect_faces()`: Run detection
  - `get_face_box()`: Get bounding box
  - `get_face_chip()`: Extract 128×128 aligned face
  - `get_landmarks()`: Get facial landmarks

- `FaceEmbedder`: InsightFace ArcFace model
  - `get_embedding()`: Face chip → 512-dim vector
  - `distance()`: Cosine distance between embeddings
  - `is_same_person()`: Binary same/different decision

### `demo_embeddings.py` — Real-Time Person Tracking
1. Detect faces in each frame
2. Extract embeddings from face chips
3. Match embeddings against known people (distance < 0.35)
4. Track and update person embeddings over time
5. Remove people not seen for 30 seconds

---

## Original Project Information

This C++ codebase is a Visual Studio 2019 project with complex build dependencies. Refer to the original repository for Windows build instructions: https://github.com/bsenftner/ffvideo

### Original Known Issues

- PlayAll plays USB cameras first (one-by-one) before other streams to avoid thread safety issues
- Rebuilding against FFmpeg 4.2.3 removed replay instabilities but reduced speed slightly

---

## Contributing

This fork focuses on Python bindings and person re-identification. For enhancements to the original C++ video player, please contribute to the upstream project. 
