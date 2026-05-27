import ctypes
import numpy as np
import cv2
from pathlib import Path
import insightface
import os

# Load the compiled C++ library
_dylib_path = Path(__file__).parent / "build" / "libffvideo_python.dylib"
_lib = ctypes.CDLL(str(_dylib_path))

# Define function signatures
_lib.ffvideo_detector_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
_lib.ffvideo_detector_create.restype = ctypes.c_void_p

_lib.ffvideo_detector_destroy.argtypes = [ctypes.c_void_p]
_lib.ffvideo_detector_destroy.restype = None

_lib.ffvideo_set_image.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int32, ctypes.c_int32, ctypes.c_float]
_lib.ffvideo_set_image.restype = None

_lib.ffvideo_detect_faces.argtypes = [ctypes.c_void_p]
_lib.ffvideo_detect_faces.restype = ctypes.c_int32

_lib.ffvideo_get_face_box.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
_lib.ffvideo_get_face_box.restype = None

_lib.ffvideo_get_landmarks.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
_lib.ffvideo_get_landmarks.restype = ctypes.c_int32

_lib.ffvideo_get_face_chip.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_uint8)]
_lib.ffvideo_get_face_chip.restype = ctypes.c_int32

_lib.ffvideo_landmark_count.argtypes = [ctypes.c_void_p]
_lib.ffvideo_landmark_count.restype = ctypes.c_int32


class FaceDetector:
    """
    This is a Python wrapper around the C++ face detector.
    
    """
    
    def __init__(self, model_path, face_model=0):
        """
        Create a detector.
        
        Args:
            model_path: Path to shape_predictor_*.dat file
            face_model: 0 = 68-point, 1 = 81-point
        """
        self.handle = _lib.ffvideo_detector_create(
            model_path.encode('utf-8'),
            face_model
        )
        if self.handle is None:
            raise RuntimeError(f"Failed to create detector: {model_path}")
        
        self.face_model = face_model
        self._landmark_count = None
    
    def __del__(self):
        """Clean up when the detector is deleted."""
        if hasattr(self, 'handle') and self.handle:
            _lib.ffvideo_detector_destroy(self.handle)
    
    def set_image(self, bgr_frame, width, height, detection_scale=0.35):
        """
        Set the image to process.

        Args:
            bgr_frame: numpy array of shape (height, width, 3) in BGR format (OpenCV format)
            width: Image width
            height: Image height
            detection_scale: 0.0-1.0, scale for HOG detection (0.35 is default)
        """
        # Convert BGR (OpenCV) → RGB → RGBA
        # OpenCV natively uses BGR, so we convert to RGB first before RGBA
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgba_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2RGBA)

        # Ensure it's a contiguous numpy array
        rgba_frame = np.ascontiguousarray(rgba_frame, dtype=np.uint8)

        _lib.ffvideo_set_image(
            self.handle,
            rgba_frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            width,
            height,
            detection_scale
        )
    
    def detect_faces(self):
        """
        Run HOG face detection.
        
        Returns:
            Number of faces found (0 = none, -1 = error)
        """
        return _lib.ffvideo_detect_faces(self.handle)
    
    def get_face_box(self, face_index):
        """
        Get the bounding box of a detected face.
        
        Args:
            face_index: 0-based index
        
        Returns:
            Tuple (x1, y1, x2, y2) in original frame coordinates
        """
        x1, y1, x2, y2 = ctypes.c_int32(), ctypes.c_int32(), ctypes.c_int32(), ctypes.c_int32()
        
        _lib.ffvideo_get_face_box(
            self.handle,
            face_index,
            ctypes.byref(x1),
            ctypes.byref(y1),
            ctypes.byref(x2),
            ctypes.byref(y2)
        )
        
        return (x1.value, y1.value, x2.value, y2.value)
    
    def get_landmarks(self, face_index):
        """
        Get facial landmarks for a detected face.
        
        Args:
            face_index: 0-based index
        
        Returns:
            Tuple of (x_coords, y_coords) numpy arrays, each of length 68 or 81
        """
        count = self.landmark_count()
        
        lx = (ctypes.c_float * count)()
        ly = (ctypes.c_float * count)()
        
        ret = _lib.ffvideo_get_landmarks(
            self.handle,
            face_index,
            lx,
            ly
        )
        
        if ret != 1:
            return None
        
        # Convert to numpy arrays
        x_coords = np.array([lx[i] for i in range(count)], dtype=np.float32)
        y_coords = np.array([ly[i] for i in range(count)], dtype=np.float32)
        
        return (x_coords, y_coords)
    
    def get_face_chip(self, face_index):
        """
        Extract a 128x128 aligned face crop (standardized for face recognition).
        
        Args:
            face_index: 0-based index
        
        Returns:
            128x128 RGBA numpy array (dtype uint8)
        """
        # Allocate 128*128*4 bytes
        chip_pixels = (ctypes.c_uint8 * (128 * 128 * 4))()
        
        ret = _lib.ffvideo_get_face_chip(
            self.handle,
            face_index,
            chip_pixels
        )
        
        if ret != 1:
            return None
        
        # Convert to numpy and reshape
        chip_array = np.array([chip_pixels[i] for i in range(128 * 128 * 4)], dtype=np.uint8)
        chip_array = chip_array.reshape((128, 128, 4))
        
        return chip_array
    
    def landmark_count(self):
        """
        Get the number of landmarks the loaded model produces.

        Returns:
            68 or 81
        """
        if self._landmark_count is None:
            self._landmark_count = _lib.ffvideo_landmark_count(self.handle)

        return self._landmark_count


class FaceEmbedder:
    """
    Face embedding extractor using InsightFace ArcFace model.

    Converts 128x128 face chips into 512-dimensional vectors for comparison.
    Two faces of the same person will have similar embeddings (small distance).
    Two faces of different people will have different embeddings (large distance).

    Usage:
        embedder = FaceEmbedder()
        chip = detector.get_face_chip(0)  # Get 128x128 chip
        embedding = embedder.get_embedding(chip)  # Get 512-dim vector
        distance = embedder.distance(embedding1, embedding2)
    """

    def __init__(self, model_name='buffalo_l'):
        """
        Initialize the face embedder.

        Args:
            model_name: InsightFace model to use
                - 'buffalo_l': Large model, high accuracy, slower (recommended)
                - 'buffalo_s': Small model, faster, reasonable accuracy
                - 'buffalo_m': Medium model, balanced
        """
        print(f"Loading InsightFace recognition model...")

        # Load just the recognition model (not the full face analysis pipeline)
        # This is much faster since we already have aligned face chips
        # The model is stored in ~/.insightface/models/buffalo_l/w600k_r50.onnx
        model_dir = os.path.expanduser('~/.insightface/models/buffalo_l')
        model_path = os.path.join(model_dir, 'w600k_r50.onnx')

        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            print("Downloading model first...")
            # Trigger download by creating a FaceAnalysis object
            _ = insightface.app.FaceAnalysis(name='buffalo_l')
            # Now the model should exist

        self.recognition_model = insightface.model_zoo.get_model(
            model_path,
            providers=['CoreMLExecutionProvider', 'CPUExecutionProvider']
        )

        print("✓ Face embedder ready")

    def _prepare_image(self, chip_rgba):

        # Extract RGB channels (drop alpha)
        # chip is in RGBA format from C++
        rgb = chip_rgba[:, :, :3]  # Take only R, G, B

        # Resize to 112×112 (what ArcFace model expects)
        # The model was trained on 112×112 aligned faces
        resized = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR)

        return resized

    def get_embedding(self, chip_rgba):
        """
        Extract 512-dimensional embedding from a face chip.

        Args:
            chip_rgba: 128×128 RGBA numpy array from detector.get_face_chip()

        Returns:
            numpy array of shape (512,) — the face embedding (normalized)
        """
        # Prepare image: 128×128 RGBA → 112×112 RGB
        # Result is (112, 112, 3) in uint8 format
        rgb_face = self._prepare_image(chip_rgba)

        # The model expects images in [0, 255] range (not normalized)
        # Keep as uint8
        # Note: The model's get_feat will handle internal normalization

        # Run the recognition model to extract embedding
        # Pass as a list with one image
        embedding = self.recognition_model.get_feat([rgb_face])

        # embedding is a 2D array (1, 512), extract the first row
        embedding = embedding[0].astype(np.float32)

        # IMPORTANT: The model returns unnormalized embeddings (norm ~6.0)
        # Normalize to unit length for proper distance calculations
        embedding_normalized = embedding / (np.linalg.norm(embedding) + 1e-8)

        return embedding_normalized

    @staticmethod
    def distance(embedding1, embedding2):
        """
        Calculate cosine distance between two embeddings.

        Returns distance in [0, 2] where:
        - 0 = identical faces
        - 1 = orthogonal (no similarity)
        - 2 = opposite direction

        Args:
            embedding1: 512-dim numpy array
            embedding2: 512-dim numpy array

        Returns:
            float: cosine distance
        """
        # Normalize vectors to unit length
        emb1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        emb2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-8)

        # Cosine similarity: dot product of normalized vectors (ranges [-1, 1])
        similarity = np.dot(emb1_norm, emb2_norm)

        # Cosine distance: 1 - similarity (ranges [0, 2])
        distance = 1.0 - similarity
        return distance

    @staticmethod
    def is_same_person(embedding1, embedding2, threshold=0.6):
        """
        Determine if two embeddings belong to the same person.

        Args:
            embedding1: 512-dim vector
            embedding2: 512-dim vector
            threshold: distance threshold (smaller = stricter matching)
                - 0.6: Strict (fewer false matches, more misses)
                - 0.5: Balanced (recommended)
                - 0.4: Loose (more false matches, fewer misses)

        Returns:
            bool: True if same person, False if different
        """
        dist = FaceEmbedder.distance(embedding1, embedding2)
        return dist < threshold
