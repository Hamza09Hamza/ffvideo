#pragma once
#ifndef _FFVIDEO_PYTHON_H_
#define _FFVIDEO_PYTHON_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Creates a FaceDetector.
 * Returns a void* handle — an opaque pointer Python holds like a ticket number.
 * Returns NULL if it fails (e.g., model file not found).
 *
 * model_path : full path to shape_predictor_68_face_landmarks.dat
 * face_model : 0 = 68-point,  1 = 81-point
 */
void* ffvideo_detector_create(const char* model_path, int face_model);

/*
 * Destroys the detector and frees memory.
 * Always call this when done — C++ does NOT have garbage collection.
 *
 * handle : the void* from ffvideo_detector_create()
 */
void ffvideo_detector_destroy(void* handle);
/*
 * Feeds a raw image into the detector for processing.
 *
 * pixels          : raw RGBA pixel bytes — this matches numpy uint8 arrays exactly
 * width / height  : image dimensions
 * detection_scale : 0.0–1.0 — use 0.35 to match what the C++ code defaults to
 */
void ffvideo_set_image(void*    handle,
                       uint8_t* pixels,
                       int32_t  width,
                       int32_t  height,
                       float    detection_scale);
/*
 * Runs HOG face detection on the image set by ffvideo_set_image().
 * Returns the number of faces found (0 = none, -1 = error/no image set).
 * Must call ffvideo_set_image() first.
 */
int32_t ffvideo_detect_faces(void* handle);

/*
 * Gets the bounding box of one detected face.
 *
 * Coordinates are mapped back to ORIGINAL frame space — ready to draw directly.
 * C++ handles the detection-scale → original-scale conversion internally.
 *
 * face_index       : 0 to (ffvideo_detect_faces() - 1)
 * out_x1 ... out_y2: pointers — C++ WRITES the rectangle values here (original pixels)
 */
void ffvideo_get_face_box(void*    handle,
                          int32_t  face_index,
                          int32_t* out_x1,
                          int32_t* out_y1,
                          int32_t* out_x2,
                          int32_t* out_y2);

/*
 * Extracts landmark points for one detected face.
 * Must call ffvideo_detect_faces() first.
 *
 * Coordinates are mapped back to ORIGINAL frame space — ready to draw directly.
 * Call ffvideo_landmark_count() first to know how many floats to allocate.
 *
 * face_index      : which face (0-based)
 * out_landmarks_x : caller-allocated array of N floats (N = ffvideo_landmark_count())
 * out_landmarks_y : same for y coordinates, also N floats
 *
 * Returns 1 on success, 0 on failure
 */
int32_t ffvideo_get_landmarks(void*    handle,
                               int32_t  face_index,
                               float*   out_landmarks_x,
                               float*   out_landmarks_y);

                       
/*
 * Extracts a standardized 128x128 aligned face crop for one detected face.
 * Uses landmarks internally to align the face (rotation, scale, centering).
 * This is what you feed directly into FaceNet or ArcFace.
 *
 * face_index : which face (0-based)
 * out_pixels : caller-allocates buffer of exactly 128*128*4 = 65536 bytes
 *
 * Returns 1 on success, 0 on failure
 */
int32_t ffvideo_get_face_chip(void*    handle,
                               int32_t  face_index,
                               uint8_t* out_pixels);


/*
 * Returns how many landmark points the loaded model produces: 68 or 81.
 *
 * WHY this matters in Python:
 *   You must allocate exactly this many floats before calling ffvideo_get_landmarks().
 *   If you always allocate 68 but the model is 81, you get a buffer overflow crash.
 *
 *   Usage in Python:
 *     count = lib.ffvideo_landmark_count(handle)   # ask first
 *     lx = (ctypes.c_float * count)()              # then allocate
 *     ly = (ctypes.c_float * count)()
 */
int32_t ffvideo_landmark_count(void* handle);


/*
 * NOTE: No coordinate conversion needed on the Python side.
 *
 * Face detection internally runs on a scaled-down image for speed,
 * but ffvideo_get_face_box() and ffvideo_get_landmarks() BOTH map
 * their results back to original frame coordinates before returning.
 *
 * Python receives numbers it can use directly on the original frame.
 * No math. No conversion. That work stays in C++.
 */


#ifdef __cplusplus
}
#endif

#endif // _FFVIDEO_PYTHON_H_
