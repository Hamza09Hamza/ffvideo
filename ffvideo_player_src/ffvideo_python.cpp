#include "ffvideo_python.h"

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_transforms.h>

#include <vector>
#include <string>
#include <cstring>  



struct FFVideoDetector
{
    dlib::frontal_face_detector              detector;   
    dlib::shape_predictor                    sp;          

    dlib::array2d<uint8_t>                   dlib_im;     
    dlib::array2d<uint8_t>                   dlib_scaled;

    std::vector<dlib::rectangle>              detections;  // filled by detect_faces()
    std::vector<dlib::full_object_detection>  landmarks;   // filled by get_landmarks()

    int32_t  original_width;
    int32_t  original_height;
    float    detection_scale;

    int      face_model;

    bool     image_set;
};



static void rgba_to_dlib_gray(const uint8_t*          rgba_pixels,
                               int32_t                 width,
                               int32_t                 height,
                               dlib::array2d<uint8_t>& out_gray)
{
    // Resize the dlib image if dimensions changed
    if (out_gray.nc() != width || out_gray.nr() != height)
        out_gray.set_size(height, width);

    for (int32_t y = 0; y < height; y++)
    {
        for (int32_t x = 0; x < width; x++)
        {
            // Each pixel is 4 bytes: R, G, B, A
            const uint8_t* p = rgba_pixels + (y * width + x) * 4;

            // ignoring A (alpha)
            out_gray[y][x] = static_cast<uint8_t>(
                0.299f * p[0] +   // Red
                0.587f * p[1] +   // Green
                0.114f * p[2]     // Blue
            );
        }
    }
}

extern "C" void* ffvideo_detector_create(const char* model_path, int face_model)
{
    FFVideoDetector* det = new FFVideoDetector();
    
    // Load the frontal face detector (built into dlib)
    det->detector = dlib::get_frontal_face_detector();
    
    // Load the shape predictor from file
    try
    {
        dlib::deserialize(model_path) >> det->sp;
    }
    catch (const std::exception& e)
    {
        delete det;
        return NULL;  // Failed to load model file
    }
    
    // Store which model we're using (0=68-point, 1=81-point)
    det->face_model = face_model;
    
    // Initialize state
    det->detection_scale = 0.35f;  // Default scale for HOG detection
    det->image_set = false;
    
    // Return opaque pointer 
    return (void*)det;
}


extern "C" void ffvideo_detector_destroy(void* handle)
{
    if (handle == NULL)
        return;  // Already destroyed or never created
    
    FFVideoDetector* det = (FFVideoDetector*)handle;
    delete det;
}

extern "C" void ffvideo_set_image(void*    handle,
                                   uint8_t* pixels,
                                   int32_t  width,
                                   int32_t  height,
                                   float    detection_scale)
{
    if (handle == NULL || pixels == NULL)
        return;
    
    FFVideoDetector* det = (FFVideoDetector*)handle;
    
    // Store original frame dimensions so we can map results back later
    det->original_width = width;
    det->original_height = height;
    det->detection_scale = detection_scale;
    
    // Convert RGBA to grayscale
    rgba_to_dlib_gray(pixels, width, height, det->dlib_im);

    // Calculate scaled dimensions for detection
    int32_t scaled_w = (int32_t)(width * detection_scale + 0.5f);
    int32_t scaled_h = (int32_t)(height * detection_scale + 0.5f);

    // Pre-allocate dlib_scaled to the correct size (critical!)
    det->dlib_scaled.set_size(scaled_h, scaled_w);

    // Resize dlib_im to dlib_scaled for detection
    dlib::interpolate_nearest_neighbor interp;
    dlib::resize_image(det->dlib_im, det->dlib_scaled, interp);

    det->image_set = true;
}

extern "C" int32_t ffvideo_detect_faces(void* handle)
{
    if (handle == NULL)
        return -1;
    
    FFVideoDetector* det = (FFVideoDetector*)handle;
    
    // Check that an image was set first
    if (!det->image_set)
        return -1;
    
    // Run HOG detector on the scaled image
    det->detections = det->detector(det->dlib_scaled);
    
    // Return the count of faces found
    return (int32_t)det->detections.size();
}

extern "C" void ffvideo_get_face_box(void*    handle,
                                      int32_t  face_index,
                                      int32_t* out_x1,
                                      int32_t* out_y1,
                                      int32_t* out_x2,
                                      int32_t* out_y2)
{
    if (handle == NULL || out_x1 == NULL || out_y1 == NULL || out_x2 == NULL || out_y2 == NULL)
        return;
    
    FFVideoDetector* det = (FFVideoDetector*)handle;
    
    // Bounds check
    if (face_index < 0 || face_index >= (int32_t)det->detections.size())
        return;
    
    // Get the detection rectangle (in scaled space)
    dlib::rectangle rect = det->detections[face_index];
    
    // Map back to original frame coordinates
    // If detection_scale = 0.35, a scaled coordinate of 100 is 100/0.35 in original space
    float inv_scale = 1.0f / det->detection_scale;
    
    *out_x1 = (int32_t)(rect.left() * inv_scale + 0.5f);
    *out_y1 = (int32_t)(rect.top() * inv_scale + 0.5f);
    *out_x2 = (int32_t)(rect.right() * inv_scale + 0.5f);
    *out_y2 = (int32_t)(rect.bottom() * inv_scale + 0.5f);
}

extern "C" int32_t ffvideo_get_landmarks(void*    handle,
                                          int32_t  face_index,
                                          float*   out_landmarks_x,
                                          float*   out_landmarks_y)
{
    if (handle == NULL || out_landmarks_x == NULL || out_landmarks_y == NULL)
        return 0;
    
    FFVideoDetector* det = (FFVideoDetector*)handle;
    
    // Bounds check
    if (face_index < 0 || face_index >= (int32_t)det->detections.size())
        return 0;
    
    // Get the detection rectangle for this face (in scaled space)
    dlib::rectangle rect = det->detections[face_index];
    
    // Run shape predictor to get landmarks
    dlib::full_object_detection landmarks = det->sp(det->dlib_scaled, rect);
    
    // Map landmarks back to original frame coordinates
    float inv_scale = 1.0f / det->detection_scale;
    
    size_t num_parts = landmarks.num_parts();
    for (size_t i = 0; i < num_parts; i++)
    {
        dlib::point p = landmarks.part(i);
        out_landmarks_x[i] = p.x() * inv_scale;
        out_landmarks_y[i] = p.y() * inv_scale;
    }
    
    return 1;  
}

extern "C" int32_t ffvideo_get_face_chip(void*    handle,
                                          int32_t  face_index,
                                          uint8_t* out_pixels)
{
    if (handle == NULL || out_pixels == NULL)
        return 0;
    
    FFVideoDetector* det = (FFVideoDetector*)handle;
    
    // Bounds check
    if (face_index < 0 || face_index >= (int32_t)det->detections.size())
        return 0;
    
    // Get detection rectangle and compute landmarks
    dlib::rectangle rect = det->detections[face_index];
    dlib::full_object_detection landmarks = det->sp(det->dlib_scaled, rect);
    
    // Compute alignment details for 128x128 standardized face
    // get_face_chip_details() handles both 68 and 81-point models
    std::vector<dlib::full_object_detection> landmark_vec;
    landmark_vec.push_back(landmarks);
    std::vector<dlib::chip_details> chip_dets = dlib::get_face_chip_details(landmark_vec, 128, 0.8);
    
    // Extract aligned 128x128 face chips from the scaled image
    dlib::array<dlib::array2d<uint8_t>> face_chips;
    dlib::extract_image_chips(det->dlib_scaled, chip_dets, face_chips);
    
    if (face_chips.size() == 0)
        return 0;
    
    // Convert grayscale chip to RGBA (replicate gray across RGB, set A=255)
    dlib::array2d<uint8_t>& chip = face_chips[0];
    uint8_t* out = out_pixels;
    
    for (int32_t y = 0; y < 128; y++)
    {
        for (int32_t x = 0; x < 128; x++)
        {
            uint8_t gray = chip[y][x];
            *out++ = gray;      // R
            *out++ = gray;      // G
            *out++ = gray;      // B
            *out++ = 255;       // A (fully opaque)
        }
    }
    
    return 1;  // Success
}
extern "C" int32_t ffvideo_landmark_count(void* handle)
{
    if (handle == NULL)
        return 0;

    FFVideoDetector* det = (FFVideoDetector*)handle;

    // Return landmark count based on the loaded model
    if (det->face_model == 0)
        return 68;  // 68-point shape predictor
    else if (det->face_model == 1)
        return 81;  // 81-point shape predictor
    else
        return 0;   // Unknown model
}