#ifndef CAMERA_H
#define CAMERA_H

#include "esp_camera.h"
#include "esp_timer.h"
#include "img_converters.h"
#include "fb_gfx.h"
#include "fd_forward.h"
#include "Arduino.h"

typedef struct {
        dl_matrix3du_t *image_matrix; //RGB matrix
        bool detected; //If there are detected faces in frame
        box_array_t *boxes; //Pointer to face box coordinates
} frame_face_t;

/*
typedef struct {
        dl_matrix3du_t **image_matrix; //RGB matrix array
        int face_number; // Number of faces
} face_array_t;
*/
// Camera configuration and initialization
void camera_config();

// Returns cut faces as FrameBuffers
esp_err_t extract_faces(frame_face_t *ff, dl_matrix3du_t **rgb_array);

//
esp_err_t capture_frame(frame_face_t *ff);

#endif
