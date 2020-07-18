#include "camera.h"
#include "fb_gfx.h"
#include "fd_forward.h"

#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

static mtmn_config_t mtmn_config = {0};

void camera_config() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  //init with high specs to pre-allocate larger buffers
  if(psramFound()){
    config.frame_size = FRAMESIZE_UXGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

#if defined(CAMERA_MODEL_ESP_EYE)
  pinMode(13, INPUT_PULLUP);
  pinMode(14, INPUT_PULLUP);
#endif

  // camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }
  // Face Recognition settings
  mtmn_config.type = FAST;
  mtmn_config.min_face = 80;
  mtmn_config.pyramid = 0.707;
  mtmn_config.pyramid_times = 4;
  mtmn_config.p_threshold.score = 0.6;
  mtmn_config.p_threshold.nms = 0.7;
  mtmn_config.p_threshold.candidate_number = 20;
  mtmn_config.r_threshold.score = 0.7;
  mtmn_config.r_threshold.nms = 0.7;
  mtmn_config.r_threshold.candidate_number = 10;
  mtmn_config.o_threshold.score = 0.7;
  mtmn_config.o_threshold.nms = 0.7;
  mtmn_config.o_threshold.candidate_number = 1;
}

// Returns cut faces as FrameBuffers
esp_err_t extract_faces(frame_face_t *ff, dl_matrix3du_t **rgb_array) {
  int x0, y0, x1, y1;
  box_array_t *boxes = ff->boxes;
  dl_matrix3du_t *frame = ff->image_matrix;
  for (int i = 0; i < boxes->len; i++) {
    x0 = (int)boxes->box[i].box_p[0];
    y0 = (int)boxes->box[i].box_p[1];
    x1 = (int)boxes->box[i].box_p[2];
    y1 = (int)boxes->box[i].box_p[3];
    int width = x1 - x0;
    int height = y1 - y0;
    rgb_array[i] = dl_matrix3du_alloc(1, width, height, 3);
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        rgb_array[i]->item[((h)*width + (w))*3] = frame->item[((h+y0-1)*width + (w+x0))*3];
        rgb_array[i]->item[((h)*width + (w))*3 + 1] = frame->item[((h+y0-1)*width + (w+x0))*3 + 1];
        rgb_array[i]->item[((h)*width + (w))*3 + 2] = frame->item[((h+y0-1)*width + (w+x0))*3 + 2];
      }  
    }
  }
  return ESP_OK;
}

esp_err_t capture_frame(frame_face_t *ff){
    ff->detected = false;
    camera_fb_t * fb = NULL;
    esp_err_t res = ESP_OK;
    int64_t fr_start = esp_timer_get_time();

    // Get FrameBuffer
    fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Camera capture failed");
        return ESP_FAIL;
    }

    size_t out_len, out_width, out_height;
    uint8_t * out_buf;
    bool s;
    int face_id = 0;

    // Generate empty 3D matrix
    // image_matrix->item: uint8_t array
    dl_matrix3du_t *image_matrix = dl_matrix3du_alloc(1, fb->width, fb->height, 3);
    if (!image_matrix) {
        esp_camera_fb_return(fb);
        Serial.println("dl_matrix3du_alloc failed");
        return ESP_FAIL;
    }

    out_buf = image_matrix->item;
    out_len = fb->width * fb->height * 3;
    out_width = fb->width;
    out_height = fb->height;

    // Transform frame to RGB and copy to image_matrix
    s = fmt2rgb888(fb->buf, fb->len, fb->format, out_buf);
    esp_camera_fb_return(fb);
    if(!s){
        dl_matrix3du_free(image_matrix);
        Serial.println("to rgb888 failed");
        return ESP_FAIL;
    }

    // Face detection
    box_array_t *net_boxes = face_detect(image_matrix, &mtmn_config);

    // If net_boxes is not NULL
    if (net_boxes){
        ff->detected = 1;
    }
    ff->boxes = net_boxes;
    ff->image_matrix = image_matrix;

    int64_t fr_end = esp_timer_get_time();
    return ESP_OK;
}
