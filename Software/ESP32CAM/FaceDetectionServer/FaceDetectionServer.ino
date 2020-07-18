#include "esp_camera.h"
#include "camera.h"
#include "tasks.h"
#include "SPI.h" 
#include <WiFi.h>
#include "esp_camera.h"
#include "esp_timer.h"
#include "img_converters.h"
#include "fb_gfx.h"
#include "fd_forward.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

#define TEST_CAMERA 0
#define SD_CARD 0

const char* ssid = "VTR-7909951";
const char* password = "sj8Sskpgtbkk";
//const char* ssid = "iPhone de Agustin";
//const char* password = "12345679";

const uint16_t port = 8090;
const char * host = "172.20.10.3";
WiFiServer server(8090);
WiFiClient client;

sensor_t *s;
frame_face_t ff;
dl_matrix3du_t *current_frame;

SemaphoreHandle_t xBinarySemaphore;
TaskHandle_t SERVER_TASK;

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);

  // WIFI Setup
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.println("...");
  }
  Serial.print("WiFi connected with IP: ");
  Serial.println(WiFi.localIP());
  server.begin();

  //Camera init
  camera_config();
  s = esp_camera_sensor_get();
  s->set_framesize(s, FRAMESIZE_CIF);

  // Server task init
  xBinarySemaphore = xSemaphoreCreateBinary();
  //xTaskCreatePinnedToCore(capture_task, "Capture task", 50000, NULL, 4, NULL, 0);
  //Serial.println("Started Capture Task");
  xTaskCreate(server_task, "Server task", 50000, NULL, 1, NULL);
  Serial.println("Started Server Task");
  xSemaphoreTake(xBinarySemaphore, 1000);
}

void loop () {
  capture_frame(&ff);
  if (ff.detected) {
    //Serial.println("Face detected!");
    current_frame = ff.image_matrix;
    xSemaphoreTake(xBinarySemaphore, 10000);
  }
  dl_matrix3du_free(ff.image_matrix);
  current_frame = NULL;
  //delay(5000);
}







#if TEST_CAMERA
  capture_frame(&ff);
  if (ff.detected) {
    int face_number = ff.boxes->len;
    dl_matrix3du_t **rgb_array = (dl_matrix3du_t**) malloc(face_number * sizeof(dl_matrix3du_t*));
    extract_faces(&ff, rgb_array);
    /*
     * Rutina para enviar fotos
     */
    // Free memory
    for (int i = 0; i < face_number; i++) {
      dl_matrix3du_free(rgb_array[i]);
    }
    free(ff.boxes->score);
    free(ff.boxes->box);
    free(ff.boxes->landmark);
    free(ff.boxes);
  }
  //Free memory
  dl_matrix3du_free(ff.image_matrix);
#endif
