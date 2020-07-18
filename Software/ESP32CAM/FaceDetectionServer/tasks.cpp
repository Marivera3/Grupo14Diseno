#include <WiFi.h>
#include "img_converters.h"
#include "fb_gfx.h"
#include "fd_forward.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

extern const uint16_t port;
extern const char * host;
extern WiFiServer server;
extern WiFiClient client;
extern sensor_t *s;
extern dl_matrix3du_t *current_frame;
extern SemaphoreHandle_t xBinarySemaphore;

void server_task(void *unused) {
  size_t _jpg_buf_len = 0;
  uint8_t *_jpg_buf = NULL;
  while(1) {
    client = server.available();
    if (client) {
      Serial.print("Connected to: "); Serial.println(client.remoteIP());
      if (current_frame) {
        Serial.println("Sending current frame: ");
        int width = current_frame->w;
        int height = current_frame->h;
        int channels = current_frame->c;
        uint8_t *item = current_frame->item;
        fmt2jpg(item, width*height*channels, width, height,
                PIXFORMAT_RGB888, 90, &_jpg_buf, &_jpg_buf_len);
        //Serial.println(_jpg_buf_len);
        client.write((char*) &_jpg_buf_len, 2);
        client.write(_jpg_buf, _jpg_buf_len);
        xSemaphoreGive(xBinarySemaphore);
        client.stop();                // terminates the connection with the client
      }
      //xSemaphoreGive(xBinarySemaphore);
    }
  }
}
/*
void capture_task(void * unused) {
  while(1) {
    capture_frame(&ff);
    yield();
    if (ff.detected) {
      Serial.println("Face detected!");
      current_frame = ff.image_matrix;
      xSemaphoreTake(xBinarySemaphore, 2000);
    }
    dl_matrix3du_free(ff.image_matrix);
    current_frame = NULL;
  }
}
//Serial.println(".");
      //String request = client.readStringUntil('\r');    // receives the message from the client
      //Serial.print("From client: "); Serial.println(request);
      //client.flush();

*/

/*
        char *buffer = (char*) malloc(width*channels*2*height);
        for (int i = 0; i < height; i++) {
          for (int j = 0; j < width*channels; j++) {
            byte nib1 = (item[i*width + j] >> 4) & 0x0F;
            byte nib2 = (item[i*width + j] >> 0) & 0x0F;
            buffer[(i*width + j)*2] = (char) nib1  < 0xA ? '0' + nib1  : 'A' + nib1  - 0xA;
            buffer[(i*width + j)*2+1] = (char) nib2  < 0xA ? '0' + nib2  : 'A' + nib2  - 0xA;
          }
          
        }
        client.println(buffer);
        free(buffer); */
