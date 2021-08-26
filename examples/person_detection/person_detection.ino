#include "src/model.h"
#include <Camera.h>

//// Global variables ////
static uint8_t INPUT_BUF[96 * 96];
static uint8_t OUTPUT_BUF[3];

//// Function called by camera streaming ////
void CamCB(CamImage img) {
  
  //// Perform image resize ////
  img.convertPixFormat(CAM_IMAGE_PIX_FMT_GRAY);
  uint8_t* originalBuf = (uint8_t*)img.getImgBuff();
  uint16_t outIndex = 0;
  
  for (int i = 0; i < 96; ++i) {
    for (int j = 0; j < 96; ++j) {
      uint16_t brightness = 0;
      
      for (int m = 0; m < 2; m++) {
        for (int n = 0; n < 2; n++) {
          uint32_t sp_row = 24 + 2 * i + m;
          uint32_t sp_col = 64 + 2 * j + n;
          uint32_t index = 320 * sp_row + sp_col;
          brightness += originalBuf[index];
        }
      }

      INPUT_BUF[outIndex] = (uint8_t) (brightness / 4);
      outIndex += 1;
    }
  }

  TVMExecute(INPUT_BUF, OUTPUT_BUF);
  boolean detectedPerson = OUTPUT_BUF[1] > OUTPUT_BUF[2];
  digitalWrite(LED_BUILTIN, detectedPerson);
}

void setup() {
  TVMInitialize();
  theCamera.begin();
  theCamera.startStreaming(true, CamCB);
}

void loop() {}
