#include "src/model.h"
#include <Camera.h>

static char LABELS[12][12] = {
  "ANGRY",
  "DISGUST",
  "FEAR",
  "HAPPY",
  "SAD",
  "SURPRISED",
  "NEUTRAL",
};

//// Global variables ////
static uint8_t INPUT_BUF[48 * 48];
static float OUTPUT_BUF[7];

byte getMaxIndex() {
  byte best = -1;
  float maximum = -1;

  for (byte i = 0; i < 7; i++) {
    if (OUTPUT_BUF[i] > maximum) {
      maximum = OUTPUT_BUF[i];
      best = i;
    }
  }

  return best;
}

//// Function called by camera streaming ////
void CamCB(CamImage img) {
  
  //// Perform image resize ////
  img.convertPixFormat(CAM_IMAGE_PIX_FMT_GRAY);
  uint8_t* originalBuf = (uint8_t*)img.getImgBuff();
  uint16_t outIndex = 0;
  
  for (int i = 0; i < 48; ++i) {
    for (int j = 0; j < 48; ++j) {
      uint16_t brightness = 0;
      
      for (int m = 0; m < 4; m++) {
        for (int n = 0; n < 4; n++) {
          uint32_t sp_row = 24 + 2 * i + m;
          uint32_t sp_col = 64 + 2 * j + n;
          uint32_t index = 320 * sp_row + sp_col;
          brightness += originalBuf[index];
        }
      }

      brightness /= 16.0;
      brightness /= 255.0;
      //brightness -= 0.5;
      //brightness *= 2;
      INPUT_BUF[outIndex] = brightness;
      outIndex += 1;
    }
  }

  Serial.write((byte*) INPUT_BUF, 48 * 48 * 4);

  /*TVMExecute(INPUT_BUF, OUTPUT_BUF);
  byte index = getMaxIndex();
  Serial.println(LABELS[index]);*/
}

void setup() {
  Serial.begin(115200);
  //TVMInitialize();

  theCamera.begin();
  theCamera.startStreaming(true, CamCB);
}

void loop() {
}
