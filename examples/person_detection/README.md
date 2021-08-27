# TVM-based Person Detector for Arduino
This example demostrates using Arduino to read images for the camera, check if they contain a person, and enable an LED if they do.

**This README assumes you have the Arduino IDE and a compatible version of TVM installed**

The example here is designed to work with the [Sony Spresense](https://www.adafruit.com/product/4419), along with its [5 MP camera](https://www.adafruit.com/product/4417). However, we will show how to modify this example for any other board/camera, or to use static images as inputs. Please note that on a Cortex-M4 core, this sketch requires **[x] kB of flash and [y] kB of RAM**.

# Choosing a Model
If you haven't already, clone this repository to your computer:

```
https://github.com/guberti/tvm-arduino-demos.git
```

If you're using an Spresense board, you should be able to open `person_detection.ino` with the Arduino IDE, flash your board, and see the script work. However, in this tutorial we will demonstrate how a sketch like this could be built from scratch.

To start, we'll need a pre-trained machine learning model. TVM supports generating Arduino projects from all popular model formats (including PyTorch, TensorFlow, TFLite, MXNet, Onnx...). Here, we will use [`models/person_detection.tflite`](../models/person_detection.tflite). It was taken directly from [TFLite's micro examples page](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples), and will allow us to easily compare performance.

This model takes as input a 96x96 unsigned 8-bit grayscale image, and outputs a 1x3 tensor of unsigned 8-bit integers with its predictions. The middle entry represents the model's confidence the input image contains a person - the other two entries represent the model's confidence it does not.

# Generating a Project
Support for command-line project generation doesn't exist as of 07/20/2021, so we'll need to write a short Python script. We'll need a few imports and constants:

```python
from pathlib import Path

import tflite
import tvm
from tvm import relay, micro

TARGET = "c -keys=cpu -link-params=1 -mcpu=cortex-m33 -model=nrf5340dk -runtime=c -system-lib=1"
INPUT_MODEL = # Add the path to your input model
OUTPUT_DIR = # The directory where your Arduino project should be generated
```

We'll then use TVM's model functions for `.tflite` models to load our model with our compilation options from earlier:

```python
with open(INPUT_MODEL, "rb") as f:
    tflite_model_buf = f.read()
tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
mod, params = relay.frontend.from_tflite(tflite_model)
with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
    mod = relay.build(mod, TARGET, params=params)
```

Lastly, using TVM's internal template directory we can generate our project:

```python
template_dir = Path(tvm.__file__).parents[2] / "apps" / "microtvm" / "arduino" / "template_project"
project = tvm.micro.generate_project(
    str(template_dir.resolve()),
    mod,
    str(output_dir.resolve()),
    {"arduino_board": "spresense", "arduino_cmd": "arduino-cli", "verbose": 0},
)
```

# Generating a Project, the Easy Way
This process for generating projects is very cumbersome, and [@guberti](https://github.com/guberti) is working on adding this support to the `tvmc` command line.

Until that happens, this repository contains a script for generating arbitrary Arduino projects from TVM. It may be used as follows:

```
python3 generate_project.py input/path/person_detection.tflite output/path/project
```

# Memory Requirements
If the Arduino does not have enough RAM to run this sketch, it will run out of memory. **If this occurs, `LED_BUILTIN` will begin blinking twice, pausing, and repeating this pattern**. If this occurs on your board, either choose a new Arduino that meets the memory requirement, or increase the memory available.

By default, the Sony SPRESENSE only makes 768 kB (50%) of its memory available to the main core. Our model, however, requires [x] kB. We can increase this by going to `Tools -> Memory` and selecting at least 1408 kB.

# Processing Static Images
Before we connect our model to a live camera feed, let's feed it a few static images to make sure it works properly. I've picked out two images from the COCO 2017 dataset, [one containing a person](https://farm3.staticflickr.com/2529/4142190207_fe9f344501_z.jpg) and [one that does not](https://farm9.staticflickr.com/8087/8466331752_b60857e9f4_z.jpg).

To feed these images into our model, we'll first need to convert them to 96x96 grayscale images. This can be done with ImageMagick or any other image editor. 96x96 grayscale versions of these images are available under [`examples/person_detection/data`](data/).

<p align="center">
  <img alt="Person" src="https://user-images.githubusercontent.com/3069006/126687844-140a94db-fe66-4890-b10a-fb0aada1e18a.png" width="40%">
&nbsp; &nbsp; &nbsp;
  <img alt="Not a person" src="https://user-images.githubusercontent.com/3069006/126687802-0450c25f-3580-4dcc-a3e1-e854baf34363.png" width="40%">
</p>

The PNG format contains a lot of unnecessary header formatting that would make our images difficult for our Arduino sketch to parse. One way to fix this is by converting them to [raw images](https://en.wikipedia.org/wiki/Raw_image_format), containing nothing but a list of 9216 unsigned 8-bit integers, each representing the brightness of a specific pixel. This can be done with ImageMagick:

```
convert person_grayscale.png -depth 8 r:person.raw
```

For your convenience, raw images have also been included in [`examples/person_detection/data`](data/).

Lastly, we must deal with the fact Arduino will only compile code files (with the extensions `.ino`, `.h`, `.c`, `.cpp`). We can work around this restriction by encoding our images as C byte arrays:

```c
static const char input_automobile[9216] = {
  0xff,
  0xff,
  ...
  0xff,
};
```

These can be generated with the script [`binary_to_c.py`](/binary_to_c.py) with a command of the form:

```
python3 binary_to_c.py \
  examples/person_detection/data/person.raw \
  examples/person_detection/person.c \
  --name person_data 
```

Pre-existing versions of these files may be found under [`examples/person_detection`](/).

# Testing Our Model

The `project.ino` file that is generated with our sketch is pretty bare:

```c
#include "src/model.h"

void setup() {
  TVMInitialize();
}

void loop() {
  //TVMExecute(input_data, output_data);
}
```

First, we'll add `#include` statements for the test images we just generated:

```c
#include "cat.c"
#include "person.c"
```

We'll then define a function that will run inference on an image, and display the results on the serial monitor:

```c
void testInference(uint8_t input_data[9216]) {
  uint8_t output[3];
  TVMExecute(input_data, output);
  
  for(int i = 0; i < 3; i++) {
    Serial.print(output[i]);
    Serial.print(", ");
  }
  Serial.println();
}
```

Lastly, we'll change setup to instantiate our model and call `testInference` on our two input images:

```c
void setup() {
  Serial.begin(9600);
  TVMInitialize();

  Serial.println("Person results:");
  testInference(person_data);
  
  Serial.println("Not a person results:");
  testInference(cat_data);

  Serial.end();
}
```

If we then run our sketch and set the serial monitor to `9600` baud, we'll see the following output:

```
Person results:
39, 235, 63, 
Not a person results:
20, 91, 225,
```

Here, the second element in our tensor represents the model's confidence the image contains a person, and the third element is the model's confidence it does not. We can ignore the first element for now. These results are promising, as it means our model correctly identified that the first image contained a person and the second did not. We're now ready to use this model for a live demo.

# Live Demonstration

**This part of the tutorial is specific to the Sony Spresense board. If using different hardware, consult the documentation to find out how images should be read from the camera.**

To build our demo, we'll need to import the Spresense's camera library at the top of our program:

```c
#include "src/model.h"
#include <Camera.h>

//// Global variables ////
static uint8_t INPUT_BUF[96 * 96];
static uint8_t OUTPUT_BUF[3];
```

In order to get the fastest framerate, we will use this library to stream 320x240 YUV422 frames to a callback function. We will set this up in our `setup` function:

```c
void setup() {
  TVMInitialize();
  theCamera.begin();
  theCamera.startStreaming(true, CamCB);
}

void loop() {}
```

Next, we must write our callback function `CamCB`. We can use the Spresense's built-in function to convert the image to grayscale:

```c
void CamCB(CamImage img) {
  
  //// Perform image resize ////
  img.convertPixFormat(CAM_IMAGE_PIX_FMT_GRAY);
  ...
}
```

We must now crop and scale the image to be 96x96. The Spresense has a library function to do this, but it does not work on grayscale images (the `convertPixFormat` function doesn't work on 96x96 images either, so reversing the order is not possible). We'll just write our own crop + scale function instead:

```c
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
```

All we have to do now is performe inference and compare the two values in our output tensor:

```
  TVMExecute(INPUT_BUF, OUTPUT_BUF);
  boolean detectedPerson = OUTPUT_BUF[1] > OUTPUT_BUF[2];
  digitalWrite(LED_BUILTIN, detectedPerson);
```

Now we only need to build and flash our sketch, and the Spresense should be running in no time! A completed version of this sketch may be found under [`examples/person_detection/person_detection.ino`](/examples/person_detection/person_detection.ino).
