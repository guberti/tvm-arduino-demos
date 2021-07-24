# TVM-based Audio Recognition for Arduino
This example demostrates using Arduino listen for audio with a microphone, and check if the words "yes" or "no" were said.

**This README assumes you have the Arduino IDE and a compatible version of TVM installed**

The example here is designed to work with the [Arduino Nano BLE Sense](https://store.arduino.cc/usa/nano-33-ble-sense) with its built-in microphone. However, Arduino's libraries make using a different microphone and board very easy.

# Choosing a Model
If you haven't already, clone this repository to your computer:

```
https://github.com/guberti/tvm-arduino-demos.git
```

If you're using an Spresense board, you should be able to open `yes_no.ino` with the Arduino IDE, flash your board, and see the script work. However, in this tutorial we will demonstrate how a sketch like this could be built from scratch.

To start, we'll need a pre-trained machine learning model. TVM supports generating Arduino projects from all popular model formats (including PyTorch, TensorFlow, TFLite, MXNet, Onnx...). Here, we will use [`models/person_detection.tflite`](../models/person_detection.tflite). It was taken directly from [TFLite's micro examples page](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples), and will allow us to easily compare performance.

This model takes as input 49 spectogram slices that each consist of 40 signed 8-bit integers. Each slice has a duration of 30ms, and a stride (e.g. "step over") of 10 ms.

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
python3 generate_project.py input/path/yes_no.tflite output/path/project
```

# Memory Requirements
If the Arduino does not have enough RAM to run this sketch, it will run out of memory. **If this occurs, `LED_BUILTIN` will begin blinking twice, pausing, and repeating this pattern**. If this occurs on your board, either choose a new Arduino that meets the memory requirement, or increase the memory available.

# Processing Static Audio
Before we connect our model to a live audio stream, let's feed it a few static files to ensure it works properly. I've included two `.wav` files from Google's [Speech Commands dataset](https://www.tensorflow.org/datasets/catalog/speech_commands) under [`examples/yes_no/data`](data/). They're also included below, translated into `.mov` files (since GitHub doesn't support built-in playback of audio files - only video).

https://user-images.githubusercontent.com/3069006/126720124-f8685e12-8a30-4f56-b187-0d9b423f3601.mov

https://user-images.githubusercontent.com/3069006/126720147-781c4e99-4b14-474b-a55a-6f061e301061.mov

To better understand what we're working with, let's understand what's inside these `.wav` files. Each contains the air pressure at a specific time (a value from -32768 to 32767), sampled at 16 kHz. Since each audio clip is one second long, each contains 16,000 samples, each of which is a signed 16 bit integer. If we scale these values to be floats from -1.0 to 1.0, we can plot their waveforms:

```python
import tensorflow as tf
import matplotlib.pyplot as plt

def load_wav(filename):
	binary = tf.io.read_file(filename)
	audio, _ = tf.audio.decode_wav(binary)
	return tf.squeeze(audio, axis=-1)

fig, (yes_ax, no_ax) = plt.subplots(1, 2)
yes_ax.plot(load_wav("data/yes.wav"))
yes_ax.set_title('Yes Waveform')
no_ax.plot(load_wav("data/no.wav"))
no_ax.set_title('No Waveform')
plt.show()
```

![waveform_plot](https://user-images.githubusercontent.com/3069006/126737265-1ffb7fbc-3f5a-4f2b-b9ca-8b8045439fcd.png)

While their are audio models that take raw waveforms as input, the model we are using wants overlapping spectrogram slices, also called an audio fingerprint. Each slice will consist of 30 ms of audio (480 16-bit integers), and each slice will overlap 10 ms with the slices before and after. Since the first and last slices do not overlap, this means we will need 49 slices to cover our one-second audio cliips.

Within each slice, we will group frequencies into 40 buckets, and then measure the loudness of the frequencies in each bucket. This will produce 40 `float32` values for each audio slide. We will discuss in more detail how to sort these frequencies when we implement this on Arduino, but for now we can use Tensorflow's `audio_microfrontend` function. These fingerprints can be plotted as spectograms, where the x-axis represents time the 30 ms window was taken, the y-axis represents the bucket number, and brighter pixels indicate higher values.

```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.lite.experimental.microfrontend.python.ops import (
    audio_microfrontend_op as frontend_op,
)

def get_spectrogram(filename):
    binary = tf.io.read_file(filename)
    audio, _ = tf.audio.decode_wav(binary, desired_channels=1, desired_samples=16000)

    int16_input = tf.cast(tf.multiply(audio, 32768), tf.int16)
    micro_frontend = frontend_op.audio_microfrontend(
        int16_input,
        sample_rate=16000,
        window_size=30,
        window_step=20,
        num_channels=40,
        out_scale=1,
        out_type=tf.float32,
    )
    output = tf.multiply(micro_frontend, (10.0 / 256.0))
    return np.rot90(output)


fig, (yes_ax, no_ax) = plt.subplots(1, 2)
yes_ax.imshow(
get_spectrogram("data/yes.wav"), cmap="viridis", interpolation="nearest"
)
yes_ax.set_title("Yes Waveform")
no_ax.imshow(
get_spectrogram("data/no.wav"), cmap="viridis", interpolation="nearest"
)
no_ax.set_title("No Waveform")
plt.show()
```

![spectogram_plot](https://user-images.githubusercontent.com/3069006/126848626-5825e58d-a607-4726-8a48-74767d282389.png)

The next step we must take to preprocess these files 

# Testing Our Model

The `project.ino` file that is generated with our sketch is pretty bare:

```c
#include "src/model.h"

static Model model;

void setup() {
  model = Model();
  //model.inference(input_data, output_data);
}

void loop() {
  // put your main code here, to run repeatedly:
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
  model.inference(input_data, output);
  
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
  model = Model();

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
