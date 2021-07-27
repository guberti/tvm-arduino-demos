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

To start, we'll need a pre-trained machine learning model. TVM supports generating Arduino projects from all popular model formats (including PyTorch, TensorFlow, TFLite, MXNet, Onnx...). Here, we will use [`models/person_detection.tflite`](../models/yes_no.tflite). It was taken directly from [TFLite's micro examples page](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples), and will allow us to easily compare performance.

This model takes as input 49 spectogram slices that each consist of 40 signed 8-bit integers. Each slice has a duration of 30ms, and a stride (e.g. "step over") of 20 ms.

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
Before we connect our model to a live audio stream, let's feed it a few static files to ensure it works properly. I chose to use these two files from Google's [Speech Commands dataset](https://www.tensorflow.org/datasets/catalog/speech_commands).

```python
YES_EXAMPLE = DATASET_DIR + "/yes/0a2b400e_nohash_0.wav"
NO_EXAMPLE = DATASET_DIR + "/no/0a2b400e_nohash_0.wav"
```

You can listen to them here:

https://user-images.githubusercontent.com/3069006/127215520-bbcd5f87-5416-4367-bd27-47297e44b7bf.mp4

https://user-images.githubusercontent.com/3069006/127215528-f03e2efe-7d12-4c76-945a-065eeee42d6b.mp4

First, let's visualize our `.wav` files. Each contains the air pressure at a specific time (a value from -32768 to 32767), sampled at 16 kHz. Since each audio clip is one second long, each contains 16,000 samples, each of which is a signed 16 bit integer. If we plot these over time, we get a waveform:

```python
def load_wav_as_samples(filename):
    binary = tf.io.read_file(filename)
    audio, _ = tf.audio.decode_wav(binary, desired_channels=1, desired_samples=16000)
    tensor = tf.cast(tf.multiply(audio, 32768), tf.int16)
    array = np.rot90(tensor.eval(session=tf.compat.v1.Session()))[0]
    return array

def graph_waveforms():
    fig, (yes_ax, no_ax) = plt.subplots(1, 2)
    yes_ax.plot(load_wav_as_samples(YES_EXAMPLE))
    yes_ax.set_title('Yes Waveform')
    no_ax.plot(load_wav_as_samples(NO_EXAMPLE))
    no_ax.set_title('No Waveform')
    fig.set_size_inches(19 * 0.7, 10 * 0.7)
    fig.set_dpi(400)
    plt.show()

graph_waveforms()
```

![waveform](https://user-images.githubusercontent.com/3069006/127216358-2f1ff89d-8b07-4c80-99e7-2aeb88e3afa6.png)

While their are audio models that take raw waveforms as input, the model we are using wants overlapping spectrogram slices, also called an audio fingerprint. Each slice will consist of 30 ms of audio (480 16-bit integers), and each slice will overlap 10 ms with the slices before and after. Since the first and last slices do not overlap, this means we will need 49 slices to cover our one-second audio cliips.

Within each slice, we will group frequencies into 40 buckets, and then measure the loudness of the frequencies in each bucket. This will produce 40 `float32` values for each audio slide. We will discuss in more detail how to sort these frequencies when we implement this on Arduino, but for now we can use Tensorflow's `get_features_for_wav` function. 

Since many Arduinos lack a floating point module and cannot perform float arithematic efficiently, the `tflite` model we're using is **quantized** and uses integers for its calculations. Thus, we'll need to convert our input data to integers using the same quantization parameters we did for our model. This will give us a 40 x 49 array of signed, 8-bit integers. 

We can then plot these as a spectogram:

```python
def get_quantized(wav_filename, sess):
  test_data = audio_processor.get_features_for_wav(wav_filename, model_settings, sess)[0]

  input_scale = 0.10140931606292725
  input_zero_point = -128
  test_data = test_data / input_scale + input_zero_point
  
  test_data = test_data.astype(np.int8)
  return test_data

def graph_spectograms():
    with tf.Session() as sess:
      yes_array = get_quantized(YES_EXAMPLE, sess)
      no_array = get_quantized(NO_EXAMPLE, sess)

    fig, (yes_ax, no_ax) = plt.subplots(1, 2)
    yes_ax.imshow(yes_array.T, cmap="viridis", origin="lower")
    yes_ax.set_title("Yes Waveform")
    no_ax.imshow(no_array.T, cmap="viridis", origin="lower")
    no_ax.set_title("No Waveform")
    fig.set_size_inches(19 * 0.7, 10 * 0.7)
    fig.set_dpi(400)
    plt.show()

graph_spectograms()
```
![spectogram](https://user-images.githubusercontent.com/3069006/127217045-a2703540-d513-4d36-a1d7-71e560dcc378.png)

In these spectograms, the x-axis represents time. As seen in the waveform, the second half of the clips is silent, so the lack of data in the right half of the graph makes sense.

Next, we must quantize and flatten these files. Since the Arduino board we are targeting probably lacks a floating-point module, the `.tflite` file has been quantized and expects an input tensor of `int8_t`s. 

Lastly, we must deal with the fact Arduino will only compile code files (with the extensions `.ino`, `.h`, `.c`, `.cpp`). We can work around this restriction by encoding our images as C byte arrays:

```c
static const char input_yes[1920] = {
  0xff,
  0xff,
  ...
  0xff,
};
```

These can be generated with the script [`binary_to_c.py`](/binary_to_c.py) with a command of the form:

```
python3 binary_to_c.py \
  examples/yes_no/data/yes_0a2b400e_nohash_0.raw \
  examples/yes_no/yes.c \
  --name yes_data 
```

Pre-existing versions of these files may be found under [`examples/yes_no`](/).

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
#include "yes.c"
#include "no.c"
```

We'll then define a function that will run inference on an image, and display the results on the serial monitor:

```c
void testInference(int8_t input_data[1920]) {
  int8_t output[4];
  model.inference(input_data, output);
  
  for(int i = 0; i < 4; i++) {
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

  Serial.println("Yes results:");
  testInference(yes_data);
  
  Serial.println("No results:");
  testInference(no_data);

  Serial.end();
}
```

If we then run our sketch and set the serial monitor to `9600` baud, we'll see the following output:

```
Yes results:
-128, -127, 126, -127, 
No results:
-128, -128, -128, 127, 
```

The four values here represent the model's confidence that the input data is silence, an unknown word, "yes", and "no" respectively. Thus, we can see our model correctly identified both the "yes" and the "no" audio samples.
