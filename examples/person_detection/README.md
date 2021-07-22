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
