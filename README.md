# Using TVM with Arduino
This repository intends to show users how [Apache TVM](https://github.com/apache/tvm) can be used to compile models for Arduino, and get competitive or faster performance compared to TensorFlow's built-in capabilities.

## Hardware Required
All we need is an Arduino-compatible board with enough flash and RAM. In theory, any board with enough flash and RAM should work, but the following boards have been tested and shown to work:
- [Sony SPRESENSE](https://www.adafruit.com/product/4419) (with optional [5 MP Camera](https://www.adafruit.com/product/4417))
- [Arduino Nano 33 BLE Sense](https://store.arduino.cc/usa/nano-33-ble-senseurl)
- [Teensy 4.1](https://www.pjrc.com/store/teensy41.html) (with optional external RAM and flash chips)
- [Teensy 4.0)(https://www.pjrc.com/store/teensy40.html)

## Software Required
Arduino support isn't yet merged into TVM, so we'll use [PR #8493](https://github.com/apache/tvm/pull/8493). Hopefully, this will be merged soon, along with command-line support for project generation, letting us skip writing Python code to actually generate the model. We'll need to clone this repository, build it from source, and add it to our Python path.

We'll also need to install the [Arduino IDE](https://www.arduino.cc/en/software) or Arduino's [command line interface](https://github.com/arduino/arduino-cli).

**If you have these dependencies installed, skip the rest of this file and go to a README for your desired example project.**

_TODO add installation instructions for TVM_
