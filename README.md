# Using TVM with Arduino
This repository intends to show users how [Apache TVM](https://github.com/apache/tvm) can be used to compile models for Arduino, and get competitive or faster performance compared to TensorFlow's built-in capabilities.

## Hardware required
All we need is an Arduino-compatible board with enough flash and RAM. In theory, any board with enough flash and RAM should work, but the following boards have been tested and shown to work:
- [Sony SPRESENSE](https://www.adafruit.com/product/4419) (with optional [5 MP Camera](https://www.adafruit.com/product/4417))
- [Arduino Nano 33 BLE Sense](https://store.arduino.cc/usa/nano-33-ble-senseurl)
- [Teensy 4.1](https://www.pjrc.com/store/teensy41.html) (with optional external RAM and flash chips)
- [Teensy 4.0)(https://www.pjrc.com/store/teensy40.html)

## Software required
Arduino support isn't yet merged into TVM, so we'll use [PR #8493](https://github.com/apache/tvm/pull/8493). Hopefully, this will be merged soon, along with command-line support for project generation, letting us skip writing Python code to actually generate the model. We'll need to clone this repository, build it from source, and add it to our Python path.

We'll also need to install the [Arduino IDE](https://www.arduino.cc/en/software) or Arduino's [command line interface](https://github.com/arduino/arduino-cli).

**If you have these dependencies installed, skip the rest of this file and go to a README for your desired example project.**

# Building TVM with Arduino support

Since Arduino support has not yet merged into TVM main (and is certainly not in the precompiled binaries), we'll need to check out the feature branch and build from scratch. Clone the branch with `git`:

```bash
git clone --branch arduino-project-api https://github.com/guberti/tvm.git tvm-arduino
```

We'll also need to initialize and clone our submodules:

```
git submodule init
git submodule update --recursive
```

You can then follow the [official instructions for building TVM from source](https://tvm.apache.org/docs/install/from_source.html#python-package-installation). Make sure to add the following lines to your `config.cmake` to build with support for microTVM, which is necessary to use Arduino.

```cmake
set(BUILD_STATIC_RUNTIME OFF)
set(USE_SORT ON)
set(USE_MICRO ON)
set(USE_LLVM llvm-config-10)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_FLAGS -Werror)
set(HIDE_PRIVATE_SYMBOLS ON)
```

# Installing Arduino environments

Follow the instructions for installing the Arduino IDE on your operating system, and install the packages for your boards like normal.

Note that the Sony SPRESENSE library has [a major bug](https://github.com/sonydevworld/spresense/issues/200) that will cause TVM to fail. This issue should be fixed in version 2.3.0 of the library, but until that occurs, you can use the [develop branch](https://github.com/sonydevworld/spresense-nuttx/compare/develop) for spresense-nuttx.
