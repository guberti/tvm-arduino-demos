# Using TVM with Arduino
This repository intends to show users how [Apache TVM](https://github.com/apache/tvm) can be used to compile models for Arduino, and get competitive or faster performance compared to TensorFlow's built-in capabilities.

## Hardware required
All we need is an Arduino-compatible board with enough flash and RAM. In theory, any board with enough flash and RAM should work, but the following boards have been tested and shown to work:
- [Adafruit Metro M4](https://www.adafruit.com/product/3382)
- [Adafruit Pybadge](https://www.adafruit.com/product/4200)
- [Arduino Due](https://store.arduino.cc/usa/due)
- [Arduino Nano 33 BLE](https://store.arduino.cc/usa/nano-33-ble-senseurl)
- [Feather S2](https://www.adafruit.com/product/4769)
- [Sony SPRESENSE](https://www.adafruit.com/product/4419) (with optional [5 MP Camera](https://www.adafruit.com/product/4417))
- [Teensy 4.0](https://www.pjrc.com/store/teensy40.html)
- [Teensy 4.1](https://www.pjrc.com/store/teensy41.html) (with optional external RAM and flash chips)
- [Wio Terminal](https://www.seeedstudio.com/Wio-Terminal-p-4509.html)


## Software required
Arduino support is an official part of the TVM source tree, but it is **not** part of any `tvm` or `tlcpack` packages. That means you'll need to build TVM from source, with microTVM support. 

There is also no `tvmc` support for microTVM project generation yet, though that is being worked on (see https://github.com/gromero/tvm/commits/tvmc_micro). You'll either have to write Python code to use microTVM, or just use the script `generate_project.py` provided in this repository. 

We'll also need to install the [Arduino IDE](https://www.arduino.cc/en/software) or Arduino's [command line interface](https://github.com/arduino/arduino-cli). You will also need to install the board-specific package for the Arduino hardware you'd like to use. On the graphical IDE, this can be done by first adding the board manager URL under `File->Preferences`, then adding the package by going to `Tools->Board->Board Manager`. For the command line interface, this can be done by running a command of the form:

```
arduino-cli core install arduino:mbed_nano
```

**If you have these dependencies installed, skip the rest of this file and go to a README for your desired example project.**

# Building TVM with Arduino support

First, we must clone TVM and initialize its submodules:

```
git clone https://github.com/apache/tvm.git
git submodule init
git submodule update --recursive
```

You can then follow the [official instructions for building TVM from source](https://tvm.apache.org/docs/install/from_source.html#python-package-installation). Make sure to add the following lines to your `config.cmake` to build with support for microTVM, which is necessary to use Arduino.

```cmake
set(BUILD_STATIC_RUNTIME OFF)
set(USE_SORT ON)
set(USE_MICRO ON)
set(USE_LLVM ON)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_FLAGS -Werror)
set(HIDE_PRIVATE_SYMBOLS ON)
```

# Installing Arduino environments

Follow the instructions for installing the Arduino IDE on your operating system, and install the packages for your boards like normal.

Note that the Sony SPRESENSE library has [a major bug](https://github.com/sonydevworld/spresense/issues/200) that will cause TVM to fail. This issue should be fixed in version 2.3.0 of the library, but until that occurs, you can use the [develop branch](https://github.com/sonydevworld/spresense-nuttx/compare/develop) for spresense-nuttx.

For the Nano 33 BLE, running out of memory will cause the board to enter a bad state, in which it will not communicate with the serial port and new code cannot be uploaded. To fix this issue, double-tap the reset button to force the board to enter the bootloader. Note that this could only be triggered by a bug - if the model detects it has run out of memory, it will usually fail gracefully and flash an error code on the LED.
