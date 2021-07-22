import argparse
import os
from pathlib import Path

import tflite
import tvm
from tvm import relay, micro

TARGET = "c -keys=cpu -link-params=1 -mcpu=cortex-m33 -model=nrf5340dk -runtime=c -system-lib=1"

def generate_project(input_model, output_dir):
    # Make sure our output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # The directory inside TVM where templates are stored
    template_dir = Path(tvm.__file__).parents[2] / "apps" / "microtvm" / "arduino" / "template_project"

    # Load our .tflite model into TVM's preferred format
    with open(input_model, "rb") as f:
        tflite_model_buf = f.read()
    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    mod, params = relay.frontend.from_tflite(tflite_model)
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = relay.build(mod, TARGET, params=params)

    project = tvm.micro.generate_project(
        str(template_dir.resolve()),
        mod,
        str(output_dir.resolve()),
        {"arduino_board": "spresense", "arduino_cmd": "arduino-cli", "verbose": 0},
    )


def main():
    parser = argparse.ArgumentParser(description='script to generate Arduino projects from .tflite models')
    parser.add_argument('input', type=str, help='input .tflite model file')
    parser.add_argument('output_dir', type=str, help='output project directory')
    args = parser.parse_args()
    generate_project(args.input, args.output_dir)


if __name__ == '__main__':
    main()
