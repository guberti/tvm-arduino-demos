import argparse
import os
from pathlib import Path

import tflite
import tvm
from tvm import relay, micro

def generate_project(input_model, output_dir):
    output_dir = Path(output_dir)

    # The directory inside TVM where templates are stored
    template_dir = Path(tvm.__file__).parents[2] / "apps" / "microtvm" / "arduino" / "template_project"

    # Load our .tflite model into TVM's preferred format
    with open(input_model, "rb") as f:
        tflite_model_buf = f.read()
    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    mod, params = relay.frontend.from_tflite(tflite_model)

    # Code will run on ALL Arduinos, but will be optimized for the cxd5602gg processor
    target = tvm.target.target.micro(
        "cxd5602gg", options=["--link-params=1", "--unpacked-api=1", "--executor=aot"]
    )
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = relay.build(mod, target, params=params)

    project = tvm.micro.generate_project(
        str(template_dir.resolve()),
        mod,
        str(output_dir.resolve()),
        {"arduino_board": "spresense", "arduino_cmd": "arduino-cli", "verbose": 0, "project_type": "example_project"},
    )


def main():
    parser = argparse.ArgumentParser(description='script to generate Arduino projects from .tflite models')
    parser.add_argument('input', type=str, help='input .tflite model file')
    parser.add_argument('output_dir', type=str, help='output project directory')
    args = parser.parse_args()
    generate_project(args.input, args.output_dir)


if __name__ == '__main__':
    main()
