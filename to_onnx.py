import tensorflow
import keras2onnx
import keras

new_model = keras.models.load_model("models/fer2013_mini_XCEPTION.119-0.65.hdf5")
onnx_model = keras2onnx.convert_keras(model, model.name)
keras2onnx.save_model(onnx_model, "out.onnx")
