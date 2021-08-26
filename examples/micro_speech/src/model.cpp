#include "model.h"
#include "parameters.h"
#include "standalone_crt/include/tvm/runtime/crt/logging.h"
#include "standalone_crt/include/tvm/runtime/crt/crt.h"
#include "standalone_crt/include/tvm/runtime/crt/packed_func.h"
#include "standalone_crt/include/tvm/runtime/crt/graph_executor.h"
#include "standalone_crt/include/dlpack/dlpack.h"

// Model
#include "model/graph_json.c"
#include "Arduino.h"

TVMModel::TVMModel()
{
  tvm_crt_error_t ret = TVMInitializeRuntime();


  TVMPackedFunc pf;
  TVMArgs args = TVMArgs_Create(NULL, NULL, 0);
  TVMPackedFunc_InitGlobalFunc(&pf, "runtime.SystemLib", &args);
  TVMPackedFunc_Call(&pf);

  TVMModuleHandle mod_syslib = TVMArgs_AsModuleHandle(&pf.ret_value, 0);

  // Create device
  int64_t device_type = kDLCPU;
  int64_t device_id = 0;

  DLDevice dev;
  dev.device_type = (DLDeviceType)device_type;
  dev.device_id = device_id;

  graph_runtime = NULL;
  TVMGraphExecutor_Create(graph_json, mod_syslib, &dev, &graph_runtime);
}


void TVMModel::TVM_inference(void *input_data, void *output_data) {
  // Reformat input data into tensor
  DLTensor input_data_tensor = {
    input_data,
    TVM_HARDWARE_DEVICE,
    TVM_INPUT_DATA_DIMENSION,
    TVM_INPUT_DATA_TYPE,
    TVM_INPUT_DATA_SHAPE,
    NULL,
    0,
  };

  // Run inputs through the model
  TVMGraphExecutor_SetInput(graph_runtime, TVM_INPUT_LAYER, (DLTensor*) &input_data_tensor);
  TVMGraphExecutor_Run(graph_runtime);

  // Prepare our output tensor
  DLTensor output_data_tensor = {
    output_data, 
    TVM_HARDWARE_DEVICE,
    TVM_OUTPUT_DATA_DIMENSION,
    TVM_OUTPUT_DATA_TYPE,
    TVM_OUTPUT_DATA_SHAPE,
    NULL,
    0,
  };

  // Populate output tensor
  TVMGraphExecutor_GetOutput(graph_runtime, 0, &output_data_tensor);
}
