#ifndef Model_h
#define Model_h

#include "standalone_crt/include/tvm/runtime/crt/graph_executor.h"

class TVMModel {
 public:
  TVMModel();
  void TVM_inference(void* input_data, void* output_data);

 private:
  TVMGraphExecutor* graph_runtime;
};

#endif
