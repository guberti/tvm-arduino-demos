#ifndef Parameters_h
#define Parameters_h

#include "standalone_crt/include/dlpack/dlpack.h"

// Some Arduinos (like the Spresense) have multiple CPUs,
// so this could be expaneded at some point
static const DLDevice TVM_HARDWARE_DEVICE = {kDLCPU, 0};

static const int TVM_INPUT_DATA_DIMENSION = 2;
static int64_t TVM_INPUT_DATA_SHAPE[] = {1, 1960};
static const DLDataType TVM_INPUT_DATA_TYPE = {kDLInt, 8, 0};
static const char* TVM_INPUT_LAYER = "Reshape_1";

static const int TVM_OUTPUT_DATA_DIMENSION = 2;
static int64_t TVM_OUTPUT_DATA_SHAPE[] = {1, 4};
static const DLDataType TVM_OUTPUT_DATA_TYPE = {kDLInt, 8, 0};

#endif
