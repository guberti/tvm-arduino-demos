#include "../../src/standalone_crt/include/tvm/runtime/crt/module.h"
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_softmax(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast_cast_subtract_14669711146056581479_(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_reshape_cast_subtract_1(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_divide_add_round_cast_clip_cast(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_reshape_cast_subtract(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t _lookup_linked_param(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
static TVMBackendPackedCFunc _tvm_func_array[] = {
    (TVMBackendPackedCFunc)tvmgen_default_fused_nn_softmax,
    (TVMBackendPackedCFunc)tvmgen_default_fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast_cast_subtract_14669711146056581479_,
    (TVMBackendPackedCFunc)tvmgen_default_fused_reshape_cast_subtract_1,
    (TVMBackendPackedCFunc)tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip,
    (TVMBackendPackedCFunc)tvmgen_default_fused_divide_add_round_cast_clip_cast,
    (TVMBackendPackedCFunc)tvmgen_default_fused_reshape_cast_subtract,
    (TVMBackendPackedCFunc)_lookup_linked_param,
};
static const TVMFuncRegistry _tvm_func_registry = {
    "\007tvmgen_default_fused_nn_softmax\000tvmgen_default_fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast_cast_subtract_14669711146056581479_\000tvmgen_default_fused_reshape_cast_subtract_1\000tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip\000tvmgen_default_fused_divide_add_round_cast_clip_cast\000tvmgen_default_fused_reshape_cast_subtract\000_lookup_linked_param\000",    _tvm_func_array,
};
static const TVMModule _tvm_system_lib = {
    &_tvm_func_registry,
};
const TVMModule* TVMSystemLibEntryPoint(void) {
    return &_tvm_system_lib;
}
;