
#include "tensorflow/core/kernels/cwise_op_sbc_mlu.h"

//TODO:补全算子注册
namespace tensorflow {
#if CAMBRICON_MLU
#define REGISTER_MLU(T)                                           \
  REGISTER_KERNEL_BUILDER(......);
  .......
#undef REGISTER_MLU
#endif
  // #if CAMBRICON_MLUSBC

  // namespace tensorflow

//REGISTER_KERNEL_BUILDER(Name("SBC").Device(DEVICE_CPU), SBCOp<float>);
}
