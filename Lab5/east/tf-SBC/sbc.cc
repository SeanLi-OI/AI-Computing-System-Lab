/*Copyright 2018 Cambricon*/
#if CAMBRICON_MLU

#include "tensorflow/stream_executor/mlu/mlu_api/lib_ops/mlu_lib_ops.h"
#include "tensorflow/stream_executor/mlu/mlu_api/ops/mlu_ops.h"
#include "tensorflow/stream_executor/mlu/mlu_api/tf_mlu_intf.h"

namespace stream_executor {
namespace mlu {
namespace ops {


Status MLUSBC::CreateMLUOp(std::vector<MLUTensor *> &inputs,
                            std::vector<MLUTensor *> &outputs, void *param) {
  //TODO:补齐create实现
  ......
  return Status::OK();
}

Status MLUSBC::Compute(const std::vector<void *> &inputs,
                        const std::vector<void *> &outputs, cnrtQueue_t queue) {
  //TODO:补齐compute实现
  ......
  return Status::OK();
}

}  // namespace ops
}  // namespace mlu
}  // namespace stream_executor

#endif  // CAMBRICON_MLU
