/*Copyright 2018 Cambricon*/
#if CAMBRICON_MLU

#include "tensorflow/stream_executor/mlu/mlu_api/lib_ops/mlu_lib_ops.h"
#include "tensorflow/stream_executor/mlu/mlu_api/ops/mlu_ops.h"
#include "tensorflow/stream_executor/mlu/mlu_api/tf_mlu_intf.h"

#define INVALID_INDEX -1

namespace stream_executor {
namespace mlu {
namespace ops {


struct OpIndex {
  int broadcast_1_index = INVALID_INDEX;
  int broadcast_2_index = INVALID_INDEX;
};

Status MLUPowerDifference::CreateMLUOp(std::vector<MLUTensor *> &inputs,
                            std::vector<MLUTensor *> &outputs, void *param) {
  TF_PARAMS_CHECK(inputs.size() > 1, "Missing input");
  TF_PARAMS_CHECK(outputs.size() > 0, "Missing output");

  MLUBaseOp *power_difference_op_ptr = nullptr;
  MLUTensor *input1 = inputs.at(0);
  MLUTensor *input2 = inputs.at(1);
  int power_c = *((int*)param);
  MLUTensor *output = outputs.at(0);

  ......

  TF_STATUS_CHECK(lib::CreatePowerDifferenceOp(......));

  ......

  return Status::OK();
}

Status MLUPowerDifference::Compute(const std::vector<void *> &inputs,
                        const std::vector<void *> &outputs, cnrtQueue_t queue) {
  
  .......
  
  TF_STATUS_CHECK(lib::ComputePowerDifferenceOp(...));

  .......
  return Status::OK();
}

}  // namespace ops
}  // namespace mlu
}  // namespace stream_executor

#endif  // CAMBRICON_MLU
