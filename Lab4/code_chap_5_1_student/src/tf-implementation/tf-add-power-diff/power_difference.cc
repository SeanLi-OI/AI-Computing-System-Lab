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

  lib::MLUTensorUtil input1_util(input1);
  lib::MLUTensorUtil input2_util(input2);
  lib::MLUTensorUtil output_util(output);

  int output_dims = output_util.dims();
  int len = 1;
  std::vector<int> output_shape(output_dims);
  for (int i = 0; i < output_dims; ++i) {
    output_shape[i] = output_util.dim_size(i);
    len *= output_shape[i];
  }

  MLUTensor* final_input1 = input1;
  MLUTensor* final_input2 = input2;

  struct OpIndex* op_index = (struct OpIndex*)std::malloc(
                                            sizeof(struct OpIndex));
  op_index->broadcast_1_index = INVALID_INDEX;
  op_index->broadcast_2_index = INVALID_INDEX;

  int idx = INVALID_INDEX;

  if (!output_util.IsSameSize(input1_util)) {
    MLUTensor* temp_1;
    TF_STATUS_CHECK(lib::CreateMLUTensor(&temp_1, MLU_TENSOR,
        input1_util.dtype(), output_shape));

    MLULOG(3) << "CreateBroadcastOp, broadcast1 input: "
              << lib::MLUTensorUtil(input1).DebugString()
              << ", broadcast1 output: "
              << lib::MLUTensorUtil(temp_1).DebugString()
              << ", XINCHAO power_c: "
              << power_c;

    MLUBaseOp* broadcast_op_ptr_1;
    TF_STATUS_CHECK(lib::CreateBroadcastOp(&broadcast_op_ptr_1,
        input1, temp_1));

    base_ops_.push_back(broadcast_op_ptr_1);
    intmd_tensors_.push_back(temp_1);
    final_input1 = temp_1;
    op_index->broadcast_1_index = ++idx;
  }

  if (!output_util.IsSameSize(input2_util)) {
    MLUTensor* temp_2;
    TF_STATUS_CHECK(lib::CreateMLUTensor(&temp_2, MLU_TENSOR,
        input2_util.dtype(), output_shape));

    MLULOG(3) << "CreateBroadcastOp, broadcast2 input: "
              << lib::MLUTensorUtil(input2).DebugString()
              << ", broadcast2 output: "
              << lib::MLUTensorUtil(temp_2).DebugString();

    MLUBaseOp* broadcast_op_ptr_2;
    TF_STATUS_CHECK(lib::CreateBroadcastOp(&broadcast_op_ptr_2,
        input2, temp_2));

    base_ops_.push_back(broadcast_op_ptr_2);
    intmd_tensors_.push_back(temp_2);
    final_input2 = temp_2;
    op_index->broadcast_2_index = ++idx;
  }

  MLULOG(3) << "CreatePowerDifferenceOp, input1: "
            << lib::MLUTensorUtil(final_input1).DebugString()
            << ", input2: " << lib::MLUTensorUtil(final_input2).DebugString()
            << ", input3: " << power_c
            << ", output: " << lib::MLUTensorUtil(output).DebugString();

  TF_STATUS_CHECK(lib::CreatePowerDifferenceOp(&power_difference_op_ptr, final_input1, final_input2, power_c, output, len));

  base_ops_.push_back(power_difference_op_ptr);
  extra_ = static_cast<void*>(op_index);

  return Status::OK();
}

Status MLUPowerDifference::Compute(const std::vector<void *> &inputs,
                        const std::vector<void *> &outputs, cnrtQueue_t queue) {
  
  void *input1 = inputs.at(0);
  void *input2 = inputs.at(1);
  void *output = outputs.at(0);
  void* broadcast_1_addr;
  void* broadcast_2_addr;

  struct OpIndex* op_index = static_cast<struct OpIndex*>(extra_);

  if (op_index->broadcast_1_index != INVALID_INDEX) {
    MLUBaseOp* broadcast_op_ptr_1 = base_ops_.at(op_index->broadcast_1_index);
    size_t broadcast_1_size;
    cnmlGetTensorSize_V2(intmd_tensors_.at(op_index->broadcast_1_index),
                        &broadcast_1_size);
    cnrtMalloc(&broadcast_1_addr, broadcast_1_size);

    lib::ComputeBroadcastOp(broadcast_op_ptr_1, queue, input1,
                      broadcast_1_addr);
  } else {
    broadcast_1_addr = input1;
  }

  if (op_index->broadcast_2_index != INVALID_INDEX) {
    MLUBaseOp* broadcast_op_ptr_2 = base_ops_.at(op_index->broadcast_2_index);
    size_t broadcast_2_size;
    cnmlGetTensorSize_V2(intmd_tensors_.at(op_index->broadcast_2_index),
                        &broadcast_2_size);
    cnrtMalloc(&broadcast_2_addr, broadcast_2_size);

    lib::ComputeBroadcastOp(broadcast_op_ptr_2, queue, input2,
                      broadcast_2_addr);
  } else {
    broadcast_2_addr = input2;
  }

  MLUBaseOp *power_difference_op = base_ops_.at(base_ops_.size() - 1);
  
  TF_STATUS_CHECK(lib::ComputePowerDifferenceOp(power_difference_op, queue, broadcast_1_addr, broadcast_2_addr, output));

  TF_CNRT_CHECK(cnrtSyncQueue(queue));

  if (op_index->broadcast_1_index != INVALID_INDEX) {
    cnrtFree(broadcast_1_addr);
  }
  if (op_index->broadcast_2_index != INVALID_INDEX) {
    cnrtFree(broadcast_2_addr);
  }

  return Status::OK();
}

}  // namespace ops
}  // namespace mlu
}  // namespace stream_executor

#endif  // CAMBRICON_MLU
