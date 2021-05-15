/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_CORE_KERNELS_CWISE_OP_POWER_DIFFERENCE_MLU_H_
#define TENSORFLOW_CORE_KERNELS_CWISE_OP_POWER_DIFFERENCE_MLU_H_
#if CAMBRICON_MLU
#include <string>
#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "tensorflow/core/kernels/cwise_ops.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/mlu_op_kernel.h"
#include "tensorflow/stream_executor/mlu/mlu_stream.h"
namespace tensorflow {
template <typename T>
class MLUPowerDifferenceOp : public MLUOpKernel {
 public:
  explicit MLUPowerDifferenceOp(OpKernelConstruction* ctx) :
          MLUOpKernel(ctx) {}

  void ComputeOnMLU(OpKernelContext* ctx) override {
      //输入数据处理与条件判断
      ......
      // stream调用Powerdifference接口
      OP_REQUIRES_OK(ctx, stream->PowerDifference(...));
  }
};

}  // namespace tensorflow

#endif  // CAMBRICON_MLU
#endif  // TENSORFLOW_CORE_KERNELS_CWISE_OP_SQUARED_DIFFERENCE_MLU_H_
