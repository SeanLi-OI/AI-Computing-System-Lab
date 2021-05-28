# Copyright (C) [2020] by Cambricon, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Bangpy sbc code"""
import os, shutil
import numpy as np
import bangpy
from bangpy import tcp, load_module
CHANNELS = 3
HEIGHT = 672
WIDTH = 1280
ALIGN_SIZE = 64
HWC_SPLIT = (HEIGHT*WIDTH//16)*CHANNELS
DATA_COUNT = ((CHANNELS) * (WIDTH) * (HEIGHT))

def sbc():
    def verify_bp(dtype):
        # 定义TCP容器
        bp = tcp.TCP()
        # 声明内置变量以及可变参数
        batch_num_ = bp.Var("batch_num_")
        core_id = bp.builtin_var("coreId")
        core_dim= bp.builtin_var("coreDim")
        cluster_id = bp.builtin_var("clusterId")
        task_id = bp.Scalar(dtype=bangpy.int32, name="task_id", value=cluster_id * core_dim + core_id)
        taskDim = bp.builtin_var("taskDim")
        input_ = bp.Tensor(shape=(batch_num_, 16/taskDim, taskDim, HWC_SPLIT), name="input_data_",
                           dtype=dtype, scope="global")
        output_ = bp.Tensor(shape=(batch_num_, 16/taskDim, taskDim, HWC_SPLIT), name="output_data_",
                            dtype=dtype, scope="global")
        split_sub_concat = bp.Tensor(shape=(int(HWC_SPLIT),), name="split_sub_concat",
                                     dtype=dtype, scope="nram")
        temp0 = bp.Tensor(shape=(192,), name="temp0",
                          dtype=dtype, scope="nram")
        a = bp.Scalar(dtype=bangpy.float32, name="a", value=123.68)
        b = bp.Scalar(dtype=bangpy.float32, name="b", value=116.78)
        c = bp.Scalar(dtype=bangpy.float32, name="c", value=103.94)

        core_loop = 16 // taskDim
        with bp.for_range(0, 192//3) as i:
            temp0[i*3] = a.astype(dtype, rounding="rd")
            temp0[i*3+1] = b.astype(dtype, rounding="rd")
            temp0[i*3+2] = c.astype(dtype, rounding="rd")

        with bp.for_range(0, batch_num_) as i:
            with bp.for_range(0, core_loop) as j:
                #数据拆分拷贝到NRAM
                bp.memcpy(split_sub_concat, input_[i][j][task_id])
                #subtract原语的cycle运算
                bp.subtract(split_sub_concat, split_sub_concat, temp0, cycle=True)
                #完成运算后拷贝回GDRAM
                bp.memcpy(output_[i][j][task_id], split_sub_concat)
            bp.sync_all()

        f = bp.BuildBANG(inputs=[input_, batch_num_], outputs=[output_],
                         kernel_name="SBCKernel")
        return f
    
    def check_target():
        fvec = verify_bp(bangpy.float16)
        if os.path.exists('./test'):
            shutil.rmtree('./test')
        fvec.save('./test/')
        new_line = []
        with open('./test/device.mlu', "r+") as f:
            line_num = 0
            for line in f:
                if line_num == 0:
                    new_line.append(
                        "__mlu_entry__ void SBCKernel(half* input_data_,half* output_data_, int batch_num_){\n"
                    )
                    line_num = 1
                else:
                    new_line.append(line)
        with open('./test/device.mlu', "w+") as f:
            f.writelines(new_line)
        os.system('cp ./test/device.mlu spilt_sub_concat_kernel.mlu')

    check_target()

if __name__ == "__main__":
    sbc()
