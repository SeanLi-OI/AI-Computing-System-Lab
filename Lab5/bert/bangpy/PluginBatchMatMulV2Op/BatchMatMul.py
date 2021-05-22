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
# pylint: disable=missing-docstring, invalid-name
import os, shutil, functools
from functools import reduce
import numpy as np
import bangpy
from bangpy import tcp, load_module, int8, int16, int32, float16, float32

NRAM_BUF_REMAIN = 1024 + 128
NRAM_BUF_SIZE = 512 * 512 - NRAM_BUF_REMAIN
SRAM_BUF_SIZE = 2000 * 1024
WRAM_BUF_SIZE = 480 * 1024
IT = float16
FT = float16

def DIV_UP(x, y):
    return (x + y - 1) / y

class BatchMatMulV2Kernel_MLU270():
    def __init__(self):
        self.name = "BatchMatMulV2Kernel_MLU270_half"
        self.tcp = tcp.TCP()
        self.coreId = self.tcp.builtin_var("coreId")
        self.coreDim = self.tcp.builtin_var("coreDim")
        self.clusterDim = self.tcp.builtin_var("clusterDim")
        self.clusterId = self.tcp.builtin_var("clusterId")
        self.dim_0 = self.tcp.Var("dim_0")
        self.dim_1 = self.tcp.Var("dim_1")
        self.m = self.tcp.Var("m")
        self.n = self.tcp.Var("n")
        self.k = self.tcp.Var("k")
        self.scale_0 = self.tcp.Var("scale_0", IT)
        self.pos_0 = self.tcp.Var("pos_0")
        self.scale_1 = self.tcp.Var("scale_1", IT)
        self.pos_1 = self.tcp.Var("pos_1")
        self.block_m = self.tcp.Scalar(dtype=int32, name="m_block", value=0)
        self.block_n = self.tcp.Scalar(dtype=int32, name="n_block", value=0)
        self.block_k = self.tcp.Scalar(dtype=int32, name="k_block", value=0)
        self.bk_align = self.tcp.Scalar(dtype=int32, name="bk_align", value=0)
        self.scale = self.tcp.Scalar(dtype=float32, name='scale',
                                     value=(1.0 / (self.scale_0*self.scale_1)).astype(float32))
        self.pos = self.tcp.Scalar(dtype=int32, name='pos',
                                   value=self.pos_0 + self.pos_1)
        self.left = self.tcp.Tensor(shape=(self.dim_0, self.dim_1, self.m, self.k),
                                         name='left_ddr', dtype=IT, scope="global")
        self.right = self.tcp.Tensor(shape=(self.dim_0, self.dim_1, self.n, self.k),
                                          name='right_ddr', dtype=IT, scope="global")
        self.dst = self.tcp.Tensor(shape=(self.dim_0, self.dim_1, self.m, self.n),
                                        name='dst_ddr', dtype=FT, scope="global")
        self.nram_buf = self.tcp.Tensor(shape=(NRAM_BUF_SIZE // 2,), name='nram_buf',
                                        dtype=float16, scope="nram")
        self.wram_buf = self.tcp.Tensor(shape=(WRAM_BUF_SIZE // 2,), name='wram_buf',
                                        dtype=float16, scope="wram")    
        self.sram_buf = self.tcp.Tensor(shape=(SRAM_BUF_SIZE // 2,), name='sram_buf',
                                        dtype=float16, scope="sram")
        def gen_map(m, n, k, k_align, buf):            
            shape_map_ = [[m, n], [m, k_align],[n,k], [m, k]]
            size_map_ = [reduce(lambda x,y:x *y, i) for i in shape_map_]
            offset_map_ = list(map(lambda x: sum(size_map_[:x + 1]), range(len(size_map_))))
            res_ = buf[:offset_map_[0]].reshape(shape_map_[0])
            in_ = buf[offset_map_[0]:offset_map_[0] + size_map_[3]].reshape(shape_map_[3])
            weight_ = buf[offset_map_[1]:offset_map_[2]].reshape(shape_map_[2])
            temp_ = buf[offset_map_[0]:offset_map_[0] + size_map_[0]].reshape(shape_map_[0])
            weight_reshape_ = buf[offset_map_[2]:offset_map_[2] + size_map_[2]].reshape(shape_map_[2])
            return res_, in_, weight_, temp_,  weight_reshape_

        self.res_nram,self.in_nram,self.weight_nram,self.temp_nram,self.weight_reshape_nram = \
            gen_map(self.block_m, self.block_n, self.block_k, self.bk_align, self.nram_buf)

        self.res_sram,self.in_sram,self.weight_sram,self.temp_sram, _ = \
            gen_map(self.block_m * 2, self.block_n * 2, self.block_k * 2, self.bk_align * 2, self.sram_buf)

        self.conv_res_shape = [1, self.block_m, 1, self.block_n]
        self.conv_in_shape = [1, self.block_m, 1, self.block_k]
        self.conv_weight_shape = [self.block_n, 1, 1, self.block_k]

    def min(self, c, a, b):
        with self.tcp.if_scope(...):
            c.assign(...)
        with self.tcp....:
            c.assign(..)

    def init_block_size(self, block_m, block_n, block_k, bk_align):
        nram_use = self.tcp.Scalar(dtype=int32, name="nram_use", value=0)
        wram_use = self.tcp.Scalar(dtype=int32, name="wram_use", value=0)
        sram_use = self.tcp.Scalar(dtype=int32, name="sram_use", value=0)
        block_temp = self.tcp.Scalar(dtype=int32, name="block_temp", value=0)
        m_align_size = k_align_size = n_align_size = 64 * self.coreDim / 2
        with self.tcp.for_range(0, DIV_UP(DIV_UP(self.m, self.clusterDim), m_align_size)) as m_:
            with self.tcp.for_range(0, DIV_UP(self.n, n_align_size)) as n_:
                with self.tcp.for_range(0, DIV_UP(self.k, k_align_size)) as k_:
                    with self.tcp.if_scope(k_ <= n_):
                        nram_use.assign((m_ * n_ + n_ * m_ + 2 * n_ * k_) * 64 * 64 * IT.bytes)
                        sram_use.assign((m_ * n_ + k_ * n_ + m_ * n_) * 128 * 128 * IT.bytes)
                    with self.tcp.else_scope():
                        nram_use.assign((m_ * n_ + k_ * m_ + 2 * n_ * k_) * 64 * 64 * IT.bytes)
                        sram_use.assign((m_ * k_ + k_ * n_ + m_ * n_) * 128 * 128 * IT.bytes)                  
                    wram_use.assign((k_ * n_) * 64 * 64 * IT.bytes)
                    with self.tcp.if_scope(tcp.all(nram_use <= NRAM_BUF_SIZE,
                                                   wram_use <= WRAM_BUF_SIZE,
                                                   sram_use <= SRAM_BUF_SIZE,
                                                   nram_use >= block_temp)):
                        block_m.assign((m_ + 1) * 64)
                        block_n.assign((n_ + 1) * 64)
                        block_k.assign((k_ + 1) * 64)
                        block_temp.assign(nram_use)
        with self.tcp.if_scope(block_k <= block_n):
            bk_align.assign(block_n)
        with self.tcp.else_scope():
            bk_align.assign(block_k)

    def load_nram(self, nram, sram, dim0, dim1, scale,  pos):
        shape = nram.shape
        self.tcp.memcpy(nram, sram[...])          
        self.tcp.sync_cluster()
        self.tcp.multiply(...)
        self.tcp.type_convert(nram.reinterpret_cast(...), ...)

    def matmul_step(self, res_nram, in_nram, weight_nram, weight_reshape_nram,
                    in_sram, weight_sram,  weight_wram,
                    in_dim0, in_dim1, weight_dim0, weight_dim1):
        self.load_nram(...)
        self.load_nram(...)
        self.tcp.reshape_filter(...)
        self.tcp.memcpy(...)        
        self.tcp.conv(...)
        self.tcp.multiply(...)

    def int_matmul_wrap(self, dst, left, right,
                        res_nram, in_nram, weight_nram, temp_nram, weight_reshape_nram,
                        res_sram, in_sram, weight_sram, temp_sram,
                        weight_wram, flag):
        dim_0 = self.coreId / 2
        dim_1 = self.coreId % 2
        res_shape = res_nram.shape
        #Cannon has two steps:
        #step0:                 #step1:
        #0,0 * 0,0, 0,1 * 1,1   #0,1 * 0,1, 0,0 * 1,0
        #1,0 * 0,0, 1,1 * 1,1   #1,1 * 0,1, 1,0 * 1,0
        #copy m,k n,k from gdram to sram
        #m,k
        with self.tcp.if_scope(self.coreId == 0x80):
            self.tcp.assign(...)
            self.tcp.memcpy(...)
            self.tcp.memcpy(...)  
        self.tcp.sync_cluster()
        for i in range(2):
            with self.tcp.if_scope(self.coreId%2 == 0):
                self.matmul_step(...)
            with self.tcp.else_scope():
                self.matmul_step(...)
            if i==0:
                self.tcp.memcpy(...)
                self.tcp.sync_cluster()
        self.tcp.memcpy(...)
        self.tcp.sync_cluster()
        self.tcp.add(...)
        with self.tcp.if_scope(flag != 0):
            with self.tcp.if_scope(self.coreId == 0x80):
                self.tcp.memcpy(temp_sram[...], dst)
            self.tcp.sync_cluster()
            self.tcp.memcpy(temp_nram, temp_sram[...])        
            self.tcp.sync_cluster()
            self.tcp.add(...)
        # self.tcp.print(res_nram[0][0])
        self.tcp.memcpy(res_sram[...], res_nram)
        self.tcp.sync_cluster()
        with self.tcp.if_scope(self.coreId == 0x80):
            self.tcp.memcpy(dst, res_sram[...])
        self.tcp.sync_cluster()

    def compute_body(self):
        m_start = self.tcp.Scalar(dtype=int32, name='m_start')
        real_m = self.tcp.Scalar(dtype=int32, name='real_m')
        real_m_in = self.tcp.Scalar(dtype=int32, name='real_m_in')
        real_n = self.tcp.Scalar(dtype=int32, name='real_n')
        real_k = self.tcp.Scalar(dtype=int32, name='real_k')
        self.init_block_size(self.block_m, self.block_n, self.block_k, self.bk_align)
        with self.tcp.for_range(0, self.dim_0, name='cur_dim0') as cur_dim0:
            with self.tcp.for_range(0, self.dim_1, name='cur_dim1') as cur_dim1:
                with self.tcp.if_scope(self.m % self.clusterDim > self.clusterId):
                    m_start.assign(self.clusterId * (self.m / self.clusterDim) + self.clusterId)
                    real_m.assign(self.m / self.clusterDim + 1)
                with self.tcp.else_scope():
                    m_start.assign(self.clusterId * (self.m / self.clusterDim) + self.m % self.clusterDim)
                    real_m.assign(self.m / self.clusterDim)
                with self.tcp.for_range(0, DIV_UP(real_m, self.block_m  * 2), name="bm") as bm:
                    with self.tcp.for_range(0, DIV_UP(self.n, self.block_n  * 2), name="bn") as bn:
                        with self.tcp.for_range(0, DIV_UP(self.k, self.block_k * 2), name="bk") as bk:
                            self.min(real_m_in, real_m - bm * self.block_m * 2, self.block_m * 2)
                            self.min(real_n, self.n - bn * self.block_n * 2, self.block_n * 2)
                            self.min(real_k, self.k - bk * self.block_k * 2, self.block_k * 2)
                            m_start += bm*self.block_m*2
                            n_start = bn*self.block_n*2
                            k_start = bk*self.block_k*2
                            # self.tcp.print(self.block_m)
                            # self.tcp.print(self.block_n)
                            # self.tcp.print(self.block_k)
                            self.int_matmul_wrap(
                                #dst tensor shape in each loop:[1, 1, real_m, real_n]
                                self.dst[...],
                                #left tensor shape in each loop:[1, 1, real_m, real_k]
                                self.left[...],
                                #right tensor shape in each loop:[1, 1, real_n, real_k]
                                self.right[...],
                                self.res_nram, self.in_nram, self.weight_nram, self.temp_nram, self.weight_reshape_nram,
                                self.res_sram, self.in_sram, self.weight_sram, self.temp_sram,
                                self.wram_buf[:self.block_n * self.block_k].reshape([self.block_n, self.block_k]), bk)

        return self.tcp.BuildBANG(inputs=[self.right, self.left,
                                  self.dim_0, self.dim_1, self.m, self.n, self.k,
                                  self.scale_0, self.pos_0, self.scale_1, self.pos_1],
                                  outputs=[self.dst],
                                  kernel_name=self.name)
def verify_test():
    try:
        f = BatchMatMulV2Kernel_MLU270().compute_body()
        if os.path.exists('./test'):
            shutil.rmtree('./test')
        f.save('./test/')
    finally:
        new_line = []
        # with open('./test/device.mlu', "r+") as f:
        with open('/root/.cache/tvm/bang_mlu_0.mlu', "r+") as f:
            line_num = 0
            for line in f:
                if line_num == 0:
                    new_line.append(
                        "__mlu_entry__ void BatchMatMulV2Kernel_MLU270_half(half *left_ddr, half *right_ddr, half* dst_ddr, int dim_0,"+\
                                        "int dim_1,int m,int n,int k,float scale_0, int pos_0,float scale_1,int pos_1) {\n"
                    )
                    line_num = 1
                else:
                    line = line.replace("\"SRAM2SRAM\"", "SRAM2SRAM")
                    new_line.append(line)
        with open('/root/.cache/tvm/bang_mlu_0.mlu', "w+") as f:
            f.writelines(new_line)
        os.system('cp /root/.cache/tvm/bang_mlu_0.mlu /home/Cambricon-CNPlugin-MLU270_test/pluginops/PluginBatchMatMulV2Op/plugin_batch_matmul_v2_kernel.mlu')
                
if __name__ == "__main__":
    verify_test()

