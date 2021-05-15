#!/bin/bash

cncc -S -O3 --bang-mlu-arch=MLU270 --bang-device-only ./gemm/gemm_SRAM.mlu -o ./gemm/gemm_SRAM.s
cnas -O2 --mcpu x86_64 -i ./gemm/gemm_SRAM.s -o ./gemm/gemm_SRAM.o
g++ -O2 -std=c++11   -I ./neuware/include  -I . -DHOST -c ./common/main.cpp -o ./common/main.o
g++ -O2 -std=c++11   -I ./neuware/include  -I . -DHOST -c ./common/mlu_gemm16.cpp -o ./common/mlu_gemm16.o
g++ -o ./gemm/test_int_matmul -L ./neuware/lib64 ./common/mlu_gemm16.o ./common/main.o ./gemm/gemm_SRAM.o -lcnrt -lopenblas
echo "build gemm done........"


