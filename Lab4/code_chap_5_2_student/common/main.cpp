/*************************************************************************
 * Copyright (C) [2019] by Cambricon, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/

#include <openblas/cblas.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>
#include "cnrt.h"
#include <cmath>
#include <cstdlib>
#include <iomanip>

using std::thread;
using std::vector;

#define COMPARE_WITH_CPU
#define ABS(a, b) (a > b ? (a - b) : (b - a))
int Mlu_gemm(int8_t *A, int8_t *B, float *C, int32_t M, int32_t N,
    int32_t K, int16_t pos1, int16_t pos2, float scale1, float scale2, float &return_time);

//  get_int8_position_scale(A, m*k, 0xFFFF, 1.98438)
void get_int8_position_scale(float *input, int size, int16_t &position, float &scale) {
  if (input == NULL || size == 0 ) {
    return;
  }
  int var = 6;
  int scale_var = std::pow(2, var + 1) - 1;
  float max_data = std::fabs(input[0]);
  // 找到输入绝对值最大值
  for (int index = 0; index < size; ++index) {
    if (std::fabs(input[index]) > max_data)
      max_data = std::fabs(input[index]);
  }
  if (max_data == 0) {
    position = 0;
    scale = 1.0;
  } else {
    position = static_cast<int16_t>(std::ceil(std::log2(max_data / scale_var)));
    scale = static_cast<float>(std::pow(2, position) * scale_var / max_data);
  }
  scale = (scale) < 1 ? 1 : (scale);
//  //printf("\n pos=%d,scale=%.2f\n",position,scale);
}

  // quanti_int8(A, quantA, (M * K), 0xFFFF, 1.98438);
void  quanti_int8(float *input,
                             int8_t *output,
                             int32_t count,
                             int16_t &pos,
                             float &scalevalue) {
  int index = 0;
  float value = 0.0;
  float fix = 0.0;

  get_int8_position_scale(input,count,pos,scalevalue);

  /***************convert float to int8 ***********************************
   *
   *             quantification use max_data and pos1
   *
   * ***********************************************************************/
  for (index = 0; index < count; ++index) {
    value = input[index];
    fix = value * scalevalue * pow(2,0-pos);
    output[index] = static_cast<int8_t>(round(fix));
  }
}

void print_info(int m,
                int n,
                int k,
                int core_num,
                float time,
                double div)
{
  std::cout << std::setw(15) << "Int8";
  std::cout << std::setw(18) << "Float16";
  std::stringstream out_input;
  out_input <<  m << "," << n << "," << k;
  std::stringstream out_time;
  out_time <<  time << " ms";
  std::cout << std::setw(15) << out_input.str()
            << std::setw(15) << core_num
            << std::setw(15) << out_time.str()
            << std::setw(15) << div
            << std::endl;
}



int testCase(int M, int N, int K) {
  /**************cpu version cblas********************/
  ////printf("\n>>>>>> TestCase M %d, K %d, N %d start <<<<<<\n", M, K, N);
  srand((unsigned int)(time(NULL)));
  const CBLAS_ORDER Order = CblasRowMajor;          // 行主序
  const CBLAS_TRANSPOSE TransA = CblasNoTrans;      // 按行展开
  const CBLAS_TRANSPOSE TransB = CblasTrans;        // 按列展开
  // int M=1;//A row c row
  // int N=16 + 0 ;//B column C column
  // int K=256 + 16;//A column B row
  float alpha = 1;
  float beta = 0;
  int lda = K;  // A column
  int ldb = K;  // B column
  int ldc = N;  // C column
  float *A = (float *)malloc(M * K * sizeof(float));
  float *B = (float *)malloc(K * N * sizeof(float));
  int8_t *quantA = (int8_t *)malloc(M * K * sizeof(int8_t));
  int8_t *quantB = (int8_t *)malloc(K * N * sizeof(int8_t));
  float *C = (float *)malloc(M * N * sizeof(float));
  float *Cmlu = (float *)malloc(M * N * sizeof(float));

  // 给A和B矩阵随机赋值
  for (int i = 0; i < M; i++)
  {
    for (int j = 0; j < K; j++)
    {
        A[i*K +j] = (i + rand()%16)/217.0;// ((i+j+1)%16 +2)/1.0
    }
  }
  int p = 0;
  for (int i = 0; i < K; i++)
  {
    for (int j = 0; j < N; j++)
    {
      B[i*N +j] = ((float)(rand()%20+3))/1003.0;//((i+j+1)%16)/17.0;//++p ;
    }
  }
    //printf("generate A & B Done.\n");
  struct timeval start;
  struct timeval end;
  float time_use;
#if 1  // TODO(zhouxiaoyong): add cpu checking
    gettimeofday(&start, NULL);
    cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    gettimeofday(&end, NULL);
    time_use = ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) /1000.0;   // 计算耗费时间，单位是ms
#endif

// 对数据进行量化处理
  int16_t pos1 = 0xFFFF;
  int16_t pos2 = 0xFFFF;
  float scale1 = 1.98438;
  float scale2 = 1.0;
  float return_time = 0.0;  
  quanti_int8(A, quantA, (M * K), pos1, scale1);
  quanti_int8(B, quantB, (K * N), pos2, scale2);
  
/***************************************************/
#define COUNT 1
  time_use = 0;
  for (int i = 0; i < COUNT; i++) {
    gettimeofday(&start, NULL);
    Mlu_gemm(quantA, quantB, Cmlu, M, N, K, pos1, pos2, scale1, scale2,return_time);
    gettimeofday(&end, NULL);
    time_use += ((end.tv_sec - start.tv_sec) * 1000000 +
                 (end.tv_usec - start.tv_usec)) /
                1000.0;
  }
  time_use /= COUNT;
  //printf("\nmlu total time use %.3f ms \n", time_use);
  //printf("A:%d x %d\n", M, K);
  //printf("B:%d x %d\n", K, N);
  //printf("C:%d x %d\n", M, N);
  //printf("pos1=%d, pos2= %d, scale1=%.2f, scale2=%.2f\n", pos1,pos2,scale1,scale2);
#ifdef COMPARE_WITH_CPU
    // //printf("mlu result:\n");
    double sum_err = 0.0;
    double sum = 0.0;
    for(int i=0;i<M;i++)
    {
       for(int j=0;j<N;j++)
       {
           if ( ABS(Cmlu[i*N+j],C[i*N+j])> 100) {
             printf("(%d,%d) %.1f cpu %.1f\n ",i,j,Cmlu[i*N+j], C[i*N+j]);
           }
           if (Cmlu[i*N+j] >65500)
           {
               printf("mlu out of range\n");
           }
           sum_err += ABS(Cmlu[i*N+j],C[i*N+j]);   
           sum += C[i*N+j];
           // //printf("err %f\n", ABS(Cmlu[i*N+j],C[i*N+j]));
       }
    }

#endif
  free(A);
  free(B);
  free(C);
  free(Cmlu);
  //printf("\nTestCase M %6d, K %6d, N %10d end , time=%4.3f ms, err= %4.5f%%\n", M, K, N,return_time,sum_err*100.0/sum);
  print_info(M,N,K,16,return_time,sum_err/sum);

  return 0;
}

int main(int argc, char* argv[]) {

  //char *str_env = std::getenv("SAVE_RESULTS");
  CNRT_CHECK(cnrtInit(0));
  cnrtDev_t dev;
  CNRT_CHECK(cnrtGetDeviceHandle(&dev, 0));
  CNRT_CHECK(cnrtSetCurrentDevice(dev));

  std::cout << "BANG C int_matmul demo:" << std::endl
            << std::setw(15) << "input data type"
            << std::setw(18) << "output data type"
            << std::setw(15) << "input M,N,K"
            << std::setw(15) << "core number"
            << std::setw(15) << "time consume"
            << std::setw(15) << "abs diff rate"
            << std::endl;

  // M , N , K
  int res = 0;
  struct timeval start;
  struct timeval end;
  float time_use;
//  res +=testCase(200,327680,256);
   // input 3 arguments for M, K, N
  int do_not_save = 0;
  if (argc == 5) {
    do_not_save = 1;
  }
  if ( argc >= 4 ) {
   int m = atoi(argv[1]);
   int n = atoi(argv[2]);
   int k = atoi(argv[3]);
   std::cout << "m = " << m << " k = " << k << " n = " << n << "\n";
   res +=testCase(m,n,k);
  }
  else {
//    std::cout << "Please enter 3 numbers for m, k, n\n\n go test :\n\n";
    res +=testCase(1,32768,256);
    res +=testCase(1,327680,200);
    res +=testCase(1,328192,256);
    res +=testCase(1,327680,512);
  }
  if (res) {
    //printf("\nFAILED: failed case num = %d\n", res);
  } else {
    //printf("\nPASSED: failed case num = %d\n", res);
  }
  cnrtDestroy();

  return 0;
}
