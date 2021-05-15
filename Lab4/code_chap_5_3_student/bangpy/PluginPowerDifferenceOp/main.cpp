#include <math.h>
#include <time.h>
#include "stdio.h"
#include <stdlib.h>
#include <sys/time.h>

#define DATA_COUNT 32768
#define POW_COUNT 2
int MLUPowerDifferenceOp(float* input1,float* input2, int pow, float*output, int dims_a);

int main() {
  float* input_x = (float*)malloc(DATA_COUNT * sizeof(float));
  float* input_y = (float*)malloc(DATA_COUNT * sizeof(float));
  float* output_data = (float*)malloc(DATA_COUNT * sizeof(float));
  float* output_data_cpu = (float*)malloc(DATA_COUNT * sizeof(float));
  FILE* f_input_x = fopen("./data/in_x.txt", "r");
  FILE* f_input_y = fopen("./data/in_y.txt", "r");
  FILE* f_output_data = fopen("./data/out.txt", "r");
  struct timeval tpend, tpstart;
  float err = 0.0;
  float cpu_sum = 0.0;
  float time_use = 0.0;

  if (f_input_x == NULL|| f_input_y == NULL || f_output_data == NULL) {
    printf("Open file fail!\n");
    return 0;
  }

  gettimeofday(&tpstart, NULL);
  srand((unsigned)time(NULL));
  for (int i = 0; i < DATA_COUNT; i++) {
    fscanf(f_input_x, "%f\n", &input_x[i]);
    fscanf(f_input_y, "%f\n", &input_y[i]);
    fscanf(f_output_data, "%f\n", &output_data_cpu[i]);
  }
  gettimeofday(&tpend, NULL);
  time_use = 1000000 * (tpend.tv_sec - tpstart.tv_sec)+ tpend.tv_usec - tpstart.tv_usec;
  printf("get data cost time %f ms\n", time_use/1000.0);

  gettimeofday(&tpstart, NULL);
  MLUPowerDifferenceOp(input_x,input_y,POW_COUNT,output_data,DATA_COUNT);
  gettimeofday(&tpend, NULL);
  time_use = 1000000 * (tpend.tv_sec - tpstart.tv_sec)+ tpend.tv_usec - tpstart.tv_usec;
  printf("compute data cost time %f ms\n", time_use/1000.0);
  printf("input x %f\n",input_x[0]);
  printf("input y %f\n",input_y[0]);
  printf("output data %f\n",output_data[0]);
  printf("output data %f\n",output_data[1]);
  printf("output data %f\n",output_data[2]);
  for(int i = 0; i < DATA_COUNT;++i)
  {
     err +=fabs(output_data_cpu[i] - output_data[i]) ;
     cpu_sum +=fabs(output_data_cpu[i]);
  }
  printf("err rate = %0.4f%%\n", err*100.0/cpu_sum);
  return 0;
}
