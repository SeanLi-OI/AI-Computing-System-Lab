
#include <stdlib.h>
#include "stdio.h"
#include "time.h"
#define DATA_COUNT 32768
int main()
{
  FILE* f_input_data = fopen("../data/pow_y.txt", "wb");

  srand((unsigned int)(time(NULL)));
  float* input_data = (float*)malloc(DATA_COUNT * sizeof(float));
  for (int i = 0; i < DATA_COUNT; i++) {
     input_data[i] =  (rand()%10+3.0)/5.0;
     fprintf(f_input_data, "%f\n", input_data[i]);
  }

  return 0;
}

