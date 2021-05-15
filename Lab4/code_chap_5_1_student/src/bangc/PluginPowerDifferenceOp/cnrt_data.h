#ifndef __CNRT_DATA_H
#define __CNRT_DATA_H

#include "cnrt.h"

typedef unsigned short half;

void cnrtConvertFloatToHalfArray(uint16_t* x, const float* y, int len) {
  for (int i = 0; i < len; i++){
    cnrtConvertFloatToHalf(x+i,y[i]);
  }
}

void cnrtConvertHalfToFloatArray(float* x, const uint16_t* y, int len) {
  for (int i = 0; i < len; i++){
    cnrtConvertHalfToFloat(x+i,y[i]);
  }
}

void cnrtConvertFloatToHalfArray(uint16_t* x, float* y, int len) {
  for (int i = 0; i < len; i++){
    cnrtConvertFloatToHalf(x+i,y[i]);
  }
}

void cnrtConvertHalfToFloatArray(float* x, uint16_t* y, int len) {
  for (int i = 0; i < len; i++){
    cnrtConvertHalfToFloat(x+i,y[i]);
  }
}


#endif
