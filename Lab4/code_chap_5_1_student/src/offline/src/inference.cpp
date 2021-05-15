#include "inference.h"
#include "cnrt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stdlib.h"
#include <sys/time.h>
#include <time.h>

namespace StyleTransfer{

Inference :: Inference(std::string offline_model){
    offline_model_ = offline_model;
}

void Inference :: run(DataTransfer* DataT){
    // TODO:load model

    // TODO:set current device

    // TODO:load extract function

    // TODO:prepare data on cpu

    // TODO:allocate I/O data memory on MLU

    // TODO:prepare input buffer

    // TODO:malloc cpu memory

    // TODO:malloc mlu memory

    // TODO:prepare output buffer

    // TODO:malloc cpu memory

    // TODO:malloc mlu memory

    // TODO:setup runtime ctx

    // TODO:bind device

    // TODO:compute offline

    // TODO:free memory spac
}

} // namespace StyleTransfer
