#ifndef DATA_TRANSFER_H_
#define DATA_TRANSFER_H_

#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <stdio.h>

namespace StyleTransfer{

typedef struct Data{

    std::vector<cv::Mat> image_processed;
    std::vector<std::vector<cv::Mat> > split_images;

    float* input_data;
    float* output_data;
    void** mlu_input;
    void** mlu_output;
    int input_num;
    int output_num;
    int in_count;
    int in_size;
    int out_count;
    int out_size;
    std::string image_name;
    std::string model_name;

    ~Data(){
//        std::cout << "DataTransfer destructor" << std::endl;
        free(input_data);
        free(output_data);
    }

}DataTransfer; // struct Data

}

#endif // DATA_TRANSFER_H_
