#include "post_processor.h"

namespace StyleTransfer{

PostProcessor :: PostProcessor(){
//    std::cout << "PostProcessor constructor" << std::endl;
}

void PostProcessor :: save_image(DataTransfer* DataT){

    std::vector<cv::Mat> mRGB(3);
    for(int i = 0; i < 3; i++){
        cv::Mat img(256, 256, CV_32FC1, DataT->output_data + 256 * 256 * i);
        mRGB[i] = img;
    }
    cv::Mat im(256, 256, CV_8UC3);
    cv::merge(mRGB,im);

    std::string file_name = DataT->image_name + std::string("_") + DataT->model_name + ".jpg";
    cv::imwrite(file_name, im);
    std::cout << "style transfer result file: " << file_name << std::endl;   
}

PostProcessor :: ~PostProcessor(){
//    std::cout << "PostProcessor destructor" << std::endl;
}

void PostProcessor :: run(DataTransfer* DataT){
    save_image(DataT);
}

} // namespace StyleTransfer
