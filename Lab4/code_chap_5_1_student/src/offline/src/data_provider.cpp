#include "data_provider.h"

namespace StyleTransfer{

DataProvider :: DataProvider(std::string file_list_){
    file_list = file_list_;
    image_channel = 3;
    batch_size = 1;
    image_num = 0;
    set_mean();
}

bool DataProvider :: get_image_file(){
    image_list.push_back(file_list);
    return true;
}

cv::Mat DataProvider :: convert_color_space(std::string file_path){
    cv::Mat sample;
    cv::Mat img = cv::imread(file_path, -1);
    if(img.channels() == 3 && image_channel == 1){
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    }else if(img.channels() == 4 && image_channel == 1){
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    }else if(img.channels() == 4 && image_channel == 3 ){
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    }else if(img.channels() == 3 && image_channel == 3){
        sample = img;
    }else{
        sample = img;
    }
    return sample;
}

cv::Mat DataProvider :: resize_image(const cv::Mat& source){
    cv::Mat sample_resized;
    cv::Mat sample;

    sample = source;
    cv::Size expected_size(256,256);
    if(sample.size() != expected_size){
        cv::resize(sample, sample_resized, cv::Size(256,256));
    }else{
        sample_resized = sample;
    }
    return sample_resized;
}

cv::Mat DataProvider :: convert_float(cv::Mat img){
    cv::Mat float_img;
    if(img.channels() == 3){
        img.convertTo(float_img, CV_32FC3);
    }else if(img.channels() == 4){
        img.convertTo(float_img, CV_32FC4);
    }else{
        img.convertTo(float_img, CV_32FC1);
    }
    return float_img;
}

cv::Mat DataProvider :: subtract_mean(cv::Mat float_image){
    cv::Mat subtracted;
    cv::subtract(float_image, mean_, subtracted);
    return subtracted;
}

void DataProvider :: set_mean(){
    float mean_value[3] = {
        0.0,
        0.0,
        0.0,
    };
    cv::Mat mean(256, 256, CV_32FC3, cv::Scalar(mean_value[0], mean_value[1], mean_value[2]));
    mean_ = mean;
}

void DataProvider :: split_image(DataTransfer* DataT){
    DataT->input_data = reinterpret_cast<float*>(malloc(sizeof(float) * 256*256*3*batch_size));
    float *data_tmp = DataT->input_data;
    for(int i = 0; i < batch_size; i++){
        DataT->split_images.push_back(std::vector<cv::Mat>());
        for(int j = 0; j < 3; j++){
            cv::Mat img(256, 256, CV_32FC1, data_tmp);
            DataT->split_images[i].push_back(img);
            data_tmp += 256*256;
        }
        cv::split(DataT->image_processed[i], DataT->split_images[i]);
    }
}

DataProvider :: ~DataProvider(){
//   std::cout << "DataProvider destructor" << std::endl;
}

void DataProvider :: run(DataTransfer* DataT){
    for(int i = 0; i < batch_size; i++){
        get_image_file();
        std::string img_path= image_list[i];
        cv::Mat img_colored = convert_color_space(img_path);
        cv::Mat img_resized = resize_image(img_colored);
        cv::Mat img_floated = convert_float(img_resized);
        DataT->image_processed.push_back(img_floated);
    }
    split_image(DataT);
}

} // namespace StyleTransfer
