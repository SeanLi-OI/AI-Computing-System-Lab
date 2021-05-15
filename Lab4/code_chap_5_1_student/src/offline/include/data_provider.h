#ifndef DATA_PROVIDER_H_  
#define DATA_PROVIDER_H_

#include "data_transfer.h"

namespace StyleTransfer {

    class DataProvider {
        public:
            DataProvider(std::string file_list_);
            bool get_image_file();    
            cv::Mat convert_color_space(std::string file_path);
            cv::Mat resize_image(const cv::Mat& source); 
            cv::Mat convert_float(cv::Mat img);
            cv::Mat subtract_mean(cv::Mat float_image);
            void set_mean();
            void split_image(DataTransfer* DataT);
            ~DataProvider();
            void run(DataTransfer* DataT);
        
        private:
            std::vector<std::string> image_list;
            std::string file_list;
            int image_num;
            int image_channel;
            cv::Mat mean_;
            int batch_size;

    }; // class DataProvider

}  // namespace StyleTransfer
#endif  // DATA_PROVIDER_H_

