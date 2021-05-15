#include "style_transfer.h"
#include <math.h>
#include <time.h>
#include "stdio.h"
#include <stdlib.h>
#include <sys/time.h>

int main(int argc, char** argv){
    // parse args
    std::string file_list = "../../images/" + std::string(argv[1]) + ".jpg";
    std::string offline_model = "../../models/offline_models/" + std::string(argv[2]) + ".cambricon";

    //creat data 
    DataTransfer* DataT =(DataTransfer*) new DataTransfer();
    DataT->image_name = argv[1];
    DataT->model_name = argv[2];
    //process image
    DataProvider *image = new DataProvider(file_list); 
    image->run(DataT);

    //running inference
    Inference *infer = new Inference(offline_model);
    infer->run(DataT);

    //postprocess image
    PostProcessor *post_process = new PostProcessor();
    post_process->run(DataT);
    
    delete DataT;
    DataT = NULL;
    delete image;
    image = NULL;
    delete infer;
    infer = NULL;
    delete post_process;
    post_process = NULL;
}
