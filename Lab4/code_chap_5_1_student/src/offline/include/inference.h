#ifndef INFERENCE_H_
#define INFERENCE_H_

#include "data_transfer.h"
#include "cnrt.h"

namespace StyleTransfer{
    class Inference{
        public:
            Inference(std::string offline_model);
            void run(DataTransfer* DataT);
        private:
            std::string offline_model_;
    }; // class Inference
}

#endif // namespace StyleTransfer
