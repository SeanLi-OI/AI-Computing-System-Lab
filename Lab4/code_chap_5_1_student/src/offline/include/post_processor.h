#ifndef POST_PROCESSOR_H_
#define POST_PROCESSOR_H_

#include "data_transfer.h"

namespace StyleTransfer{

    class PostProcessor{
        public:
            PostProcessor();
            void save_image(DataTransfer* DataT);
            ~PostProcessor();
            void run(DataTransfer* DataT);
    }; // class PostProcessor 

}

#endif // namespace StyleTransfer
