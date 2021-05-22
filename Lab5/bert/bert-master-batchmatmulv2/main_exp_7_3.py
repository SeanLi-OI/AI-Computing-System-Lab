import os 
import sys
from squad_output_dir_128_small.inference_pb_demo import inference_pb 

def main():
    os.system('./install_stu_tensorflow.sh')
    print('---------------------------')
    os.system('./run_aicse.sh')
    print('---------------------------')
    model = sys.argv[1]
    inference_pb(model)


if __name__ == '__main__':
    main()
