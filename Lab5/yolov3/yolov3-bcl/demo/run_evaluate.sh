#!/bin/bash
set -e
#export TF_CPP_MIN_VLOG_LEVEL=1
#export TF_CPP_MIN_MLU_LOG_LEVEL=3
#export MLU_VISIBLE_DEVICES=""

usage () {
    echo "Usage:"
    echo "./run_evaluate.sh    "
    echo "              batch_size: batch size"
    echo "              core_num: 1"
    echo "              image_num: 1/2/.... will run image_num images"
    echo "              precision: float/int8"
    echo "              core_version: MLU100/MLU270"
    echo "./run_evaluate.sh 1 True 1 float MLU100"
}

if [ $# -lt 5 ]; then
    usage
    exit 1
fi

batch_size=$1
core_num=$2
number=$3
precision=$4
core_version=$5

if [ ${core_version} == "MLU270" ]; then
    if [ ${precision} != "int8" ]; then
        echo "core_version=MLU270, precision must be int8!"
        exit 1
    fi
fi

if [ ${precision} == "int8" ]; then
    network=yolov3_int8_bang_shape_new.pb
else
    network=yolov3.pb
fi

result_file="yolov3_${precision}_result"
rm -rf ${result_file}

# change the path of the model
MODEL_PATH="${AICSE_MODELS_MODEL_HOME}/yolov3/${network}"
echo ${MODEL_PATH}
DATASET_PATH="./data/dataset/"
COCO_DATASET_HOME="${AICSE_MODELS_DATA_HOME}/yolov3"
FILE_LIST_PATH="${COCO_DATASET_HOME}/COCO/val2017/"

cp "${DATASET_PATH}/coco_val.txt" "${DATASET_PATH}/coco_val_test.txt"
sed -i s:^0:${FILE_LIST_PATH}0:g "${DATASET_PATH}/coco_val_test.txt"

echo
echo "=== Host Demo: MLU run ${network} ==="
echo
python -u evaluate.py \
    --graph=${MODEL_PATH} \
    --core_num=${core_num}  \
    --precision=${precision} \
    --core_version=${core_version} \
    --records="${DATASET_PATH}/coco_val_test.txt" \
    --number=${number} \
    --result_path=${result_file} \
    --batch_size=${batch_size} 2>&1 | tee ${network}.log
rm -rf ./mAP/predicted ./mAP/ground-truth
cp -r ${result_file}/mAP/predicted ./mAP
cp -r ${result_file}/mAP/ground-truth ./mAP
cd ./mAP
python -u main.py
cd ../
mkdir "${result_file}/mAP/picture_result"
PICTURE_RESULT_PATH="./mAP/results/"
PICTURE_NEW_PATH="${result_file}/mAP/picture_result"
cp "${PICTURE_RESULT_PATH}/Ground-Truth Info.png" "$PICTURE_NEW_PATH/Ground-Truth Info.png"
cp "${PICTURE_RESULT_PATH}/Predicted Objects Info.png" "$PICTURE_NEW_PATH/Predicted Objects Info.png"
cp "${PICTURE_RESULT_PATH}/mAP.png" "$PICTURE_NEW_PATH/mAP.png"
cd -
