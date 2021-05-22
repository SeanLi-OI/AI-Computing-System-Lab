#!/bin/bash
set -e
export TF_CPP_MIN_MLU_LOG_LEVEL=0
export TF_CPP_MIN_VLOG_LEVEL=0

usage () {
    echo "Usage:"
    echo "      network                  [note]  resnext101|resnet18|resnet50_v1|resnet101_v1|resnet152_v1|resnet50_v2|"
    echo "                                       resnet101_v2|resnet152_v2|vgg16|vgg19|inception_v1|"
    echo "                                       inception_v2|inception_v3|mobileNet|mobileNet_v2|mobileNet_v3|efficientNet_b0|efficientNet_b3|"
    echo "      int8/int16               [note]  chooose int8/int16 model"
    echo "      batch_size               [note]  1/2/4/.."
    echo "      core_num                 [note]  1/4/16 core number "
    echo "      image_mode               [note]  small/big, optional(small is false)"
    echo "      core_version             [note]  MLU270/MLU220, default is MLU270"
}
if [ $# -lt 5 ]; then
    usage
    exit -1
fi
network=$1
dtype=$2
batch_size=$3
core_num=$4
image_mode=$5

if [ $# -eq 6 ]; then
  core_version=$6
else
  core_version="MLU270"
fi

core_num_or_mp="--core_num=${core_num} --inst_nosplit=false"
if [ "${dtype}" == "float16" ];then
    precision=" --precision=float32"
elif [ "${dtype}" == "int8" ];then
    precision=" --precision=int8"
elif [ "${dtype}" == "int16" ];then
    precision=" --precision=int16"
else
    echo "dtype must be float16/int8/int16"
    exit -1
fi

filepath=$(cd "$(dirname ${BASH_SOURCE[0]})";pwd)

if [[ ! ${TENSORFLOW_HOME} ]];then
  if type git > /dev/null 2>&1 && git rev-parse --is-inside-work-tree > /dev/null 2>&1;then
    TENSORFLOW_HOME=$( git rev-parse --show-toplevel )
  else
    1>&2 echo "ERROR: TENSORFLOW_HOME is not set, please set TENSORFLOW_HOME to tensorflow project root"
    exit 1
  fi
fi
modelpath="/home/cambricon/xinchao/tf_models_0227/tensorflow_models/cambricon_examples/bert_fast/output_dir_squad_v1.1"
if [ -z ${LD_LIBRARY_PATH} ]; then
    export LD_LIBRARY_PATH="${TENSORFLOW_HOME}/third_party/mlu/lib"
fi

if [ -z ${TENSORFLOW_MODEL_HOME} ]; then
    echo "The env variable TENSORFLOW_MODEL_HOME do not set !!!"
    exit -1
else
    if [ ! -d ${modelpath} ];then
        echo "The ${network} does not exit"
        exit -1
    fi
fi
# specify the path to store the offline model
tensorflow_offline_model="${TENSORFLOW_HOME}/tensorflow/cambricon_examples/demo_data/cambricon"
if [ ! -d ${tensorflow_offline_model} ];then
    mkdir ${tensorflow_offline_model}
fi
# find the model
if [[ "${image_mode}" == "small" ]]; then
    model_txt="/home/cambricon/xinchao/tf_models_0227/tensorflow_models/cambricon_examples/bert_fast/output_dir_squad_v1.1/frozen_model.txt"
    graph="--graph=/home/cambricon/xinchao/tf_models_0227/tensorflow_models/cambricon_examples/bert_fast/output_dir_squad_v1.1/frozen_model_int16.pb"
else
    model_txt="${TENSORFLOW_MODEL_HOME}/1080P_models/${network}_model.txt"
    graph="--graph=${TENSORFLOW_MODEL_HOME}/1080P_models/${network}_${dtype}.pb"
fi
# the config file and the executable program
param=${tensorflow_offline_model}/${network}/${network}_${dtype}_${core_num}_${batch_size}batch.txt
pb_to_cambricon="${TENSORFLOW_HOME}/tensorflow/cambricon_examples/tools/pb_to_cambricon"

#prepare txt and generate the offline model
pushd ${tensorflow_offline_model}
    if [ ! -d ${network} ];then
        mkdir ${network}
    fi
    pushd ${network}
        cp -f ${model_txt} ${network}_${dtype}_${core_num}_${batch_size}batch.txt
        line1=`sed -n '4p' ${param}|cut -d ',' -f 1`
        line2=${batch_size}
        line3=`sed -n '4p' ${param}|cut -d ',' -f 3-5`
        line="${line1},${line2},${line3}"
        sed -i "4c ${line}" ${network}_${dtype}_${core_num}_${batch_size}batch.txt
        sed -i "1c model_name:${network}_${dtype}_${core_num}_${batch_size}batch.cambricon" ${network}_${dtype}_${core_num}_${batch_size}batch.txt
        model=${network}_${dtype}_${core_num}_${batch_size}batch.cambricon
        model_twins=${network}_${dtype}_${core_num}_${batch_size}batch.cambricon_twins
        if [ ! -e ${model} ];then
            echo "=== Host Demo: run pb_to_cambricon ${network} ${dtype} ${core_version} ==="
            pushd ${filepath}
                ${pb_to_cambricon}/pb_to_cambricon.host \
                    ${graph} \
                    --param_file=${param} \
                    --mlu_core_version=${core_version} \
                    ${core_num_or_mp} ${precision}
                if [ -e ${model_twins} ];then
                    # delete the useless file
                    mv ${model} ${tensorflow_offline_model}/${network}/${network}_${dtype}_${core_num}cm_${batch_size}batch.cambricon
                    rm ${model_twins}
                else
                    echo "waring:${tensorflow_offline_model}/${network}/${network}_${dtype}_${core_num}cm_${batch_size}batch.cambricon is not exist.There may be problems with the process of generating the ${model}"
                    exit -1
                fi
           popd
        fi
    popd
popd
