#!/bin/bash

usage () {
    echo "excution mode:"
    echo "      auto: quantify model and evaluate, choose it for the first time "
    echo "      evaluate: evaluate directly if you have a int model already"
}

if [ $# -lt 1 ]; then
    usage
    exit 1
fi

if [ $1 != "freeze" ] && [ $1 != "quantify" ] && [ $1 != "evaluate" ] && [ $1 != "auto" ] && [ $1 != "accuracy"]; then
    usage
    exit 1
fi

run_mode=$1

BERT_BASE_DIR="${AICSE_MODELS_DATA_HOME}/bert/uncased_L-12_H-768_A-12"
#CHECKPOINT_DIR="./model.ckpt-37000"
CHECKPOINT_DIR="${AICSE_MODELS_MODEL_HOME}/bert/model.ckpt-37000"
OUTPUT_DIR="./output_dir_squad_v1.1"
BATCH_SIZE=1
SQE_LEN=128
echo $BERT_BASE_DIR
echo $CHECKPOINT_DIR

export MLU_STATIC_NODE_FUSION=true
export MLU_OP_CACHE=true

echo
echo "=============start run bert==============="
echo

if [ ${run_mode} == "freeze" ] || [ ${run_mode} == "auto" ]; then
    # save frozen graph
    echo
    echo "=============freeze begin==============="
    echo
    python run_squad.py \
      --vocab_file=${BERT_BASE_DIR}/vocab.txt \
      --bert_config_file=${BERT_BASE_DIR}/bert_config.json \
      --predict_batch_size=${BATCH_SIZE} \
      --max_seq_length=${SQE_LEN} \
      --hidden_size=768 \
      --init_checkpoint=${CHECKPOINT_DIR} \
      --output_dir=${OUTPUT_DIR} \
      --export_frozen_graph=true \
      --do_predict=false \
      --predict_file=${AICSE_MODELS_DATA_HOME}/bert/squad/dev-v1.1.json
    echo "=============freeze end==============="
fi

if [ ${run_mode} == "quantify" ] || [ ${run_mode} == "auto" ]; then
    echo
    echo "=============quantify begin==============="
    echo
    # quantify pb
    python fppb_to_intpb.py Bert_int16.ini
    echo "=============quantify end==============="
fi

if [ ${run_mode} == "evaluate" ] || [ ${run_mode} == "auto" ]; then
    # evaluate on MLU
    echo
    echo "=============evaluate begin==============="
    echo
    python run_squad.py \
      --vocab_file=${BERT_BASE_DIR}/vocab.txt \
      --bert_config_file=${BERT_BASE_DIR}/bert_config.json \
      --predict_batch_size=${BATCH_SIZE} \
      --max_seq_length=${SQE_LEN} \
      --hidden_size=768 \
      --output_dir=${OUTPUT_DIR} \
      --export_frozen_graph=false \
      --do_predict=true \
      --predict_file=${AICSE_MODELS_DATA_HOME}/bert/squad/dev-v1.1.json
    echo "=============evaluate end==============="
fi

if [ ${run_mode} == "accuracy" ] || [ ${run_mode} == "auto" ]; then
    echo "=============show accuracy==============="
    python evaluate-v1.1.py \
        ${AICSE_MODELS_DATA_HOME}/bert/squad/dev-v1.1.json \
        $OUTPUT_DIR/predictions.json
fi
