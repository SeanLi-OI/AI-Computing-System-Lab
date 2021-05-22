#!/bin/bash

DATA_DIR="${TENSORFLOW_MODELS_AND_DATA}/data"

COLA=${DATA_DIR}/CoLA
MRPC=${DATA_DIR}/MRPC

export MLU_VISIBLE_DEVICES="0"
export MLU_STATIC_NODE_FUSION=false

echo
echo "=============start run bert==============="
echo

# MRPC
BERT_BASE_DIR="${TENSORFLOW_MODELS_AND_DATA}/models/uncased_L-12_H-768_A-12"
python run_classifier.py \
  --task_name=MRPC \
  --do_train=false \
  --do_eval=true \
  --do_predict=false \
  --data_dir=${MRPC} \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --bert_config_file=${BERT_BASE_DIR}/bert_config.json \
  --init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
  --max_seq_length=128 \
  --do_lower_case=True \
  --training_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epoch=3.0 \
  --output_dir="./output_dir_mrpc" 2>&1 | tee log


# COLA
#BERT_BASE_DIR="${TENSORFLOW_MODELS_AND_DATA}/models/cased_L-12_H-768_A-12"
#python run_classifier.py \
#  --task_name=COLA \
#  --do_train=false \
#  --do_eval=true \
#  --do_predict=false \
#  --data_dir=${COLA} \
#  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
#  --bert_config_file=${BERT_BASE_DIR}/bert_config.json \
#  --init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
#  --max_seq_length=128 \
#  --do_lower_case=False \
#  --training_batch_size=32 \
#  --learning_rate=2e-5 \
#  --num_train_epoch=3.0 \
#  --output_dir="./output_dir_cola" 2>&1 | tee log
