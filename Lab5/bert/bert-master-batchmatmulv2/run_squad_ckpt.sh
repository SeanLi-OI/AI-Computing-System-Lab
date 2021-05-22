#!/bin/bash

BERT_BASE_DIR="${TENSORFLOW_MODELS_DATA_HOME}/bert/uncased_L-12_H-768_A-12"
SQUAD_DIR="${TENSORFLOW_MODELS_DATA_HOME}/bert/squad"
OUTPUT_DIR="./squad_ckpt_output_dir_128"
CHECKPOINT_DIR="${TENSORFLOW_MODELS_MODEL_HOME}/bert/model.ckpt-37000"
echo
echo "=============start run bert==============="
echo

#for seq_len in 128; do
log_file="./log_file"
output_file="./result_file"
#for seq_len in 128 192 256 320 384 448 512; do
for seq_len in 128; do
for batch_size in 1; do
  python run_squad_ckpt.py \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=${CHECKPOINT_DIR} \
    --do_train=false \
    --train_file=$SQUAD_DIR/train-v1.1.json \
    --do_predict=true \
    --predict_batch_size=$batch_size \
    --predict_file=$SQUAD_DIR/dev-v1.1.json \
    --train_batch_size=4 \
    --learning_rate=3e-5 \
    --num_train_epochs=2.0 \
    --max_seq_length=$seq_len \
    --doc_stride=128 \
    --output_dir=$OUTPUT_DIR
  python squad/evaluate-v1.1.py \
      squad/dev-v1.1.json \
      $OUTPUT_DIR/predictions.json |& tee $log_file
  f0=`cat $log_file | grep -F 'exact_match' | awk -F': ' '{print $3}'`
  f1=`echo $f0 | awk -F'}' '{print $1}'`
  em0=`cat $log_file | grep -F 'exact_match' | awk -F': ' '{print $2}'`
  em=`echo $em0 | awk -F',' '{print $1}'`
  echo "$batch_size $seq_len $f1 $em" >> $output_file
done
done
