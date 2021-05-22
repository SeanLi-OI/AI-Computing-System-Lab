#!/bin/bash
mkdir -p ./models_and_data/models
mkdir -p ./models_and_data/data
models_path="./models_and_data/models"
data_path="./models_and_data/data"
# download the models
wget -P ${models_path} https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
echo "Success download the bert uncase model!"
wget -P ${models_path} https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
echo "Success download the bert case model!"
unzip -q -d ${models_path} ./models_and_data/models/uncased_L-12_H-768_A-12.zip
unzip -q -d ${models_path} ./models_and_data/models/cased_L-12_H-768_A-12.zip
# download the datasets
python download_glue_data.py --data_dir ${data_path}
# copy finetuned models
# put env variable
export TENSORFLOW_MODELS_AND_DATA="${PWD}/models_and_data"
