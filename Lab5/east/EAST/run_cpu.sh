export MLU_VISIBLE_DEVICES=
MODEL_PATH="${AICSE_MODELS_MODEL_HOME}/east/east_icdar2015_resnet_v1_50_rbox"
python eval_cpu.py --checkpoint_path=${MODEL_PATH} --output_dir=./cpu_pb
