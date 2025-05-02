#!/bin/bash
# BASE_MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
# BASE_MODEL_NAME="qwen/qwen2.5-7b-instruct"
# BASE_MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
# BASE_MODEL_NAME="qwen/qwen2.5-vl-32B-instruct"
# BASE_MODEL_NAME="qwen/qwen2-audio-7b-instruct"
# BASE_MODEL_NAME="deepseek-ai/deepseek-coder-33b-instruct"
BASE_MODEL_NAME="google/gemma-3-27b-it"

echo "----------------------- FP16 -----------------------"
sudo bash ../drop_cache.sh
python3 baseline3_safetensors.py --coding_type encode --model_name ${BASE_MODEL_NAME}-fp16/
sudo bash ../drop_cache.sh
python3 baseline3_safetensors.py --coding_type decode --model_name ${BASE_MODEL_NAME}-fp16/

echo "----------------------- BF16 -----------------------"
sudo bash ../drop_cache.sh
python3 baseline3_safetensors.py --coding_type encode --model_name ${BASE_MODEL_NAME}-bf16/
sudo bash ../drop_cache.sh
python3 baseline3_safetensors.py --coding_type decode --model_name ${BASE_MODEL_NAME}-bf16/

echo "----------------------- INT8 (FP16) -----------------------"
sudo bash ../drop_cache.sh
python3 baseline3_safetensors.py --coding_type encode --model_name ${BASE_MODEL_NAME}-fp16-int8/
sudo bash ../drop_cache.sh
python3 baseline3_safetensors.py --coding_type decode --model_name ${BASE_MODEL_NAME}-fp16-int8/

echo "----------------------- INT8 (BF16) -----------------------"
sudo bash ../drop_cache.sh
python3 baseline3_safetensors.py --coding_type encode --model_name ${BASE_MODEL_NAME}-bf16-int8/
sudo bash ../drop_cache.sh
python3 baseline3_safetensors.py --coding_type decode --model_name ${BASE_MODEL_NAME}-bf16-int8/
