#!/bin/bash

# Check if model name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_name>"
    echo "Example: $0 meta-llama/Llama-3.1-8B-Instruct"
    echo ""
    echo "Models used in the paper:"
    echo "  - meta-llama/Llama-3.1-8B-Instruct"
    echo "  - qwen/qwen2.5-7b-instruct"
    echo "  - mistralai/Mistral-7B-Instruct-v0.3"
    echo "  - qwen/qwen2.5-vl-32B-instruct"
    echo "  - qwen/qwen2-audio-7b-instruct"
    echo "  - deepseek-ai/deepseek-coder-33b-instruct"
    echo "  - google/gemma-3-27b-it"
    exit 1
fi

BASE_MODEL_NAME="$1"
echo "Using model: $BASE_MODEL_NAME"

echo "----------------------- FP16 -----------------------"
sudo bash ../drop_cache.sh
python3 baseline3_safetensors.py --coding_type encode --model_name "${BASE_MODEL_NAME}" --dtype fp16
sudo bash ../drop_cache.sh
python3 baseline3_safetensors.py --coding_type decode --model_name "${BASE_MODEL_NAME}" --dtype fp16

echo "----------------------- BF16 -----------------------"
sudo bash ../drop_cache.sh
python3 baseline3_safetensors.py --coding_type encode --model_name "${BASE_MODEL_NAME}" --dtype bf16
sudo bash ../drop_cache.sh
python3 baseline3_safetensors.py --coding_type decode --model_name "${BASE_MODEL_NAME}" --dtype bf16

echo "----------------------- INT8 (FP16) -----------------------"
sudo bash ../drop_cache.sh
python3 baseline3_safetensors.py --coding_type encode --model_name "${BASE_MODEL_NAME}" --dtype fp16 --int8
sudo bash ../drop_cache.sh
python3 baseline3_safetensors.py --coding_type decode --model_name "${BASE_MODEL_NAME}" --dtype fp16 --int8

echo "----------------------- INT8 (BF16) -----------------------"
sudo bash ../drop_cache.sh
python3 baseline3_safetensors.py --coding_type encode --model_name "${BASE_MODEL_NAME}" --dtype bf16 --int8
sudo bash ../drop_cache.sh
python3 baseline3_safetensors.py --coding_type decode --model_name "${BASE_MODEL_NAME}" --dtype bf16 --int8
