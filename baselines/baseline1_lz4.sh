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

mkdir -p ../build && cd ../build || exit 1
cmake ..
make baseline1_lz4

echo "----------------------- FP16 -----------------------"
sudo bash ../drop_cache.sh
./baseline1_lz4 encode fp16 "${BASE_MODEL_NAME}"
sudo bash ../drop_cache.sh
./baseline1_lz4 decode fp16 "${BASE_MODEL_NAME}"

echo "----------------------- FP16-INT8 -----------------------"
sudo bash ../drop_cache.sh
./baseline1_lz4 encode fp16-int8 "${BASE_MODEL_NAME}"
sudo bash ../drop_cache.sh
./baseline1_lz4 decode fp16-int8 "${BASE_MODEL_NAME}"

echo "----------------------- BF16 -----------------------"
sudo bash ../drop_cache.sh
./baseline1_lz4 encode bf16 "${BASE_MODEL_NAME}"
sudo bash ../drop_cache.sh
./baseline1_lz4 decode bf16 "${BASE_MODEL_NAME}"

echo "----------------------- BF16-INT8 -----------------------"
sudo bash ../drop_cache.sh
./baseline1_lz4 encode bf16-int8 "${BASE_MODEL_NAME}"
sudo bash ../drop_cache.sh
./baseline1_lz4 decode bf16-int8 "${BASE_MODEL_NAME}"