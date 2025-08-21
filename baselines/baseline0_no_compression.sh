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
make baseline0_no_compression

echo "-----------FP16 encode-----------"
sudo bash ../drop_cache.sh
./baseline0_no_compression encode fp16 "${BASE_MODEL_NAME}"

echo "-----------BF16 encode-----------"
sudo bash ../drop_cache.sh
./baseline0_no_compression encode bf16 "${BASE_MODEL_NAME}"

echo "-----------FP16-INT8 encode-----------"
sudo bash ../drop_cache.sh
./baseline0_no_compression encode fp16-int8 "${BASE_MODEL_NAME}"

echo "-----------BF16-INT8 encode-----------"
sudo bash ../drop_cache.sh
./baseline0_no_compression encode bf16-int8 "${BASE_MODEL_NAME}"

echo "-----------FP16 decode-----------"
sudo bash ../drop_cache.sh
./baseline0_no_compression decode fp16 "${BASE_MODEL_NAME}"

echo "-----------BF16 decode-----------"
sudo bash ../drop_cache.sh
./baseline0_no_compression decode bf16 "${BASE_MODEL_NAME}"

echo "-----------FP16-INT8 decode-----------"
sudo bash ../drop_cache.sh
./baseline0_no_compression decode fp16-int8 "${BASE_MODEL_NAME}"

echo "-----------BF16-INT8 decode-----------"
sudo bash ../drop_cache.sh
./baseline0_no_compression decode bf16-int8 "${BASE_MODEL_NAME}"
cd ..
