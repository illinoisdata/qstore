#!/bin/bash

# Check if model name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_name>"
    echo "Example: $0 deepseek-ai/deepseek-coder-33b-instruct"
    exit 1
fi

# Models used in the paper:
# meta-llama/Llama-3.1-8B-Instruct
# qwen/qwen2.5-7b-instruct
# mistralai/Mistral-7B-Instruct-v0.3
# qwen/qwen2.5-vl-32B-instruct
# qwen/qwen2-audio-7b-instruct
# deepseek-ai/deepseek-coder-33b-instruct
# google/gemma-3-27b-it

MODEL_NAME="$1"
echo "Using model: $MODEL_NAME"

mkdir -p build && cd build || exit 1
cmake ..

echo "------------------ Save compressed 16-bit + 8-bit models --------------------"
sudo bash ../drop_cache.sh
make final_encode && ./final_encode "$MODEL_NAME"

echo "------------------ Load 16-bit model (which will also load 8-bit by default) --------------------"
sudo bash ../drop_cache.sh
make final_decode && ./final_decode "$MODEL_NAME"

echo "------------------ Load only 8-bit model --------------------"
sudo bash ../drop_cache.sh
make final_decode_quantized && ./final_decode_quantized "$MODEL_NAME"