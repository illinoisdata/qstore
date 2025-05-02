#!/bin/bash
mkdir -p build && cd build || exit 1
cmake ..
make baseline4_zipnn

echo "--------------------- FP16 encode ---------------------"
sudo bash ../drop_cache.sh
./baseline4_zipnn encode fp16

echo "--------------------- BF16 encode ---------------------"
sudo bash ../drop_cache.sh
./baseline4_zipnn encode bf16

echo "--------------------- FP16 decode ---------------------"
sudo bash ../drop_cache.sh
./baseline4_zipnn decode fp16

echo "--------------------- BF16 decode ---------------------"
sudo bash ../drop_cache.sh
./baseline4_zipnn decode bf16