#!/bin/bash
mkdir -p build && cd build || exit 1
cmake ..
make baseline2_zstd

echo "----------------------- FP16 -----------------------"
sudo bash ../drop_cache.sh
./baseline2_zstd encode fp16
sudo bash ../drop_cache.sh
./baseline2_zstd decode fp16

echo "----------------------- FP16-INT8 -----------------------"
sudo bash ../drop_cache.sh
./baseline2_zstd encode fp16-int8
sudo bash ../drop_cache.sh
./baseline2_zstd decode fp16-int8

echo "----------------------- BF16 -----------------------"
sudo bash ../drop_cache.sh
./baseline2_zstd encode bf16
sudo bash ../drop_cache.sh
./baseline2_zstd decode bf16

echo "----------------------- BF16-INT8 -----------------------"
sudo bash ../drop_cache.sh
./baseline2_zstd encode bf16-int8
sudo bash ../drop_cache.sh
./baseline2_zstd decode bf16-int8