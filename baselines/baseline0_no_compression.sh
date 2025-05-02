#!/bin/bash
mkdir -p ../build && cd ../build || exit 1
cmake ..
make baseline0_no_compression

echo "-----------FP16 encode-----------"
sudo bash ../drop_cache.sh
./baseline0_no_compression encode fp16

echo "-----------BF16 encode-----------"
sudo bash ../drop_cache.sh
./baseline0_no_compression encode bf16

echo "-----------FP16-INT8 encode-----------"
sudo bash ../drop_cache.sh
./baseline0_no_compression encode fp16-int8

echo "-----------BF16-INT8 encode-----------"
sudo bash ../drop_cache.sh
./baseline0_no_compression encode bf16-int8

echo "-----------FP16 decode-----------"
sudo bash ../drop_cache.sh
./baseline0_no_compression decode fp16

echo "-----------BF16 decode-----------"
sudo bash ../drop_cache.sh
./baseline0_no_compression decode bf16

echo "-----------FP16-INT8 decode-----------"
sudo bash ../drop_cache.sh
./baseline0_no_compression decode fp16-int8

echo "-----------BF16-INT8 decode-----------"
sudo bash ../drop_cache.sh
./baseline0_no_compression decode bf16-int8
cd ..
