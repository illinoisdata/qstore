#!/bin/bash
mkdir -p build && cd build || exit 1
cmake ..

echo "------------------ Save compressed 16-bit + 8-bit models --------------------"
sudo bash ../drop_cache.sh
make final_encode && ./final_encode

echo "------------------ Load 16-bit model (which will also load 8-bit by default) --------------------"
sudo bash ../drop_cache.sh
make final_decode && ./final_decode

echo "------------------ Load only 8-bit model --------------------"
sudo bash ../drop_cache.sh
make final_decode_quantized && ./final_decode_quantized