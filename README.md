## QStore: Quantization-Aware Compressed Model Storage

QStore presents a unified, **lossless compressed format** with associated encoding and decoding schemes that can be used to store high-precision and low-precision foundation models together.
It reduces the overall storage footprint by up to **2.2× (45% of the original size)** while enabling up to **1.7×** and **1.8×** faster model saving and loading versus existing approaches

A model pair stored in the QStore format can be losslessly decoded to load the low or high-precision model (or both). Instead of storing low-precision and high-precision models separately, QStore stores the low-precision model and only the residual information needed to reconstruct high-precision models. The size of the residual information is significantly smaller than the original high-precision models, thus, achieving much higher savings in storage cost. The low-precision models can be loaded quickly just like before. The high-precision models can also be reconstructed efficiently in memory by merging low-precision data and the residual with QStore’s lightweight decoding logic. 

### Testing Environment and Setup
* Ubuntu 22.04 LTS
* C++20 (gcc 11.4)
* Python 3.10.12

```
sudo apt update
sudo apt install build-essential
sudo apt install cmake pkg-config zip
sudo apt install python3-pip
```

Setup repository:
```
git clone https://github.com/illinoisdata/qstore
cd qstore
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh
```

To download data:
Create a python venv, then install the following:
```
<Install torch (cpu version)>
pip install transformers
pip install accelerate
```

`download_models.ipynb` shows a basic workflow that can be used to download and save the unquantized and quantized models from huggingface. Our approach will use the saved data as input.

### Run QStore:

Run:

The OS cache can be cleared between runs by using `sudo bash drop_cache.sh` for accurate benchmarking.
Before running, the FiniteStateEntropy library (https://github.com/Cyan4973/FiniteStateEntropy) needs to be cloned and placed in the `baselines/zipnn_c_api/` folder. This will be used to run Huffman (Huff0) compression/decompression.

For convenience, all the commands to run QStore are available in `qstore.sh`:

```
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
```

Statistics and times will be printed out at the end. After decompression, the tensors will be verified to ensure they are exactly the same as the original downloaded data.

Variation of load time with read bandwidth speed can be measured in `read_bandwidth_variation.sh`

To run baselines, please refer to `baselines/README.md`.