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

The OS cache can be cleared between runs by using `sudo bash drop_cache.sh` for accurate benchmarking.
Before running, the FiniteStateEntropy library (https://github.com/Cyan4973/FiniteStateEntropy) needs to be cloned and placed in the `baselines/zipnn_c_api/` folder. This will be used to run Huffman (Huff0) compression/decompression.

For convenience, all the commands to run QStore are available in `qstore.sh`:

```bash
# Usage: ./qstore.sh <model_name>
# Example:
./qstore.sh deepseek-ai/deepseek-coder-33b-instruct
```

**Models used in the paper:**
- `meta-llama/Llama-3.1-8B-Instruct`
- `qwen/qwen2.5-7b-instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `qwen/qwen2.5-vl-32B-instruct`
- `qwen/qwen2-audio-7b-instruct`
- `deepseek-ai/deepseek-coder-33b-instruct`
- `google/gemma-3-27b-it`

The script will:
1. Compile the project 
2. Save compressed 16-bit + 8-bit models
3. Load 16-bit model (which will also load 8-bit by default)
4. Load only 8-bit model

Statistics and times will be printed out at the end. After decompression, the tensors will be verified to ensure they are exactly the same as the original downloaded data.

Variation of load time with read bandwidth speed can be measured in `read_bandwidth_variation.sh`

## Running Baselines

This section contains baseline compression methods for comparison with qstore.

### Prerequisites

1. **Model Data**: Ensure model files are available in `~/benchmark_data/` with the proper directory structure:
   ```
   ~/benchmark_data/
   ├── meta-llama/
   │   ├── Llama-3.1-8B-Instruct-fp16/
   │   ├── Llama-3.1-8B-Instruct-bf16/
   │   ├── Llama-3.1-8B-Instruct-fp16-int8/
   │   └── Llama-3.1-8B-Instruct-bf16-int8/
   └── [other model directories...]
   ```

2. **Compilation**: Build all baselines by running:
   ```bash
   mkdir -p build && cd build
   cmake .. && make
   ```

### Quick Start - Run All Baselines

The easiest way to run comprehensive benchmarks is using the automated script:

```bash
# Run all baselines for a specific model
./baselines/run_all_baselines.sh "meta-llama/Llama-3.1-8B-Instruct"

# Run all baselines for ALL paper models (takes a long time!)
./baselines/run_all_baselines.sh

# Show help and available models
./baselines/run_all_baselines.sh --help
```

The script will automatically:
- Run all 5 baseline methods
- Test all data types (FP16, BF16, FP16-INT8, BF16-INT8)
- Log detailed results with timing information
- Handle errors gracefully with user prompts

### Individual Baseline Scripts

You can also run individual baseline methods:

#### Baseline 0 (No Compression)
Plain saving and loading in C++:
```bash
./baselines/baseline0_no_compression.sh "meta-llama/Llama-3.1-8B-Instruct"
```

#### Baseline 1 (LZ4 Compression)
```bash
./baselines/baseline1_lz4.sh "meta-llama/Llama-3.1-8B-Instruct"
```

#### Baseline 2 (ZSTD Compression)
```bash
./baselines/baseline2_zstd.sh "meta-llama/Llama-3.1-8B-Instruct"
```

#### Baseline 3 (SafeTensors)
**Note**: Before running this baseline, you must first run qstore encoding for the model, which saves the index file containing tensor names to use.

This baseline:
1. Uses the tensor names saved during qstore encoding
2. Creates a new safetensors file with only relevant tensors
3. Saves and loads using safetensors' Python API

```bash
# First run qstore to generate the index file
./qstore.sh "meta-llama/Llama-3.1-8B-Instruct"

# Then run the safetensors baseline
./baselines/baseline3_safetensors.sh "meta-llama/Llama-3.1-8B-Instruct"
```

#### Baseline 4 (ZipNN)
```bash
./baselines/baseline4_zipnn.sh "meta-llama/Llama-3.1-8B-Instruct"
```

### Data Types and Stages

Each baseline supports multiple data types:
- **fp16**: 16-bit floating point
- **bf16**: BFloat16 format
- **fp16-int8**: INT8 quantized from FP16
- **bf16-int8**: INT8 quantized from BF16

Each baseline runs both:
- **Encode**: Compression/saving stage
- **Decode**: Decompression/loading stage

### Output and Results

Results are logged to timestamped files:
- Single model: `baseline_results_[model_name]_[timestamp].log`
- All models: `baseline_results_all_models_[timestamp].log`

The logs contain:
- Compilation output
- Memory usage statistics
- Encoding/decoding times
- File sizes
- Error information (if any)

### Performance Notes

To compute total storage size, saving time, and loading time for model pairs, add the respective sizes and times for 16-bit and 8-bit data components.

---

# Model Chain - Quantization and Storage Analysis

## Overview

This directory contains a **prototype implementation** to quantize and analyze the storage costs of storing a multi-level model chain (16-bit, 8-bit, 4-bit precisions of the same model) using QStore.
We also provide scripts that compute the cost of storing these precisions with other baselines like ZSTD and ZipNN. We use GPT-Q for quantization since it is a popular 2nd order quantization method that is commonly used in research and industry.

**Models tested in the paper:**
- `mistralai/mistral-7b-instruct-v0.2`
- `qwen/qwen2-7b`
- `meta-llama/llama-3.2-3b`

## Prerequisites

Install dependencies:
```bash
pip install torch transformers datasets dahuffman numpy matplotlib zipnn python-zstandard
```


## Scripts Overview

### 1. Model Quantization (`quantize_and_save.py`)

Main script for quantizing models to different bit precisions using GPTQ quantization.

- Quantizes 16-bit models to 4-bit, 8-bit, and saves versions of each
- Uses C4 dataset for calibration
- Saves quantized models to specified directories

```bash
python model_chain/quantize_and_save.py \
  --model_name mistralai/mistral-7b-instruct-v0.2 \
  --cache_dir /path/to/cache \
  --output_dir /path/to/output \
  --hf_token your_token_here 
```

**Output:**
- Creates three model directories with quantized weights
- Prints tensor shapes and quantization statistics
- 4-bit model includes: `qweight`, `qzeros`, `scales`, `g_idx` tensors
- 8-bit model includes: similar quantized tensors
- 16-bit model includes: standard `weight` tensors

### 2. Huffman Compression Analysis

#### 4-bit Huffman Analysis (`huffman_4_bit.py`)

Analyzes compression efficiency of 4-bit quantized weights using Huffman coding.

```bash
python model_chain/huffman_4_bit.py \
  --model_dir /path/to/output \
  --model_name mistral-7b-instruct-v0.2 \
  --num_workers 32
```

#### 8-bit Huffman Analysis (`huffman_8_bit.py`)

Similar analysis for 8-bit quantized weights.

```bash
python model_chain/huffman_8_bit.py \
  --model_dir /path/to/output \
  --model_name mistral-7b-instruct-v0.2 \
  --num_workers 128
```

### 3. Baseline Compression Analysis

#### ZipNN Compression Analysis (`zipnn_analysis.py`)

Analyzes compression efficiency of 16-bit model weights using ZipNN, a specialized neural network compression library.

```bash
python model_chain/zipnn_analysis.py \
  --model_dir /path/to/output \
  --model_name mistral-7b-instruct-v0.2 \
  --num_workers 64
```

#### ZSTD Compression Analysis (`zstd_analysis.py`)

Comprehensive compression analysis using ZSTD (Zstandard) compression for all quantization levels (16-bit, 8-bit, 4-bit).

```bash
python model_chain/zstd_analysis.py \
  --model_dir /path/to/output \
  --model_name mistral-7b-instruct-v0.2 \
  --num_workers 8
```

### 4. QStore Conditional Encoding 

#### Conditional 16 | 8 QStore Encoding (`conditional_analysis_16_8.py`)

Uses QStore's conditional encoding to store the 16-bit model given the 8-bit model 

```bash
python model_chain/conditional_analysis_16_8.py \
  --model_dir /path/to/output \
  --model_name mistral-7b-instruct-v0.2 \
  --num_workers 8
```

#### Conditional 8 | 4 QStore Encoding (`conditional_analysis_8_4.py`)

Store the conditional 8-bit model given the 4-bit model with QStore

```bash
python model_chain/conditional_analysis_8_4.py \
  --model_dir /path/to/output \
  --model_name mistral-7b-instruct-v0.2 \
  --num_workers 8
```

#### Conditional 16 | 4 QStore Encoding (`conditional_analysis_16_4.py`)

This isn't really required since we can represent the model using 16|8 and 8|4 encodings and store the 4-bit model separately. But storing a 16|4 + 4 combination is also possible with QStore and it is also efficient.

```bash
python model_chain/conditional_analysis_16_4.py \
  --model_dir /path/to/output \
  --model_name mistral-7b-instruct-v0.2 \
  --num_workers 8
```

**Conditional Analysis Features:**
- Automated checkpoint saving/loading
- Detailed logging with timestamps
- Progress tracking for large-scale analysis
- Statistical aggregation across model layers
- Memory-optimized processing for large models