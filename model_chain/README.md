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
python quantize_and_save.py \
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
python huffman_4_bit.py \
  --model_dir /path/to/output \
  --model_name mistral-7b-instruct-v0.2 \
  --num_workers 32
```

#### 8-bit Huffman Analysis (`huffman_8_bit.py`)

Similar analysis for 8-bit quantized weights.

```bash
python huffman_8_bit.py \
  --model_dir /path/to/output \
  --model_name mistral-7b-instruct-v0.2 \
  --num_workers 128
```

### 3. Baseline Compression Analysis

#### ZipNN Compression Analysis (`zipnn_analysis.py`)

Analyzes compression efficiency of 16-bit model weights using ZipNN, a specialized neural network compression library.

```bash
python zipnn_analysis.py \
  --model_dir /path/to/output \
  --model_name mistral-7b-instruct-v0.2 \
  --num_workers 64
```

#### ZSTD Compression Analysis (`zstd_analysis.py`)

Comprehensive compression analysis using ZSTD (Zstandard) compression for all quantization levels (16-bit, 8-bit, 4-bit).

```bash
python zstd_analysis.py \
  --model_dir /path/to/output \
  --model_name mistral-7b-instruct-v0.2 \
  --num_workers 8
```

### 4. QStore Conditional Encoding 

#### Conditional 16 | 8 QStore Encoding (`conditional_analysis_16_8.py`)

Uses QStore's conditional encoding to store the 16-bit model given the 8-bit model 

```bash
python conditional_analysis_16_8.py \
  --model_dir /path/to/output \
  --model_name mistral-7b-instruct-v0.2 \
  --num_workers 8
```

#### Conditional 8 | 4 QStore Encoding (`conditional_analysis_8_4.py`)

Store the conditional 8-bit model given the 4-bit model with QStore

```bash
python conditional_analysis_8_4.py \
  --model_dir /path/to/output \
  --model_name mistral-7b-instruct-v0.2 \
  --num_workers 8
```

#### Conditional 16 | 4 QStore Encoding (`conditional_analysis_16_4.py`)

This isn't really required since we can represent the model using 16|8 and 8|4 encodings and store the 4-bit model separately. But storing a 16|4 + 4 combination is also possible with QStore and it is also efficient.

```bash
python conditional_analysis_16_4.py \
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