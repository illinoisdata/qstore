## Running Baselines

This directory contains baseline compression methods for comparison with qstore.

### Prerequisites

1. **Model Data**: Ensure model files are available in `/home/raunaks/benchmark_data/` with the proper directory structure:
   ```
   /home/raunaks/benchmark_data/
   ├── meta-llama/
   │   ├── Llama-3.1-8B-Instruct-fp16/
   │   ├── Llama-3.1-8B-Instruct-bf16/
   │   ├── Llama-3.1-8B-Instruct-fp16-int8/
   │   └── Llama-3.1-8B-Instruct-bf16-int8/
   └── [other model directories...]
   ```

2. **Compilation**: Build all baselines by running:
   ```bash
   cd /home/raunaks/qstore
   mkdir -p build && cd build
   cmake .. && make
   ```

### Quick Start - Run All Baselines

The easiest way to run comprehensive benchmarks is using the automated script:

```bash
# Run all baselines for a specific model
./run_all_baselines.sh "meta-llama/Llama-3.1-8B-Instruct"

# Run all baselines for ALL paper models (takes a long time!)
./run_all_baselines.sh

# Show help and available models
./run_all_baselines.sh --help
```

**Models used in the paper:**
- `meta-llama/Llama-3.1-8B-Instruct`
- `qwen/qwen2.5-7b-instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `qwen/qwen2.5-vl-32B-instruct`
- `qwen/qwen2-audio-7b-instruct`
- `deepseek-ai/deepseek-coder-33b-instruct`
- `google/gemma-3-27b-it`

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
./baseline0_no_compression.sh "meta-llama/Llama-3.1-8B-Instruct"
```

#### Baseline 1 (LZ4 Compression)
```bash
./baseline1_lz4.sh "meta-llama/Llama-3.1-8B-Instruct"
```

#### Baseline 2 (ZSTD Compression)
```bash
./baseline2_zstd.sh "meta-llama/Llama-3.1-8B-Instruct"
```

#### Baseline 3 (SafeTensors)
**Note**: Before running this baseline, you must first run `./final_encode` for qstore, which saves the index file containing tensor names to use.

This baseline:
1. Uses the tensor names saved during qstore encoding
2. Creates a new safetensors file with only relevant tensors
3. Saves and loads using safetensors' Python API

```bash
./baseline3_safetensors.sh "meta-llama/Llama-3.1-8B-Instruct"
```

#### Baseline 4 (ZipNN)
```bash
./baseline4_zipnn.sh "meta-llama/Llama-3.1-8B-Instruct"
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