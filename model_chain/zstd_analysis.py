import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
from concurrent.futures import ThreadPoolExecutor
import dahuffman
from collections import Counter
from datasets import load_dataset
import numpy as np
import zstandard as zstd
import time
import json
import os

torch.set_printoptions(precision=20)

if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU available")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ZSTD compression analysis for all model quantizations")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Base directory containing quantized models")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model identifier (e.g., mistral-7b-instruct-v0.2)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of worker threads (default: 8)")
    return parser.parse_args()

def load_models(model_dir, model_name):
    """Load 4-bit, 8-bit, and 16-bit models."""
    model_4bit_path = os.path.join(model_dir, f"{model_name}-gptq-4bit")
    model_8bit_path = os.path.join(model_dir, f"{model_name}-gptq-8bit")
    model_16bit_path = os.path.join(model_dir, f"{model_name}-fp16")
    
    print(f"Loading models from {model_dir}")
    
    model_4bit = AutoModelForCausalLM.from_pretrained(
        model_4bit_path,
        device_map="auto"
    )
    st_4bit = model_4bit.state_dict()
    print(f"4-bit qweight shape: {st_4bit['model.layers.0.self_attn.k_proj.qweight'].shape}")
    print(f"4-bit qzeros shape: {st_4bit['model.layers.0.self_attn.k_proj.qzeros'].shape}")
    print(f"4-bit scales shape: {st_4bit['model.layers.0.self_attn.k_proj.scales'].shape}")
    print(f"4-bit g_idx shape: {st_4bit['model.layers.0.self_attn.k_proj.g_idx'].shape}")

    model_8bit = AutoModelForCausalLM.from_pretrained(
        model_8bit_path,
        device_map="auto"
    )
    st_8bit = model_8bit.state_dict()
    print(f"8-bit qweight shape: {st_8bit['model.layers.0.self_attn.k_proj.qweight'].shape}")
    print(f"8-bit qzeros shape: {st_8bit['model.layers.0.self_attn.k_proj.qzeros'].shape}")
    print(f"8-bit scales shape: {st_8bit['model.layers.0.self_attn.k_proj.scales'].shape}")
    print(f"8-bit g_idx shape: {st_8bit['model.layers.0.self_attn.k_proj.g_idx'].shape}")

    # Load 16-bit model
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_16bit_path,
        device_map="auto",
    )
    st_16bit = model_fp16.state_dict()
    print(f"16-bit weight shape: {st_16bit['model.layers.0.self_attn.k_proj.weight'].shape}")
    
    return st_4bit, st_8bit, st_16bit

# Parse arguments and load models
args = parse_args()
st_4bit, st_8bit, st_16bit = load_models(args.model_dir, args.model_name)

# Get all weight layers that exist in all three models
def get_common_weight_layers():
    """Get layer names that exist in all three models"""
    layers_16bit = set()
    layers_4bit = set()
    layers_8bit = set()
    
    # Extract 16-bit layer names
    for key in st_16bit.keys():
        if key.endswith('.weight') and 'embed' not in key and 'norm' not in key:
            layers_16bit.add(key.replace('.weight', ''))
    
    # Extract 4-bit layer names
    for key in st_4bit.keys():
        if key.endswith('.qweight'):
            layers_4bit.add(key.replace('.qweight', ''))
    
    # Extract 8-bit layer names
    for key in st_8bit.keys():
        if key.endswith('.qweight'):
            layers_8bit.add(key.replace('.qweight', ''))
    
    # Find common layers
    common_layers = list(layers_16bit & layers_4bit & layers_8bit)
    return sorted(common_layers)

common_layers = get_common_weight_layers()
print(f"Found {len(common_layers)} common weight layers")
print("First 10 layers:")
for i, layer in enumerate(common_layers[:10]):
    print(f"  {i+1}. {layer}")
if len(common_layers) > 10:
    print(f"  ... and {len(common_layers) - 10} more")

def process_layer_zstd_16bit(layer_name):
    """Process a single layer for ZipNN 16-bit compression"""
    if f"{layer_name}.weight" in st_16bit:
        compressor = zstd.ZstdCompressor()
        # Convert to float16 to ensure 16-bit representation
        weight_tensor = st_16bit[f"{layer_name}.weight"].cpu()
        weight_numpy = weight_tensor.numpy().astype(np.float16)
        
        # Convert to bytes explicitly
        array_bytes = weight_numpy.tobytes()
        compressed_data = compressor.compress(array_bytes)
        
        original_bits = weight_numpy.size * 16
        compressed_bits = len(compressed_data) * 8
        avg_bits = compressed_bits / weight_numpy.size
        
        # Correctly convert bits to Gigabytes
        bits_to_gb = 8 * 1024 * 1024 * 1024
        
        return {
            'original_size': original_bits / bits_to_gb,
            'compressed_size': compressed_bits / bits_to_gb,
            'avg_bits_per_value': avg_bits,
            'num_values': weight_numpy.size
        }
    return None

def extract_all_8bit_weights_gpu(qweight_8bit):
    """Ultra-fast GPU-based extraction of all 8-bit weight values"""
    # Keep everything on GPU
    qweight_flat = qweight_8bit.flatten().unsqueeze(1)  # Shape: [N, 1]
    
    # Create bit shift tensor on GPU
    bit_shifts = torch.arange(4, device=qweight_8bit.device) * 8  # [0, 8, 16, 24]
    
    # Vectorized extraction using broadcasting
    extracted = (qweight_flat >> bit_shifts) & 0xFF
    
    # Flatten and return as list
    return extracted.flatten().cpu().numpy().astype(np.uint8).tolist()

def process_layer_zstd_8bit(layer_name):
    """Process a single layer for Huffman 8-bit compression"""
    if f"{layer_name}.qweight" in st_8bit:
        print(f"  - Processing {layer_name} - Huffman 8-bit (GPU optimized)...")
        qweight_8bit = st_8bit[f"{layer_name}.qweight"]
        # weights_8bit = extract_all_8bit_weights_gpu(qweight_8bit)
        
        # np_8bit = np.array(weights_8bit, dtype=np.uint8)
        np_8bit = qweight_8bit.cpu().numpy()
        compressor = zstd.ZstdCompressor()
        compressed = compressor.compress(np_8bit.tobytes())
        
        original_bits = np_8bit.size * 8 * 4
        compressed_bits = len(compressed) * 8
        avg_bits = compressed_bits / np_8bit.size
        
        return {
            'original_size': original_bits / 1024/1024/1024/8,
            'compressed_size': compressed_bits/1024/1024/1024/8,
            'avg_bits_per_value': avg_bits,
            'num_values': len(np_8bit)
        }
    return None

def extract_all_4bit_weights_gpu(qweight):
    """Ultra-fast GPU-based extraction of all 4-bit weight values"""
    # Keep everything on GPU, use PyTorch operations
    qweight_flat = qweight.flatten().unsqueeze(1)  # Shape: [N, 1]
    
    # Create bit shift tensor on GPU
    bit_shifts = torch.arange(8, device=qweight.device) * 4  # [0, 4, 8, 12, 16, 20, 24, 28]
    
    # Vectorized extraction using broadcasting
    # qweight_flat: [N, 1], bit_shifts: [8] -> result: [N, 8]
    extracted = (qweight_flat >> bit_shifts) & 0xF
    
    # Flatten and return as list
    return extracted.flatten().cpu().numpy().astype(np.uint8).tolist()

def process_layer_zstd_4bit(layer_name):
    """Process a single layer for Huffman 4-bit compression"""
    if f"{layer_name}.qweight" in st_4bit:
        print(f"  - Processing {layer_name} - Huffman 4-bit (GPU optimized)...")
        qweight_4bit = st_4bit[f"{layer_name}.qweight"]
        # weights_4bit = extract_all_4bit_weights_gpu(qweight_4bit)
        
        # np_4bit = np.array(weights_4bit, dtype=np.uint8)
        np_4bit = qweight_4bit.cpu().numpy()
        compressor = zstd.ZstdCompressor()
        compressed = compressor.compress(np_4bit.tobytes())
        
        original_bits = np_4bit.size * 4 * 8
        compressed_bits = len(compressed) * 8
        avg_bits = compressed_bits / np_4bit.size
        
        return {
            'original_size': original_bits / 1024/1024/1024/8,
            'compressed_size': compressed_bits/1024/1024/1024/8,
            'avg_bits_per_value': avg_bits,
            'num_values': len(np_4bit)
        }
    return None

def process_layer_all_bits(layer_name):
    """Process a single layer for all bit widths (16-bit, 8-bit, 4-bit)"""
    print(f"  - Processing {layer_name} - All bit widths...")
    
    results = {}
    
    # Process 16-bit
    result_16bit = process_layer_zstd_16bit(layer_name)
    if result_16bit:
        results['16bit'] = result_16bit
    
    # Process 8-bit
    result_8bit = process_layer_zstd_8bit(layer_name)
    if result_8bit:
        results['8bit'] = result_8bit
    
    # Process 4-bit
    result_4bit = process_layer_zstd_4bit(layer_name)
    if result_4bit:
        results['4bit'] = result_4bit
    
    return layer_name, results

# Parallelize over configurable threads
compression_results = {}

print(f"Processing {len(common_layers)} layers using {args.num_workers} threads...")

with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
    # Submit all tasks
    future_to_layer = {executor.submit(process_layer_all_bits, layer_name): layer_name 
                      for layer_name in common_layers}
    
    # Collect results as they complete
    completed = 0
    for future in future_to_layer:
        layer_name, results = future.result()
        if results:  # Only store if we have at least one result
            compression_results[layer_name] = results
        completed += 1
        if completed % 10 == 0:
            print(f"Completed {completed}/{len(common_layers)} layers")

print(f"Finished processing all {len(compression_results)} layers")

# Compute aggregate statistics for all bit widths
stats = {
    '16bit': {'total_orig_size': 0, 'total_compressed_size': 0, 'total_num_values': 0, 'total_avg_bits': 0, 'layer_count': 0},
    '8bit': {'total_orig_size': 0, 'total_compressed_size': 0, 'total_num_values': 0, 'total_avg_bits': 0, 'layer_count': 0},
    '4bit': {'total_orig_size': 0, 'total_compressed_size': 0, 'total_num_values': 0, 'total_avg_bits': 0, 'layer_count': 0}
}

for layer_name, layer_results in compression_results.items():
    for bit_width, result in layer_results.items():
        stats[bit_width]['total_orig_size'] += result['original_size']
        stats[bit_width]['total_compressed_size'] += result['compressed_size']
        stats[bit_width]['total_num_values'] += result['num_values']
        stats[bit_width]['total_avg_bits'] += result['avg_bits_per_value'] * result['num_values']
        stats[bit_width]['layer_count'] += 1

# Print results for each bit width
for bit_width in ['16bit', '8bit', '4bit']:
    if stats[bit_width]['layer_count'] > 0:
        print(f"\n{bit_width} compression results:")
        print(f"  Layers processed: {stats[bit_width]['layer_count']}")
        print(f"  Total original size: {stats[bit_width]['total_orig_size']:.6f} GB")
        print(f"  Total compressed size: {stats[bit_width]['total_compressed_size']:.6f} GB")
        print(f"  Total num values: {stats[bit_width]['total_num_values']}")
        
        avg_bits = stats[bit_width]['total_avg_bits'] / stats[bit_width]['total_num_values']
        
        compression_ratio = 0
        if stats[bit_width]['total_compressed_size'] > 0:
            compression_ratio = stats[bit_width]['total_orig_size'] / stats[bit_width]['total_compressed_size']
        
        print(f"  Average bits per value: {avg_bits:.4f}")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
    else:
        print(f"\n{bit_width}: No layers processed")