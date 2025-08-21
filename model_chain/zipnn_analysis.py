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
import zipnn
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
    parser = argparse.ArgumentParser(description="ZipNN compression analysis for 16-bit models")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Base directory containing quantized models")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model identifier (e.g., mistral-7b-instruct-v0.2)")
    parser.add_argument("--num_workers", type=int, default=64,
                        help="Number of worker threads (default: 64)")
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

def process_layer_zipnn_16bit(layer_name):
    """Process a single layer for ZipNN 16-bit compression"""
    if f"{layer_name}.weight" in st_16bit:
        print(f"  - Processing {layer_name} - ZipNN 16-bit...")
        compressor = zipnn.ZipNN()
        # Preserve bfloat16 bit structure by viewing as uint16
        weight_tensor = st_16bit[f"{layer_name}.weight"].cpu()
        weight_numpy = weight_tensor.numpy().astype(np.float16)
        compressed_data = compressor.compress(weight_numpy)
        
        original_bits = weight_numpy.size * 16
        compressed_bits = len(compressed_data) * 8
        avg_bits = compressed_bits / weight_numpy.size
        
        return layer_name, {
            'original_size': original_bits / 1024/1024/1024/8,
            'compressed_size': compressed_bits/1024/1024/1024/8,
            'avg_bits_per_value': avg_bits,
            'num_values': weight_numpy.size
        }
    return layer_name, None

# Parallelize over configurable threads
zipnn_16bit_results = {}

print(f"Processing {len(common_layers)} layers using {args.num_workers} threads...")

with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
    # Submit all tasks
    future_to_layer = {executor.submit(process_layer_zipnn_16bit, layer_name): layer_name 
                      for layer_name in common_layers}
    
    # Collect results as they complete
    completed = 0
    for future in future_to_layer:
        layer_name, result = future.result()
        if result is not None:
            zipnn_16bit_results[layer_name] = result
        completed += 1
        if completed % 10 == 0:
            print(f"Completed {completed}/{len(common_layers)} layers")

print(f"Finished processing all {len(zipnn_16bit_results)} layers with valid 16-bit weights")

# Compute aggregate statistics
total_orig_size_zipnn = 0
total_compressed_size_zipnn = 0
total_num_values_zipnn = 0

for layer_name in common_layers:
    total_orig_size_zipnn += zipnn_16bit_results[layer_name]['original_size']
    total_compressed_size_zipnn += zipnn_16bit_results[layer_name]['compressed_size']
    total_num_values_zipnn += zipnn_16bit_results[layer_name]['num_values']

print(f"Total original size: {total_orig_size_zipnn}")
print(f"Total compressed size: {total_compressed_size_zipnn}")
print(f"Total num values: {total_num_values_zipnn}")
print(f"Total avg bits per value: {total_compressed_size_zipnn * 8 * 1024 * 1024 * 1024 / total_num_values_zipnn}")
print(f"Compression ratio: {total_orig_size_zipnn / total_compressed_size_zipnn:.2f}x")