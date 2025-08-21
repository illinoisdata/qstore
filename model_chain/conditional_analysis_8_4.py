import os
import sys
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import dahuffman
from collections import Counter
from datasets import load_dataset
import numpy as np
import zipnn
import time
import json
import logging
from pathlib import Path

torch.set_printoptions(precision=20)

# Setup logging
def setup_logging():
    """Setup logging configuration"""
    log_dir = Path("conditional_analysis_8_4_logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"conditional_analysis_8_4_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_dir

# Setup checkpoint management
def load_checkpoint(checkpoint_file):
    """Load existing checkpoint results"""
    if Path(checkpoint_file).exists():
        logging.info(f"Loading checkpoint from {checkpoint_file}")
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    else:
        logging.info("No existing checkpoint found, starting fresh")
        return {}

def save_checkpoint(checkpoint_file, results):
    """Save current results to checkpoint file"""
    with open(checkpoint_file, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Checkpoint saved to {checkpoint_file}")

def save_layer_result(log_dir, layer_name, result):
    """Save individual layer result"""
    layer_file = log_dir / f"layer_{layer_name.replace('.', '_')}.json"
    with open(layer_file, 'w') as f:
        json.dump({
            'layer_name': layer_name,
            'timestamp': datetime.now().isoformat(),
            'result': result
        }, f, indent=2)
    logging.info(f"Layer result saved: {layer_file}")

def get_progress_summary(log_dir):
    """Get a summary of processing progress"""
    checkpoint_file = log_dir / "checkpoint.json"
    if not checkpoint_file.exists():
        return None
        
    with open(checkpoint_file, 'r') as f:
        checkpoint_data = json.load(f)
    
    completed_layers = set(checkpoint_data.get('completed_layers', []))
    total_layers = len(common_layers)
    remaining_layers = [layer for layer in common_layers if layer not in completed_layers]
    
    return {
        'total_layers': total_layers,
        'completed_layers': len(completed_layers),
        'remaining_layers': len(remaining_layers),
        'progress_percentage': (len(completed_layers) / total_layers) * 100,
        'last_updated': checkpoint_data.get('last_updated'),
        'next_layers_to_process': remaining_layers[:5]  # Show next 5 layers
    }

if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU available")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Conditional analysis 8|4 QStore encoding")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Base directory containing quantized models")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model identifier (e.g., mistral-7b-instruct-v0.2)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of worker threads (default: 8)")
    return parser.parse_args()

def load_models(model_dir, model_name):
    """Load 8-bit and 4-bit models."""
    model_4bit_path = os.path.join(model_dir, f"{model_name}-gptq-4bit")
    model_8bit_path = os.path.join(model_dir, f"{model_name}-gptq-8bit")
    
    print(f"Loading models from {model_dir}")
    
    # Load models
    print("Loading 4-bit GPTQ model...")
    model_4bit = AutoModelForCausalLM.from_pretrained(
        model_4bit_path,
        device_map="auto"
    )
    st_4bit = model_4bit.state_dict()
    print(f"4-bit qweight shape: {st_4bit['model.layers.0.self_attn.k_proj.qweight'].shape}")
    print(f"4-bit qzeros shape: {st_4bit['model.layers.0.self_attn.k_proj.qzeros'].shape}")
    print(f"4-bit scales shape: {st_4bit['model.layers.0.self_attn.k_proj.scales'].shape}")
    print(f"4-bit g_idx shape: {st_4bit['model.layers.0.self_attn.k_proj.g_idx'].shape}")

    print("Loading 8-bit GPTQ model...")
    model_8bit = AutoModelForCausalLM.from_pretrained(
        model_8bit_path,
        device_map="auto"
    )
    st_8bit = model_8bit.state_dict()
    print(f"8-bit qweight shape: {st_8bit['model.layers.0.self_attn.k_proj.qweight'].shape}")
    print(f"8-bit qzeros shape: {st_8bit['model.layers.0.self_attn.k_proj.qzeros'].shape}")
    print(f"8-bit scales shape: {st_8bit['model.layers.0.self_attn.k_proj.scales'].shape}")
    print(f"8-bit g_idx shape: {st_8bit['model.layers.0.self_attn.k_proj.g_idx'].shape}")
    
    return st_8bit, st_4bit

# Parse arguments and load models
args = parse_args()
st_8bit, st_4bit = load_models(args.model_dir, args.model_name)

print(st_8bit["model.layers.0.self_attn.k_proj.qweight"].shape)
print(st_8bit["model.layers.0.self_attn.k_proj.qzeros"].shape)
print(st_8bit["model.layers.0.self_attn.k_proj.scales"].shape)
print(st_8bit["model.layers.0.self_attn.k_proj.g_idx"].shape)

def get_common_weight_layers():
    """Get layer names that exist in both 4-bit and 8-bit models"""
    layers_4bit = set()
    layers_8bit = set()
    
    # Extract 4-bit layer names
    for key in st_4bit.keys():
        if key.endswith('.qweight'):
            layers_4bit.add(key.replace('.qweight', ''))
    
    # Extract 8-bit layer names
    for key in st_8bit.keys():
        if key.endswith('.qweight'):
            layers_8bit.add(key.replace('.qweight', ''))
    
    # Find common layers
    common_layers = list(layers_4bit & layers_8bit)
    return sorted(common_layers)

common_layers = get_common_weight_layers()

# Wrapper function for multiprocessing
def process_single_layer(args):
    """
    Wrapper function for processing a single layer in parallel
    Returns (layer_name, result, error) tuple
    """
    layer_name, device_id = args
    
    # Set device for this worker
    if torch.cuda.is_available() and device_id is not None:
        torch.cuda.set_device(device_id % torch.cuda.device_count())
    
    try:
        print(f"Worker processing {layer_name} on device {device_id}")
        start_time = time.time()
        result = vectorized_conditional_huffman_8_4_no_rounding(layer_name)
        end_time = time.time()
        print(f"Worker completed {layer_name} in {end_time - start_time:.2f} seconds")
        
        # Clean up GPU memory after processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return (layer_name, result, None)  # (layer_name, result, error)
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"Worker error processing {layer_name}: {error_msg}")
        return (layer_name, None, error_msg)  # (layer_name, result, error)

# Conditional Huffman encoding for 8-bit weights given 4-bit weights
def vectorized_conditional_huffman_8_4_no_rounding(layer_name):
    """
    Vectorized implementation for conditional encoding: 8-bit weights | 4-bit weights
    Uses exact scale values for precise grouping
    """
    print(f"Vectorized conditional 8|4 analysis (no rounding) for {layer_name}...")
    
    # Get required tensors from 8-bit model (target)
    scales_8bit = st_8bit[f"{layer_name}.scales"]  # Shape: [num_groups, output_dim]
    g_idx_8bit = st_8bit[f"{layer_name}.g_idx"]    # Shape: [input_dim]
    qweight_8bit = st_8bit[f"{layer_name}.qweight"]  # Shape: [packed_input_dim, output_dim]
    
    # Get required tensors from 4-bit model (condition)
    scales_4bit = st_4bit[f"{layer_name}.scales"]  # Shape: [num_groups, output_dim]
    g_idx_4bit = st_4bit[f"{layer_name}.g_idx"]    # Shape: [input_dim]
    qweight_4bit = st_4bit[f"{layer_name}.qweight"]  # Shape: [packed_input_dim, output_dim]
    
    # Verify that both models have the same structure
    assert scales_8bit.shape == scales_4bit.shape, f"Scale shapes don't match: {scales_8bit.shape} vs {scales_4bit.shape}"
    assert g_idx_8bit.shape == g_idx_4bit.shape, f"g_idx shapes don't match: {g_idx_8bit.shape} vs {g_idx_4bit.shape}"
    
    # Use the structure from either model (they should be the same)
    input_dim_size = g_idx_8bit.shape[0]
    output_dim_size = scales_8bit.shape[1]
    device = scales_8bit.device
    
    print(f"  Processing {input_dim_size} x {output_dim_size} weight matrix...")
    print(f"  Groups: {scales_8bit.shape[0]}, Group size: {input_dim_size // scales_8bit.shape[0]}")
    
    # Step 1: Create all coordinate pairs
    input_coords = torch.arange(input_dim_size, device=device)
    output_coords = torch.arange(output_dim_size, device=device)
    input_grid, output_grid = torch.meshgrid(input_coords, output_coords, indexing='ij')
    
    # Flatten for vectorized operations
    input_flat = input_grid.flatten()  # [input_dim * output_dim]
    output_flat = output_grid.flatten()  # [input_dim * output_dim]
    
    # Step 2: Get group IDs for all positions (use 4-bit model for grouping - should be same as 8-bit)
    g_idx_4bit = g_idx_4bit.to(device)
    scales_4bit = scales_4bit.to(device)
    qweight_4bit = qweight_4bit.to(device)
    
    g_idx_8bit = g_idx_8bit.to(device)
    scales_8bit = scales_8bit.to(device)
    qweight_8bit = qweight_8bit.to(device)
    
    group_ids_flat = g_idx_4bit[input_flat]  # [input_dim * output_dim]
    
    # Step 3: Extract scale values from 4-bit model (conditioning variable)
    scale_vals_4bit_flat = scales_4bit[group_ids_flat, output_flat]  # [input_dim * output_dim]
    
    # Step 4: Extract all 8-bit quantized values (target variable)
    # For 8-bit: each weight takes 1 byte, so packing factor is 4 (32-bit integers / 8 bits per weight)
    packed_indices_8bit = input_flat // 4  # 4 weights per 32-bit integer
    bit_positions_8bit = input_flat % 4     # Position within the packed integer
    packed_vals_8bit = qweight_8bit[packed_indices_8bit, output_flat]
    bit_shifts_8bit = bit_positions_8bit * 8     # 8 bits per weight
    quantized_8bit_flat = (packed_vals_8bit >> bit_shifts_8bit) & 0xFF  # Extract 8 bits [input_dim * output_dim]
    
    # Step 5: Extract all 4-bit quantized values (conditioning variable)
    # For 4-bit: each weight takes 4 bits, so packing factor is 8 (32-bit integers / 4 bits per weight)
    packed_indices_4bit = input_flat // 8  # 8 weights per 32-bit integer
    bit_positions_4bit = input_flat % 8     # Position within the packed integer
    packed_vals_4bit = qweight_4bit[packed_indices_4bit, output_flat]
    bit_shifts_4bit = bit_positions_4bit * 4     # 4 bits per weight
    quantized_4bit_flat = (packed_vals_4bit >> bit_shifts_4bit) & 0xF  # Extract 4 bits [input_dim * output_dim]
    
    # Convert to CPU with appropriate data types
    scale_vals_4bit_cpu = scale_vals_4bit_flat.cpu().numpy().astype(np.float32)  # float32 sufficient for exact comparison
    quantized_4bit_cpu = quantized_4bit_flat.cpu().numpy().astype(np.uint8)  # uint8 sufficient for 4-bit values (0-15)
    quantized_8bit_cpu = quantized_8bit_flat.cpu().numpy().astype(np.uint8)  # uint8 perfect for 8-bit values (0-255)
    
    # Step 6: Group by exact scale values first, then by 4-bit quantized values
    print(f"  Grouping by exact scale values from 4-bit model...")
    
    # Use exact scale values without rounding
    unique_scales = np.unique(scale_vals_4bit_cpu)
    
    total_conditional_bits = 0
    total_weights = 0
    
    print(f"  Processing {len(unique_scales)} unique scale values...")
    
    # Process in smaller batches to avoid memory issues
    batch_size = len(unique_scales) // 10 + 1
    
    for i in range(0, len(unique_scales), batch_size):
        batch_scales = unique_scales[i:i+batch_size]
        
        for scale_val in batch_scales:
            # Get all positions with this exact scale value
            scale_mask = scale_vals_4bit_cpu == scale_val
            scale_quantized_4bit = quantized_4bit_cpu[scale_mask]
            scale_quantized_8bit = quantized_8bit_cpu[scale_mask]
            
            # Group by 4-bit quantized values within this scale
            unique_quantized_4bit = np.unique(scale_quantized_4bit)
            
            for q_val_4bit in unique_quantized_4bit:
                # Get all 8-bit weights for this (scale_value, 4bit_quantized_value) combination
                q_mask = scale_quantized_4bit == q_val_4bit
                subgroup_8bit_weights = scale_quantized_8bit[q_mask]
                
                if len(subgroup_8bit_weights) > 1:
                    try:
                        # Apply Huffman coding to the 8-bit quantized values
                        huffman_codec = dahuffman.HuffmanCodec.from_data(subgroup_8bit_weights)
                        compressed = huffman_codec.encode(subgroup_8bit_weights)
                        subgroup_bits = len(compressed) * 8
                    except Exception as e:
                        # Fallback to no compression if Huffman fails
                        print(f"    Huffman failed for scale {scale_val:.12f}, q_val_4bit {q_val_4bit}: {e}")
                        subgroup_bits = len(subgroup_8bit_weights) * 8  # 8 bits per value
                else:
                    # Single value, no compression needed
                    subgroup_bits = 8  # 8 bits for one value
                
                total_conditional_bits += subgroup_bits
                total_weights += len(subgroup_8bit_weights)
    
    print(f"  Completed! Total weights: {total_weights}, Total bits: {total_conditional_bits}")
    
    # Calculate metrics
    original_bits = total_weights * 8  # Original 8-bit weights
    avg_bits = total_conditional_bits / total_weights if total_weights > 0 else 0
    compression_ratio = original_bits / total_conditional_bits if total_conditional_bits > 0 else 0
    
    return {
        'original_size': original_bits,
        'compressed_size': total_conditional_bits,
        'avg_bits_per_value': avg_bits,
        'compression_ratio': compression_ratio,
        'num_values': total_weights,
        'num_scale_groups': len(unique_scales)
    }

def main():
    """Main function with parallelized processing and checkpointing"""
    # Setup logging
    log_dir = setup_logging()
    checkpoint_file = log_dir / "checkpoint.json"
    
    logging.info("=== Starting Conditional Analysis 8|4 with Parallelization ===")
    logging.info(f"Total layers to process: {len(common_layers)}")
    logging.info(f"Checkpoint file: {checkpoint_file}")
    logging.info(f"Using {8} worker threads")
    
    # Check for existing progress
    progress = get_progress_summary(log_dir)
    if progress:
        logging.info(f"Resuming from checkpoint:")
        logging.info(f"  Progress: {progress['completed_layers']}/{progress['total_layers']} "
                    f"({progress['progress_percentage']:.1f}%)")
        logging.info(f"  Last updated: {progress['last_updated']}")
        if progress['next_layers_to_process']:
            logging.info(f"  Next layers: {', '.join(progress['next_layers_to_process'])}")
    
    # Load existing checkpoint
    checkpoint_data = load_checkpoint(checkpoint_file)
    completed_layers = set(checkpoint_data.get('completed_layers', []))
    
    # Filter out already completed layers
    remaining_layers = [layer for layer in common_layers if layer not in completed_layers]
    logging.info(f"Remaining layers to process: {len(remaining_layers)}")
    
    if completed_layers:
        logging.info(f"Skipping {len(completed_layers)} already completed layers")
    
    # Initialize totals from checkpoint
    total_orig_size = checkpoint_data.get('total_orig_size', 0)
    total_compressed_size = checkpoint_data.get('total_compressed_size', 0)
    total_num_values = checkpoint_data.get('total_num_values', 0)
    layer_results = checkpoint_data.get('layer_results', {})
    
    if remaining_layers:
        logging.info("Starting parallel processing with 8 workers...")
        
        # Prepare arguments for parallel processing
        # Distribute layers across available GPUs if any
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        args_list = []
        
        for i, layer_name in enumerate(remaining_layers):
            device_id = i % num_gpus if num_gpus > 0 else None
            args_list.append((layer_name, device_id))
        
        start_time = time.time()
        
        # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid model reloading
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit all tasks
            future_to_layer = {executor.submit(process_single_layer, args): args[0] 
                             for args in args_list}
            
            # Process completed tasks
            completed_count = 0
            total_to_process = len(remaining_layers)
            
            for future in future_to_layer:
                layer_name = future_to_layer[future]
                try:
                    layer_name_result, result, error = future.result()
                    completed_count += 1
                    
                    if error is None and result is not None:
                        # Success
                        logging.info(f"Completed {layer_name} ({completed_count}/{total_to_process})")
                        logging.info(f"  Average bits per value: {result['avg_bits_per_value']:.4f}")
                        logging.info(f"  Compression ratio: {result['compression_ratio']:.2f}x")
                        logging.info(f"  Number of scale groups: {result['num_scale_groups']:,}")
                        
                        # Update totals
                        total_orig_size += result['original_size']
                        total_compressed_size += result['compressed_size']
                        total_num_values += result['num_values']
                        
                        # Store layer result
                        layer_results[layer_name] = result
                        completed_layers.add(layer_name)
                        
                        # Save individual layer result
                        save_layer_result(log_dir, layer_name, result)
                        
                        # Update checkpoint
                        checkpoint_data.update({
                            'completed_layers': list(completed_layers),
                            'total_orig_size': total_orig_size,
                            'total_compressed_size': total_compressed_size,
                            'total_num_values': total_num_values,
                            'layer_results': layer_results,
                            'last_updated': datetime.now().isoformat()
                        })
                        save_checkpoint(checkpoint_file, checkpoint_data)
                        
                        # Log progress every 10 layers
                        if completed_count % 10 == 0:
                            current_progress = (len(completed_layers) / len(common_layers)) * 100
                            logging.info(f"Overall progress: {current_progress:.1f}% "
                                       f"({len(completed_layers)}/{len(common_layers)} layers)")
                        
                    else:
                        # Error occurred
                        logging.error(f"Error processing {layer_name} ({completed_count}/{total_to_process}): {error}")
                        
                except Exception as e:
                    logging.error(f"Exception processing {layer_name}: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
        
        total_time = time.time() - start_time
        logging.info(f"Parallel processing completed in {total_time:.2f} seconds")
    
    # Final results
    logging.info("=== Final Results ===")
    logging.info(f"Total original size: {total_orig_size}")
    logging.info(f"Total compressed size: {total_compressed_size}")
    if total_num_values > 0:
        avg_bits_per_value = total_compressed_size / total_num_values
        logging.info(f"Total avg bits per value: {avg_bits_per_value:.4f}")
    if total_compressed_size > 0:
        compression_ratio = total_orig_size / total_compressed_size
        logging.info(f"Total compression ratio: {compression_ratio:.2f}x")
    
    # Save final summary
    summary = {
        'total_layers_processed': len(completed_layers),
        'total_orig_size': total_orig_size,
        'total_compressed_size': total_compressed_size,
        'total_num_values': total_num_values,
        'avg_bits_per_value': total_compressed_size / total_num_values if total_num_values > 0 else 0,
        'compression_ratio': total_orig_size / total_compressed_size if total_compressed_size > 0 else 0,
        'completed_layers': list(completed_layers),
        'layer_results': layer_results,
        'completion_time': datetime.now().isoformat(),
        'analysis_type': '8bit_given_4bit'
    }
    
    summary_file = log_dir / "final_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Final summary saved to {summary_file}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        import traceback
        logging.error(traceback.format_exc())
