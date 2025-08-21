#!/usr/bin/env python3
"""
Quantization script for GPTQ 4-bit and 8-bit models.
This script loads the original model, quantizes it to 4-bit and 8-bit, and saves the quantized models.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from datasets import load_dataset
import os
import argparse
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_c4_calibration_dataset(num_samples=128):
    """Get C4 calibration dataset for GPTQ quantization."""
    dataset = load_dataset("c4", "en", split="validation", streaming=True)
    calibration_texts = []
    
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        text = sample['text']
        calibration_texts.append(text[:2048])
    
    return calibration_texts

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Quantize models using GPTQ")
    parser.add_argument("--model_name", type=str, required=True,
                        help="HuggingFace model name (e.g., mistralai/mistral-7b-instruct-v0.2)")
    parser.add_argument("--cache_dir", type=str, required=True,
                        help="Directory for model caching")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Base output directory for quantized models")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token (optional, can also use HF_TOKEN env var)")
    parser.add_argument("--num_samples", type=int, default=128,
                        help="Number of calibration samples (default: 128)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup
    model_name = args.model_name
    cache_dir = args.cache_dir
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    
    # Generate model identifier for directories
    model_id = model_name.split("/")[-1] if "/" in model_name else model_name
    
    # Output directories
    output_dir_4bit = os.path.join(args.output_dir, f"{model_id}-gptq-4bit")
    output_dir_8bit = os.path.join(args.output_dir, f"{model_id}-gptq-8bit")
    output_dir_16bit = os.path.join(args.output_dir, f"{model_id}-fp16")
    
    # Create output directories
    os.makedirs(output_dir_4bit, exist_ok=True)
    os.makedirs(output_dir_8bit, exist_ok=True)
    os.makedirs(output_dir_16bit, exist_ok=True)
    
    print("Loading calibration dataset...")
    calibration_dataset = get_c4_calibration_dataset(args.num_samples)
    
    print("Loading tokenizer...")
    tokenizer_kwargs = {"cache_dir": cache_dir}
    if hf_token:
        tokenizer_kwargs["token"] = hf_token
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    
    print("Loading original 16-bit model...")
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
        "cache_dir": cache_dir,
        "revision": "main"
    }
    if hf_token:
        model_kwargs["token"] = hf_token
    
    model_fp16 = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # print("Saving 16-bit model...")
    model_fp16.save_pretrained(output_dir_16bit)
    
    # Print original model info
    st_16bit = model_fp16.state_dict()
    print(st_16bit.keys())
    print(f"Original model k_proj weight shape: {st_16bit['model.layers.0.self_attn.k_proj.weight'].shape}")
    
    # 4-bit quantization
    print("\n" + "="*50)
    print("Starting 4-bit quantization...")
    print("="*50)
    
    gptq_config_4bit = GPTQConfig(
        bits=4,
        dataset=calibration_dataset,
        tokenizer=tokenizer,
        group_size=128
    )
    
    model_4bit_kwargs = {
        "quantization_config": gptq_config_4bit,
        "device_map": "auto",
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
        "cache_dir": cache_dir,
        "revision": "main"
    }
    if hf_token:
        model_4bit_kwargs["token"] = hf_token
    
    model_4bit = AutoModelForCausalLM.from_pretrained(model_name, **model_4bit_kwargs)
    
    # Print 4-bit model info
    st_4bit = model_4bit.state_dict()
    print(f"4-bit qweight shape: {st_4bit['model.layers.0.self_attn.k_proj.qweight'].shape}")
    print(f"4-bit qzeros shape: {st_4bit['model.layers.0.self_attn.k_proj.qzeros'].shape}")
    print(f"4-bit scales shape: {st_4bit['model.layers.0.self_attn.k_proj.scales'].shape}")
    print(f"4-bit g_idx shape: {st_4bit['model.layers.0.self_attn.k_proj.g_idx'].shape}")
    
    # print("Saving 4-bit model...")
    model_4bit.save_pretrained(output_dir_4bit)
    
    # 8-bit quantization
    print("\n" + "="*50)
    print("Starting 8-bit quantization...")
    print("="*50)
    
    gptq_config_8bit = GPTQConfig(
        bits=8,
        dataset=calibration_dataset,
        tokenizer=tokenizer,
        group_size=128
    )
    
    model_8bit_kwargs = {
        "quantization_config": gptq_config_8bit,
        "device_map": "auto",
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
        "cache_dir": cache_dir,
        "revision": "main"
    }
    if hf_token:
        model_8bit_kwargs["token"] = hf_token
    
    model_8bit = AutoModelForCausalLM.from_pretrained(model_name, **model_8bit_kwargs)
    
    # Print 8-bit model info
    st_8bit = model_8bit.state_dict()
    print(f"8-bit qweight shape: {st_8bit['model.layers.0.self_attn.k_proj.qweight'].shape}")
    print(f"8-bit qzeros shape: {st_8bit['model.layers.0.self_attn.k_proj.qzeros'].shape}")
    print(f"8-bit scales shape: {st_8bit['model.layers.0.self_attn.k_proj.scales'].shape}")
    print(f"8-bit g_idx shape: {st_8bit['model.layers.0.self_attn.k_proj.g_idx'].shape}")
    
    print("Saving 8-bit model...")
    model_8bit.save_pretrained(output_dir_8bit)
    
    print("\n" + "="*50)
    print("Quantization complete!")
    print("="*50)
    print(f"16-bit model saved to: {output_dir_16bit}")
    print(f"4-bit model saved to: {output_dir_4bit}")
    print(f"8-bit model saved to: {output_dir_8bit}")

if __name__ == "__main__":
    main()
