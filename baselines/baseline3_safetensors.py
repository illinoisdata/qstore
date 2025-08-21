import torch
from transformers import AutoConfig, PretrainedConfig, AutoModelForCausalLM
from safetensors import safe_open
from safetensors.torch import save_file, load_file
import os
from collections import defaultdict
import json
import time
from argparse import ArgumentParser
from safetensors.torch import load

def load_huggingface_model(model_path, tensor_names=None):
    device="cpu"
    
    if tensor_names is None:
        # no tensor names provided, load all tensors
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device
        )
        state_dict = model.state_dict()
        for _, param in state_dict.items():
            _ = param.data_ptr()
            
        return state_dict
    
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    tensor_file_map = defaultdict(list)
    
    with open(index_path, "r") as f:
        index_data = json.load(f)
    
    weight_map = index_data.get("weight_map", {})

    for tensor_name in tensor_names:
        file_name = weight_map[tensor_name]
        tensor_file_map[file_name].append(tensor_name)
    
    state_dict = defaultdict()
    for file_name, tensors_in_file in tensor_file_map.items():
        file_path = os.path.join(model_path, file_name)
        with safe_open(file_path, framework="pt", device=device) as f:
            for key in tensors_in_file:
                state_dict[key] = f.get_tensor(key)    

    return state_dict

def determine_dtype(model_name):
    """Determine the appropriate dtype based on model name."""
    # Models that use bf16
    bf16_models = [
        "qwen/qwen2-audio-7b-instruct",
        "deepseek-ai/deepseek-coder-33b-instruct", 
        "google/gemma-3-27b-it"
    ]
    
    # Extract base model name (remove dtype suffixes if present)
    base_name = model_name.split('-fp16')[0].split('-bf16')[0].split('-int8')[0]
    
    return "bf16" if base_name in bf16_models else "fp16"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--coding_type",
        type=str,
        help="encode or decode",
        required=True
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="base model name (e.g., meta-llama/Llama-3.1-8B-Instruct)",
        required=True
    )
    parser.add_argument(
        "--dtype",
        type=str,
        help="data type: fp16 or bf16 (auto-detected if not specified)",
        default=None
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="use int8 quantized model"
    )
    args = parser.parse_args()

    torch.set_num_threads(48)
    
    # Simple dtype handling
    if args.dtype is None:
        args.dtype = determine_dtype(args.model_name)
    
    # Model directory name includes dtype and optionally int8
    if args.int8:
        model_dir_name = f"{args.model_name}-{args.dtype}-int8"
    else:
        model_dir_name = f"{args.model_name}-{args.dtype}"
    base_model_name = args.model_name
    
    # Expand home directory and construct paths
    base_data_dir = os.path.expanduser("~/benchmark_data/")
    model_path = os.path.join(base_data_dir, model_dir_name)
    
    print(f"Base model name: {base_model_name}")
    print(f"Model directory: {model_dir_name}")
    print(f"Data type: {args.dtype}")
    print(f"Model path: {model_path}")

    if args.coding_type == "encode":
        # Automatically determine tensor_names_file path based on base model and dtype
        mixedprec_dir_name = f"{base_model_name}-mixedprec-{args.dtype}-int8"
        tensor_names_file = os.path.join(base_data_dir, mixedprec_dir_name, "tensor_index.tsv")
        print(f"Tensor names file: {tensor_names_file}")

        desired_keys = []
        with open(tensor_names_file, "r") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                desired_keys.append(line.split("\t")[0])

        print(len(desired_keys))

        safetensors_path = os.path.join(model_path, "baseline3.safetensors")
        print(safetensors_path)
        if os.path.exists(safetensors_path):
            os.remove(safetensors_path)
            print(f"Removed existing file: {safetensors_path}")

        # load the relevant weights
        state_dict = load_huggingface_model(model_path, desired_keys)

        start = time.perf_counter()
        save_file(state_dict, safetensors_path)
        try:
            fd = os.open(safetensors_path, os.O_RDONLY)
            os.fsync(fd)
        except:
            print("Error during fsync")
        finally:
            os.close(fd)
        end = time.perf_counter()
        elapsed = end - start
        print(f"Time taken for encoding: {elapsed:.4f} seconds")

    elif args.coding_type == "decode":
        start = time.perf_counter()
        safetensors_path = os.path.join(model_path, "baseline3.safetensors")
        with open(safetensors_path, "rb") as f:
            all_bytes = f.read()
        state_dict = load(all_bytes)
        end = time.perf_counter()
        elapsed2 = end - start
        print(f"Time taken for decoding: {elapsed2:.4f} seconds")
