import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch
from safetensors.torch import load_file, save_file

import gc

def weight_dequant_cpu(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    assert x.dim() == 2 and s.dim() == 2, "Expect 2D tensors for x and s"
    M, N = x.shape
    n_m = (M + block_size - 1) // block_size
    n_n = (N + block_size - 1) // block_size

    y = torch.empty((M, N), dtype=torch.bfloat16, device="cpu")
    for bm in range(n_m):
        m0 = bm * block_size
        m1 = min(m0 + block_size, M)
        for bn in range(n_n):
            n0 = bn * block_size
            n1 = min(n0 + block_size, N)
            scale = s[bm, bn].item()
            sub = x[m0:m1, n0:n1].to(torch.float32) * scale
            y[m0:m1, n0:n1] = sub.to(torch.bfloat16)
    return y

def main(fp8_path, bf16_path):
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(bf16_path, exist_ok=True)
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]
    
    loaded_files = {}
    fp8_weight_names = []

    def get_tensor(tensor_name):
        file_name = weight_map[tensor_name]
        if file_name not in loaded_files:
            file_path = os.path.join(fp8_path, file_name)
            loaded_files[file_name] = load_file(file_path, device="cpu")
        return loaded_files[file_name][tensor_name]

    safetensor_files = list(glob(os.path.join(fp8_path, "*.safetensors")))
    safetensor_files.sort()
    for safetensor_file in tqdm(safetensor_files, desc="weight file convert"):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cpu")
        loaded_files[file_name] = current_state_dict
        
        new_state_dict = {}
        for weight_name, weight in current_state_dict.items():
            if weight_name.endswith("_scale_inv"):
                continue
            elif weight.element_size() == 1:
                scale_inv_name = f"{weight_name}_scale_inv"
                try:
                    scale_inv = get_tensor(scale_inv_name)
                    fp8_weight_names.append(weight_name)
                    new_state_dict[weight_name] = weight_dequant_cpu(weight, scale_inv)
                except KeyError:
                    print(f"Warning: {weight_name}loss scale factor")
                    new_state_dict[weight_name] = weight
            else:
                new_state_dict[weight_name] = weight
                
        new_safetensor_file = os.path.join(bf16_path, file_name)
        save_file(new_state_dict, new_safetensor_file)
        
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    new_model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
    for weight_name in fp8_weight_names:
        scale_inv_name = f"{weight_name}_scale_inv"
        if scale_inv_name in weight_map:
            weight_map.pop(scale_inv_name)
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)
    print(f"Finish, Result in: {bf16_path}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-fp8-hf-path", type=str, required=True, help="Kimi-K2 FP8 model")
    parser.add_argument("--output-bf16-hf-path", type=str, required=True, help="BF16 model (After convert)")
    args = parser.parse_args()
    main(args.input_fp8_hf_path, args.output_bf16_hf_path)