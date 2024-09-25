import torch
from safetensors.torch import load_file, save_file

def safetensor_to_ckpt(safetensor_path, ckpt_path):
    # Load the safetensor file
    safetensor_dict = load_file(safetensor_path)
    
    # Save the dictionary as a PyTorch checkpoint
    torch.save(safetensor_dict, ckpt_path)
    print(f"Successfully converted {safetensor_path} to {ckpt_path}")

safetensor_path = ''
ckpt_path = ''
safetensor_to_ckpt(safetensor_path, ckpt_path)
