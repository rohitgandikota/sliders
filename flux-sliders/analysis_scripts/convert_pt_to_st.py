import torch
import safetensors.torch
from pathlib import Path
import numpy as np

def analyze_state_dict(state_dict, file_name):
    print(f"\n=== Analysis for {file_name} ===")
    
    # Get all layer names
    print("\n1. Layer Structure:")
    print("-------------------")
    cnt = 0
    for key in state_dict.keys():
        tensor = state_dict[key]
        print(key,tensor.shape, tensor.dtype)
        cnt += 1
    
    print(f"\nTotal layers: {cnt}")
    # Analyze LoRA layers specifically
    print("\n2. LoRA Weights Analysis:")
    print("------------------------")
    lora_up_layers = {k: v for k, v in state_dict.items() if 'lora_up' in k}
    lora_down_layers = {k: v for k, v in state_dict.items() if 'lora_down' in k}
    
    print(f"\nFound {len(lora_up_layers)} LoRA up layers and {len(lora_down_layers)} LoRA down layers")
    
    # Analyze each LoRA pair
    for up_key in lora_up_layers:
        base_key = up_key.replace('lora_up', '')
        down_key = up_key.replace('lora_up', 'lora_down')
        
        if down_key in lora_down_layers:
            up_weight = state_dict[up_key]
            down_weight = state_dict[down_key]
            
            print(f"\nLoRA Pair Analysis for {base_key}")
            print("-" * 40)
            
            # Compute effective weight (without scaling)
            effective_weight = (up_weight @ down_weight)
            
            stats = {
                'up': {
                    'min': up_weight.min().item(),
                    'max': up_weight.max().item(),
                    'mean': up_weight.mean().item(),
                    'std': up_weight.std().item()
                },
                'down': {
                    'min': down_weight.min().item(),
                    'max': down_weight.max().item(),
                    'mean': down_weight.mean().item(),
                    'std': down_weight.std().item()
                },
                'effective': {
                    'min': effective_weight.min().item(),
                    'max': effective_weight.max().item(),
                    'mean': effective_weight.mean().item(),
                    'std': effective_weight.std().item()
                }
            }
            
            print(f"Up weights ({up_key}):")
            print(f"  Shape: {up_weight.shape}")
            print(f"  Range: [{stats['up']['min']:.6f}, {stats['up']['max']:.6f}]")
            print(f"  Mean: {stats['up']['mean']:.6f}")
            print(f"  Std: {stats['up']['std']:.6f}")
            
            print(f"\nDown weights ({down_key}):")
            print(f"  Shape: {down_weight.shape}")
            print(f"  Range: [{stats['down']['min']:.6f}, {stats['down']['max']:.6f}]")
            print(f"  Mean: {stats['down']['mean']:.6f}")
            print(f"  Std: {stats['down']['std']:.6f}")
            
            print(f"\nEffective weights (up @ down):")
            print(f"  Shape: {effective_weight.shape}")
            print(f"  Range: [{stats['effective']['min']:.6f}, {stats['effective']['max']:.6f}]")
            print(f"  Mean: {stats['effective']['mean']:.6f}")
            print(f"  Std: {stats['effective']['std']:.6f}")

def convert_pt_to_safetensors(pt_path, output_path=None, analyze=True):
    # Load the .pt file
    state_dict = torch.load(pt_path)
    
    # Analyze the state dict if requested
    if analyze:
        analyze_state_dict(state_dict, Path(pt_path).name)
    
    # If output path is not specified, use the same name but with .safetensors extension
    if output_path is None:
        output_path = str(Path(pt_path).with_suffix('.safetensors'))
    
    # Save as safetensors
    safetensors.torch.save_file(state_dict, output_path)
    print(f"\nConverted {pt_path} to {output_path}")

def analyze_safetensors_file(safetensors_path):
    # Load the safetensors file
    state_dict = safetensors.torch.load_file(safetensors_path)
    analyze_state_dict(state_dict, Path(safetensors_path).name)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to .pt or .safetensors file or directory")
    parser.add_argument("--output_path", type=str, help="Output path (optional)")
    parser.add_argument("--analyze_only", action="store_true", help="Only analyze without converting")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    
    if input_path.is_file():
        if args.analyze_only:
            if input_path.suffix == '.pt':
                state_dict = torch.load(str(input_path))
            elif input_path.suffix == '.safetensors':
                state_dict = safetensors.torch.load_file(str(input_path))
            else:
                raise ValueError("Input file must be either .pt or .safetensors")
            analyze_state_dict(state_dict, input_path.name)
        else:
            if input_path.suffix == '.pt':
                convert_pt_to_safetensors(str(input_path), args.output_path)
            else:
                print("Input file is already in safetensors format")
    elif input_path.is_dir():
        for file in input_path.glob("*.{pt,safetensors}"):
            if args.analyze_only:
                if file.suffix == '.pt':
                    state_dict = torch.load(str(file))
                else:
                    state_dict = safetensors.torch.load_file(str(file))
                analyze_state_dict(state_dict, file.name)
            else:
                if file.suffix == '.pt':
                    convert_pt_to_safetensors(str(file))
                else:
                    print(f"Skipping {file} as it's already in safetensors format")