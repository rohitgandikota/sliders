import torch
import safetensors.torch
from pathlib import Path
import numpy as np
import json

# Global dictionary to store name mappings
layer_name_mappings = {}

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
    
    # # Analyze each LoRA pair
    # for up_key in lora_up_layers:
    #     base_key = up_key.replace('lora_up', '')
    #     down_key = up_key.replace('lora_up', 'lora_down')
        
    #     if down_key in lora_down_layers:
    #         up_weight = state_dict[up_key]
    #         down_weight = state_dict[down_key]
            
    #         print(f"\nLoRA Pair Analysis for {base_key}")
    #         print("-" * 40)
            
    #         # Compute effective weight (without scaling)
    #         effective_weight = (up_weight @ down_weight)
            
    #         stats = {
    #             'up': {
    #                 'min': up_weight.min().item(),
    #                 'max': up_weight.max().item(),
    #                 'mean': up_weight.mean().item(),
    #                 'std': up_weight.std().item()
    #             },
    #             'down': {
    #                 'min': down_weight.min().item(),
    #                 'max': down_weight.max().item(),
    #                 'mean': down_weight.mean().item(),
    #                 'std': down_weight.std().item()
    #             },
    #             'effective': {
    #                 'min': effective_weight.min().item(),
    #                 'max': effective_weight.max().item(),
    #                 'mean': effective_weight.mean().item(),
    #                 'std': effective_weight.std().item()
    #             }
    #         }
            
    #         print(f"Up weights ({up_key}):")
    #         print(f"  Shape: {up_weight.shape}")
    #         print(f"  Range: [{stats['up']['min']:.6f}, {stats['up']['max']:.6f}]")
    #         print(f"  Mean: {stats['up']['mean']:.6f}")
    #         print(f"  Std: {stats['up']['std']:.6f}")
            
    #         print(f"\nDown weights ({down_key}):")
    #         print(f"  Shape: {down_weight.shape}")
    #         print(f"  Range: [{stats['down']['min']:.6f}, {stats['down']['max']:.6f}]")
    #         print(f"  Mean: {stats['down']['mean']:.6f}")
    #         print(f"  Std: {stats['down']['std']:.6f}")
            
    #         print(f"\nEffective weights (up @ down):")
    #         print(f"  Shape: {effective_weight.shape}")
    #         print(f"  Range: [{stats['effective']['min']:.6f}, {stats['effective']['max']:.6f}]")
    #         print(f"  Mean: {stats['effective']['mean']:.6f}")
    #         print(f"  Std: {stats['effective']['std']:.6f}")

def convert_layer_name(old_name):
    """Convert layer names from flux_slider format to flux_ostris format."""
    # If we've seen this name before, return the cached conversion
    if old_name in layer_name_mappings:
        return layer_name_mappings[old_name]
    
    new_name = old_name
    
    # Handle transformer blocks
    if old_name.startswith('lora_unet_transformer_blocks_'):
        
        parts = old_name.split('_')
        block_num = parts[4]  # Get block number
        # print(parts)
        # Convert lora weights
        if 'lora_up' in old_name:
            new_name = old_name.replace('lora_up', 'lora_B')
        elif 'lora_down' in old_name:
            new_name = old_name.replace('lora_down', 'lora_A')
        
        # print(new_name, block_num)
        # Replace prefix and convert to dot notation
        new_name = new_name.replace(
            f'lora_unet_transformer_blocks_{block_num}_attn_to_',
            f'transformer.transformer_blocks.{block_num}.attn.to_'
        ).replace(
            f'lora_unet_transformer_blocks_{block_num}_attn_add_',
            f'transformer.transformer_blocks.{block_num}.attn.add_'
        )
        # print(new_name)
        
    # Handle single transformer blocks
    elif old_name.startswith('lora_unet_single_transformer_blocks_'):
        parts = old_name.split('_')
        block_num = parts[5]  # Get block numbes
        # Convert lora weights
        if 'lora_up' in old_name:
            new_name = old_name.replace('lora_up', 'lora_B')
        elif 'lora_down' in old_name:
            new_name = old_name.replace('lora_down', 'lora_A')
        
        print(new_name)
        # Replace prefix and convert to dot notation
        new_name = new_name.replace(
            f'lora_unet_single_transformer_blocks_{block_num}_attn_to_',
            f'transformer.single_transformer_blocks.{block_num}.attn.to_'
        ).replace(
            f'lora_unet_single_transformer_blocks_{block_num}_attn_add_',
            f'transformer.single_transformer_blocks.{block_num}.attn.add_'
        )
    # Store mapping
    layer_name_mappings[old_name] = new_name
    return new_name

def save_name_mappings(output_path):
    """Save the layer name mappings to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(layer_name_mappings, f, indent=2)
    print(f"\nSaved layer name mappings to {output_path}")

def convert_pt_to_safetensors(pt_path, output_path=None, analyze=True):
    # Load the .pt file
    state_dict = torch.load(pt_path)
    
    # Create new state dict with converted names
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = convert_layer_name(key)
        new_state_dict[new_key] = value
        # raise Exception(f"new_key: {new_key}")
    
    # # Analyze the state dict if requested
    # if analyze:
    #     analyze_state_dict(new_state_dict, Path(pt_path).name)
    
    # If output path is not specified, use the same name but with .safetensors extension
    if output_path is None:
        output_path = str(Path(pt_path).with_suffix('.safetensors'))
    
    # Save as safetensors
    #print 5 keys of the new_state_dict
    print(list(new_state_dict.keys())[:5])
    safetensors.torch.save_file(new_state_dict, output_path)
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
    
    if not args.analyze_only:
        save_name_mappings('outputs/layer_name_mappings.json')
    
# python analysis_scripts/convert_pt_to_st.py --input_path flux-sliders/outputs/person-obese-mod/slider_0.pt --output_path outputs/person-obse-mode.safetensors
# python analysis_scripts/convert_pt_to_st.py --input_path outputs/person-obse-mode.safetensors --analyze_only > outputs/person-obse-mode-layers.txt