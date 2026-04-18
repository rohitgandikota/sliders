import re
from collections import defaultdict

def analyze_lora_layers(filepath):
    # Dictionary to store parameter counts for each block and sub-block
    param_counts = defaultdict(lambda: defaultdict(int))
    
    # Regular expression to parse layer names and shapes
    layer_pattern = r'(lora_\w+)_(\d+)?_?(\w+)?'
    
    with open(filepath, 'r') as f:
        for line in f:
            if 'torch.Size' not in line:
                continue
                
            # Parse line into name and shape
            parts = line.strip().split(' ')
            layer_name = parts[0]
            shape_str = parts[1]
            
            # Extract shape dimensions
            shape = ""#eval(shape_str)
            
            # Calculate parameters (multiply all dimensions)
            params = 1
            for dim in shape:
                params *= dim if isinstance(dim, int) else 1
                
            # Parse layer name to get block hierarchy
            name_parts = layer_name.split('.')
            base_name = name_parts[0]
            
            # Extract block and sub-block information
            match = re.match(layer_pattern, base_name)
            if match:
                main_block = match.group(1)
                block_num = match.group(2)
                block_type = match.group(3)
                
                # Update parameter counts
                if block_num:
                    full_block = f"{main_block}_{block_num}"
                    param_counts[main_block][block_num] += params
                else:
                    param_counts[main_block]["total"] += params

    # Generate detailed report
    report = []
    total_params = 0
    
    for main_block, sub_blocks in param_counts.items():
        block_total = sum(sub_blocks.values())
        total_params += block_total
        
        report.append(f"{main_block} - {block_total}")
        
        # Add sub-block details
        for sub_block, params in sub_blocks.items():
            if sub_block != "total":
                report.append(f"    {main_block}_{sub_block} - {params}")
                
    report.append(f"\nTotal Parameters: {total_params}")
    
    return "\n".join(report)

# Usage
filepath = "outputs/smiling_xl_layers_info.txt"
detailed_breakdown = analyze_lora_layers(filepath)
print(detailed_breakdown)