
from typing import List, Set
from natsort import natsorted
from pathlib import Path

def pretty_print_torch_module_keys(
    keys: list,
    indent: int = 4,
    # max_part_num: int = 3,
    # max_examples: int = 2,
    max_part_num: int = 2,
    max_examples: int = 1,
    show_counts: bool = True
) -> None:
    """
    Pretty print PyTorch module keys with hierarchical grouping.
    
    Args:
        keys: List of parameter/buffer keys from state_dict
        max_part_num: Maximum number of dot-separated parts to show (0=no truncation)
        indent: Number of spaces for indentation
        max_examples: Maximum example keys to show per group
        show_counts: Whether to show count of keys in each group
    """
    # Group keys by their truncated prefix
    from collections import defaultdict
    groups = defaultdict(list)
    for key in keys:
        if max_part_num <= 0:  # No truncation
            groups[key].append(key)
        else:
            # Split into parts and rejoin the first N parts
            parts = key.split('.')
            prefix = '.'.join(parts[:max_part_num]) if len(parts) > max_part_num else key
            groups[prefix].append(key)

    for prefix, members in sorted(groups.items()):
        _s = f"{' ' * indent}{prefix}"
        count_str = f" ({len(members)} keys)" if show_counts else ""
        # _s += f"{count_str}:"
        print(_s)
        
        # Show example keys (full paths)
        examples = members[:max_examples]
        for ex in examples:
            # print(f"{' ' * (indent * 2)}- {ex[len(prefix):]}")
            print(f"{' ' * (indent * 2)}{ex[len(prefix):]}")
        
        if len(members) > max_examples:
            print(f"{' ' * (indent * 2)}... (and {len(members) - max_examples} more)")




def get_representative_moduleNames(
    all_keys: List[str], 
    ignore_prefixes: tuple = tuple(),
    keep_index: int = 0, treat_alpha_digit: bool = True) -> Set[str]:
    """
    Filter state dict keys to keep only representative items (specific index in any numbered sequence).
    Args:
        all_keys: List of all keys from state_dict (all are leaf nodes)
            eg. ['learnable_vector', 'model.diffusion_model.time_embed.0.weight', 'model.diffusion_model.time_embed.0.bias',
        keep_index: Which index to keep when multiple numbered items exist (default 0 for first)
        treat_alpha_digit: If True, also treat letter+digit combinations (e.g., 'attn1', 'attn2') as numbered sequences
    Returns:
        Set of filtered keys preserving only representative items
    """
    import re
    if ignore_prefixes:
        all_keys = [k for k in all_keys if not any(k.startswith(p) for p in ignore_prefixes)]
    num_pattern = re.compile(r'\.(\d+)\.') # Pattern to match numbers in paths (e.g., '.0.', '.1.', etc.)
    # Group keys by their pattern (replace numbers with X for grouping)
    from collections import defaultdict
    groups = defaultdict(list)
    
    for key in all_keys:
        # Create a pattern by replacing all numbers with 'X'
        pattern = re.sub(r'\.(\d+)\.', '.X.', key)
        # Also handle numbers at the end of the key
        pattern = re.sub(r'\.(\d+)$', '.X', pattern)
        
        if treat_alpha_digit:
            # Also replace letter+digit combinations (e.g., 'attn1' -> 'attnX')
            pattern = re.sub(r'\.([a-zA-Z]+)(\d+)\.', r'.\1X.', pattern)
            pattern = re.sub(r'\.([a-zA-Z]+)(\d+)$', r'.\1X', pattern)
        
        groups[pattern].append(key)
    # print(f"Debug groups: {groups}")
    
    filtered_keys = []
    for pattern, keys_in_group in groups.items():
        if len(keys_in_group) == 1:
            # Only one key in this pattern group - keep it
            filtered_keys.extend(keys_in_group)
        else:
            # Multiple keys - find the one with the target index
            def get_numeric_indices(key):
                # Extract all numeric indices from the key (pure numbers)
                matches = re.findall(r'\.(\d+)(?:\.|$)', key)
                indices = [int(x) for x in matches]
                
                if treat_alpha_digit:
                    # Also extract indices from letter+digit combinations
                    alpha_digit_matches = re.findall(r'\.([a-zA-Z]+)(\d+)(?:\.|$)', key)
                    for _, digit in alpha_digit_matches:
                        indices.append(int(digit))
                
                return tuple(indices)
            
            # Sort by numeric indices 
            keys_in_group.sort(key=get_numeric_indices)
            
            # Try to find the key with the desired index
            target_found = False
            for key in keys_in_group:
                if treat_alpha_digit:
                    # For alpha+digit mode, check if any alpha+digit combination has the target index
                    alpha_digit_matches = re.findall(r'\.([a-zA-Z]+)(\d+)(?:\.|$)', key)
                    for prefix, digit in alpha_digit_matches:
                        if int(digit) == keep_index:
                            filtered_keys.append(key)
                            target_found = True
                            break
                    if target_found:
                        break
                else:
                    # For normal mode, check pure numeric indices
                    indices = get_numeric_indices(key)
                    # Check if the first (primary) index matches keep_index  
                    if indices and indices[0] == keep_index:
                        filtered_keys.append(key)
                        target_found = True
                        break
            
            # If target index not found, fall back to the first available
            if not target_found:
                filtered_keys.append(keys_in_group[0])
    
    filtered_keys = natsorted(filtered_keys)
    return filtered_keys

def get_no_grad_and_has_grad_keys(
    model, only_representative: bool = True, 
    ignore_prefixes: tuple = tuple(),
    verbose: int = 1,  # for print (not for file save. for save, we log all ) 0,1: only print at last, 2: print at each step
    get_representative_moduleNames_at_first :bool = False,
    save_path: str = None,  # if not None, save detailed log to file
):
    # don't use state_dict() (it lacks gradient information)
    all_params = dict(model.named_parameters())
    keys = list(all_params.keys())
    
    # For file logging, collect all messages
    log_messages = []
    
    def print_(*msg, verb=1):
        if verbose >= verb:
            print(*msg)
        if save_path is not None:
            log_messages.extend(msg)
    
    if only_representative and get_representative_moduleNames_at_first:
        keys = get_representative_moduleNames(keys, ignore_prefixes=ignore_prefixes)
    
    k_has_grad = []
    k_no_grad = []  # dont require grad or .grad is 0
    
    for name in keys:
        if name not in all_params:
            print_(f"{name} not found in named_parameters (might be buffer)", verb=3)
            k_no_grad.append(name)
            continue
            
        param = all_params[name]
        if param.requires_grad:
            if param.grad is None:
                print_(f"{name} has grad but grad is None", verb=3)
                k_no_grad.append(name)
            elif param.grad.sum() == 0:
                print_(f"{name} has grad but grad is 0", verb=3)
                k_no_grad.append(name)
            else:
                print_(f"{name} has grad !=0", verb=4)
                k_has_grad.append(name)
        else:
            k_no_grad.append(name)
    if only_representative and not get_representative_moduleNames_at_first:
        k_no_grad  = get_representative_moduleNames(k_no_grad,  ignore_prefixes=ignore_prefixes)
        k_has_grad = get_representative_moduleNames(k_has_grad, ignore_prefixes=ignore_prefixes)
            
    print_("No grad:", verb=2)
    for name in k_no_grad:
        print_(f"  - {name}", verb=2)
    print_("Has grad:", verb=2)
    if 0:
        print_("<skip.>", verb=2)
    else:
        for name in k_has_grad:
            print_(f"  - {name}", verb=2)
    print_(f"Total: {len(k_no_grad) + len(k_has_grad)} {len(k_has_grad)=}", verb=1)
    
    if save_path is not None:
        Path(save_path).write_text('\n'.join(log_messages), encoding='utf-8')  # !diskW
        print(f"> {save_path}")
    
    return k_has_grad, k_no_grad






if __name__=='__main__':
    # Example usage:
    all_keys = [
        'face_ID_model.facenet.input_layer.0.weight',
        'face_ID_model.facenet.input_layer.1.weight',
        'face_ID_model.facenet.input_layer.1.bias',
        'face_ID_model.facenet.input_layer.1.running_mean',
        'face_ID_model.facenet.input_layer.1.running_var',
        'face_ID_model.facenet.input_layer.1.num_batches_tracked',
        'face_ID_model.facenet.input_layer.2.weight',
        
        'learnable_vector',
        'model.diffusion_model_refNet.time_embed.0.weight',
        'model.diffusion_model_refNet.time_embed.0.weight.xxx',
        'model.diffusion_model_refNet.time_embed.0.bias',
        'model.diffusion_model_refNet.time_embed.0.xxxx.0',
        'model.diffusion_model_refNet.time_embed.0.xxxx.1',
        'model.diffusion_model_refNet.time_embed.0.xxxx.2',
        
        'model.diffusion_model_refNet.time_embed.1.weight',
        'model.diffusion_model_refNet.time_embed.1.bias',
        'model.diffusion_model_refNet.time_embed.0.submodule.param',
        'model.diffusion_model_refNet.time_embed.1.submodule.param',
        'model.diffusion_model_refNet.input_blocks.0.weight',
        'model.diffusion_model_refNet.input_blocks.1.weight',
        'model.diffusion_model_refNet.middle_block.0.weight',
        'model.diffusion_model_refNet.output_blocks.0.bias',
        'model.diffusion_model_refNet.output_blocks.1.bias',
        'model.diffusion_model_refNet.output_blocks.2.bias',

        'model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_k.weight',
        'model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_out.0.bias',
        'model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_out.0.weight',
        'model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_q.weight',
        'model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_v.weight',
        'model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_k.weight',
        'model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_out.0.bias',
        'model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_q.weight',
        'model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn3.xxxxx',
    ]
    

    import torch
    sd = torch.load('checkpoints/pretrained.ckpt')
    all_keys = sd['state_dict'].keys()
    filtered = get_representative_moduleNames(all_keys)
    print(f"Filtered representative keys (keep_index=0, default):")
    for key in sorted(filtered):
        print(f"  - {key}")
    
