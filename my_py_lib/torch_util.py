import torch
def count_model_params(model, log=False)->int:
    total_params = sum(p.numel() for p in model.parameters())
    if log:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params
def list_layers(model):
    """
    Lists each layer's name, type, and parameter size in a PyTorch model.
    """
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Sequential):
            continue  # Skip sequential layers
        
        layer_info = {}
        layer_info["name"] = name
        layer_info["type"] = str(type(module))
        
        params = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        layer_info["params"] = params
        
        layers.append(layer_info)
    
    return layers

def recursive_to(data: dict, device: torch.device) -> dict:
    """Recursively move all tensors in a nested structure to the target device."""
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.to(device, non_blocking=True)
        elif isinstance(value, dict):
            data[key] = recursive_to(value, device)
    return data

def cleanup_gpu_memory():
    import gc
    if torch.cuda.is_available():
        gc.collect() # Force garbage collection
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        # Clear any remaining cached allocations
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            torch.cuda.reset_peak_memory_stats()
        print(f"GPU memory cleaned. Allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB, "
              f"Cached: {torch.cuda.memory_reserved()/1024**3:.2f}GB")

def custom_repr_v3(self):
    stats = []
    if self.numel() > 0:
        dtype_str = str(self.dtype).replace('torch.', '')
        stats.append(dtype_str)
        stats.append(f"μ={self.float().mean().item():.2f}")
        stats.append(f"{self.min().item():.2f}~{self.max().item():.2f}")
        stats.append(f"med={self.float().median().item():.2f}")
        if  1  :
            uniques = torch.unique(self.flatten())
            if len(uniques) <= 6:
                stats.append(f"uniq={uniques.tolist()}")
            else:
                stats.append(f"uniq=[{uniques[0].item():.2f},...,{uniques[-1].item():.2f}]")
    return f'<T {str(tuple(self.shape))[1:-1]} {" ".join(stats)}>'

def to_device(obj, device, *args, **kwargs):
    """
    Recursively moves tensors in a nested structure to the specified device,

    Args:
      device: The target PyTorch device (e.g., 'cuda:0' or 'cpu').
      *args:
      **kwargs: Keyword arguments to be passed to the tensor.to() method 
                (e.g., non_blocking=True).

    Returns:
      The object with all tensors moved to the specified device.
    """
    if torch.is_tensor(obj): # Pass the device and any additional arguments to the .to() method
        return obj.to(device, *args, **kwargs)
    elif isinstance(obj, dict): # Recursively call to_device on each value in the dictionary
        return {k: to_device(v, device, *args, **kwargs) for k, v in obj.items()}
    elif isinstance(obj, list): # Recursively call to_device on each element in the list
        return [to_device(elem, device, *args, **kwargs) for elem in obj]
    else: # Return the object unchanged if it's not a tensor, dict, or list
        return obj