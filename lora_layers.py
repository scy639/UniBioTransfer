"""
LoRA (Low-Rank Adaptation) implementation for MLP layers.
Replaces qkv projections in attention and the FFN MLP layers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
USE_LORA :bool = 1  # enable LoRA replacement for MLP layers (and Conv2d)
if USE_LORA:
    LORA_dropout :float = 0.0  # LoRA dropout rate
    LORA_apply_to_conv :bool = 1  # also apply LoRA to Conv2d layers
    LORA_freeze_base :bool = False
    LORA_DEBUG :bool = 0
    FORCE_SAME_RANK_ACROSS_TASKS :bool = 0
    DONT_lora_if_dim_lt :int = 90  # 0: disable. increase for low-dim layers (e.g., in/out conv dim < 32)
    DONT_lora_if_rankFrac_gt :float = 0.3

class LoRALinear(nn.Module):
    """
    LoRA layer that wraps a frozen Linear layer with low-rank adaptation.
    
    Args:
        original_linear: original nn.Linear layer that will be frozen
        rank: LoRA rank (r)
        dropout: dropout probability
    """
    def __init__(
        self, 
        original_linear: nn.Linear,
        rank: int = 4,
        dropout: float = 0.0,
        freeze_base: bool = True,
    ):
        super().__init__()
        
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.scaling = 2.0
        
        # Freeze the original weights
        self.original_linear = original_linear
        if freeze_base:
            for param in self.original_linear.parameters():
                param.requires_grad = False
        
        # LoRA low-rank decomposition: W = W_0 + B @ A, where B: out_features x rank, A: rank x in_features
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        
        # Initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)  # initialize B to 0 so LoRA has no initial effect
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output from the frozen original linear layer
        result = self.original_linear(x)
        
        # LoRA low-rank update: x @ A^T @ B^T
        # x: (..., in_features)
        # lora_A: (rank, in_features) -> A^T: (in_features, rank)
        # lora_B: (out_features, rank) -> B^T: (rank, out_features)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        
        return result + lora_out * self.scaling
    
    def __repr__(self):
        return f"LoRALinear(in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, scaling={self.scaling})"


def replace_linear_with_lora(
    module: nn.Module,
    rank: int = 4,
    dropout: float = 0.0,
    target_modules: list = None,
    verbose: bool = True,
):
    """
    Recursively replace nn.Linear layers within a module with LoRALinear wrappers.
    
    Args:
        module: module whose linear layers should be replaced
        rank: LoRA rank
        dropout: dropout probability
        target_modules: specific module names to replace; None means all linears
                       e.g.: ['to_q', 'to_k', 'to_v', 'to_out'] for attention
                             ['net.0', 'net.2'] for FeedForward
        verbose: whether to log replacements
    
    Returns:
        the module with replacements applied
    """
    replaced_count = 0
    
    for name, child in module.named_children():
        # Skip modules not in the target list (if filtering is enabled)
        if target_modules is not None and name not in target_modules:
            # Continue recursing into child modules
            replace_linear_with_lora(child, rank, dropout, target_modules, verbose)
            continue
        
        if isinstance(child, nn.Linear):
            # Replace with LoRALinear
            lora_layer = LoRALinear(child, rank=rank, dropout=dropout, freeze_base=LORA_freeze_base)
            setattr(module, name, lora_layer)
            replaced_count += 1
            if verbose:
                print(f"[LoRA] Replaced {name}: {child.in_features} -> {child.out_features} with rank={rank}")
        elif isinstance(child, nn.Sequential):
            # Handle Sequential containers (e.g., FeedForward nets)
            new_sequential = nn.Sequential()
            for idx, submodule in enumerate(child):
                if isinstance(submodule, nn.Linear):
                    lora_layer = LoRALinear(submodule, rank=rank, dropout=dropout, freeze_base=LORA_freeze_base)
                    new_sequential.add_module(str(idx), lora_layer)
                    replaced_count += 1
                    if verbose:
                        print(f"[LoRA] Replaced {name}.{idx}: {submodule.in_features} -> {submodule.out_features} with rank={rank}")
                else:
                    new_sequential.add_module(str(idx), submodule)
            setattr(module, name, new_sequential)
        else:
            # Recurse into the remaining submodules
            replace_linear_with_lora(child, rank, dropout, target_modules, verbose)
    
    return module


def count_lora_parameters(module: nn.Module):
    """
    Count LoRA parameters within a module.
    
    Returns:
        dict: {'trainable': trainable params, 'frozen': frozen params, 'total': total params}
    """
    trainable_params = 0
    frozen_params = 0
    
    for name, param in module.named_parameters():
        num_params = param.numel()
        if param.requires_grad:
            trainable_params += num_params
        else:
            frozen_params += num_params
    
    total_params = trainable_params + frozen_params
    
    return {
        'trainable': trainable_params,
        'frozen': frozen_params,
        'total': total_params,
        'trainable_ratio': trainable_params / total_params if total_params > 0 else 0,
    }


def print_lora_parameters(module: nn.Module, name: str = "Model"):
    """Print LoRA parameter statistics."""
    stats = count_lora_parameters(module)
    print(f"\n{'='*60}")
    print(f"{name} Parameter Statistics:")
    print(f"{'='*60}")
    print(f"Trainable params: {stats['trainable']:,} ({stats['trainable_ratio']*100:.2f}%)")
    print(f"Frozen params:    {stats['frozen']:,} ({(1-stats['trainable_ratio'])*100:.2f}%)")
    print(f"Total params:     {stats['total']:,}")
    print(f"{'='*60}\n")


class LoRAConv2d(nn.Module):
    """
    LoRA layer for Conv2d.
    
    Treat Conv2d as a matrix multiplication:
    - flatten kernel: (out_channels, in_channels, k, k) -> (out_channels, in_channels*k*k)
    - apply low-rank decomposition: W = W_0 + B @ A
    
    Args:
        original_conv: original nn.Conv2d layer that will be frozen
        rank: LoRA rank (r)
        dropout: dropout probability
    """
    def __init__(
        self,
        original_conv: nn.Conv2d,
        rank: int = 4,
        dropout: float = 0.0,
        freeze_base: bool = True,
    ):
        super().__init__()
        
        self.out_channels = original_conv.out_channels
        self.in_channels = original_conv.in_channels
        self.kernel_size = original_conv.kernel_size
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups
        
        self.rank = rank
        self.scaling = 2.0
        
        # Freeze the original weights
        self.original_conv = original_conv
        if freeze_base:
            for param in self.original_conv.parameters():
                param.requires_grad = False
        
        # LoRA low-rank decomposition
        # lora_A: (rank, in_channels, kernel_size, kernel_size)
        # lora_B: (out_channels, rank, 1, 1) - via 1x1 convolution
        self.lora_A = nn.Parameter(torch.zeros(
            rank, 
            self.in_channels // self.groups, 
            self.kernel_size[0], 
            self.kernel_size[1]
        ))
        self.lora_B = nn.Parameter(torch.zeros(self.out_channels, rank, 1, 1))
        
        # Initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)  # initialize B to 0
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        print(f"param orig:lora (M) = {self.original_conv.weight.numel()/1024/1024}:{self.lora_A.numel()+self.lora_B.numel()/1024/1024}")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output from the frozen original convolution
        # Use no_grad to avoid computing gradients for the base weights
        result = self.original_conv(x)
        
        # LoRA low-rank update
        # first apply lora_A (down projection) then lora_B (up projection)
        x_dropped = self.dropout(x)
        lora_out = F.conv2d(
            x_dropped,
            self.lora_A,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
        lora_out = F.conv2d(lora_out, self.lora_B)
        
        return result + lora_out * self.scaling
    
    def __repr__(self):
        return (f"LoRAConv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, rank={self.rank}, scaling={self.scaling})")





def _auto_lora_rank(in_features: int, out_features: int) -> int:
    m = min(in_features, out_features)
    r = max(LORA_rank_min, int(round(m / max(1.0, LORA_rank_ratio))))
    if (LORA_rank_max is not None) and (r > LORA_rank_max):
        r = LORA_rank_max
    return max(1, r)

def _svd_low_rank(M: torch.Tensor, rank: int):
    # M: [out, in]
    orig_device = M.device
    orig_dtype = M.dtype
    if 1:
        M = M.to(device=torch.device('cuda'), dtype=torch.float32)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    r = min(rank, U.shape[1], Vh.shape[0])
    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :]
    S_root = torch.sqrt(torch.clamp(S_r, min=0))
    B = U_r @ torch.diag(S_root)          # [out, r]
    A = torch.diag(S_root) @ Vh_r          # [r, in]
    
    B = B.to(device=orig_device, dtype=orig_dtype)
    A = A.to(device=orig_device, dtype=orig_dtype)
    S = S.to(device=orig_device, dtype=orig_dtype)
    return B, A, S


def _svdvals_squared(M: torch.Tensor) -> torch.Tensor:
    # Return squared singular values (energy), sorted in descending order; M: [out, in]
    orig_device = M.device
    orig_dtype = M.dtype
    if 1:
        M = M.to(device=torch.device('cuda'), dtype=torch.float32)
    S = torch.linalg.svdvals(M)
    S2 = (S.float() ** 2)
    return S2.to(device=orig_device, dtype=torch.float32)


def _compute_adaptive_rank_from_S2_list(
    list_S2: list,
    avg_threshold: float = None,
    min_threshold: float = None,
    max_rank: int = None,
) -> int:
    # list_S2: squared singular value vectors (descending) for each matrix
    assert len(list_S2) > 0
    if avg_threshold is None:
        avg_threshold = ADAPTIVE_RANK_AVG_ENERGY_THRESH
    if min_threshold is None:
        min_threshold = ADAPTIVE_RANK_MIN_ENERGY_THRESH

    totals = []
    lengths = []
    for s2 in list_S2:
        assert s2.numel() > 0
        total = s2.sum()
        # Quick fail: zero ΔW has zero energy, so thresholds can't be evaluated
        assert float(total.item()) > 0.0, "Zero energy in weight_diff; cannot determine adaptive rank"
        totals.append(total)
        lengths.append(int(s2.shape[0]))

    R_cap = min(lengths)
    if LORA_rank_max is not None:
        R_cap = min(R_cap, int(LORA_rank_max))
    if max_rank is not None:
        R_cap = min(R_cap, int(max_rank))
    R_cap = max(1, R_cap)

    # Iterate ranks r to see if both average and minimum energy ratios meet thresholds
    for r in range(1, R_cap + 1):
        ratios = []
        for s2, total in zip(list_S2, totals):
            captured = s2[:r].sum()
            ratios.append(float((captured / total).item()))
        avg_ratio = sum(ratios) / len(ratios)
        min_ratio = min(ratios)
        if (avg_ratio >= avg_threshold) and (min_ratio >= min_threshold):
            ret = min(int(R_cap), max(int(LORA_rank_min), int(r)))
            return ret
    # If no rank satisfies both thresholds, fail fast instead of silently degrading
    raise AssertionError(f"No rank satisfies avg>={avg_threshold} and min>={min_threshold} up to R_cap={R_cap}")

def _compute_per_task_ranks_from_S2_list(
    list_S2: list,
    min_threshold: float = None,
    max_rank: int = None,
) -> list:
    # Compute rank per matrix so its energy ratio >= min_threshold (uses min threshold only)
    assert len(list_S2) > 0
    if min_threshold is None:
        min_threshold = ADAPTIVE_RANK_MIN_ENERGY_THRESH
    ret = []
    for i, s2 in enumerate(list_S2):
        assert s2.numel() > 0
        total = s2.sum()
        assert float(total.item()) > 0.0, "Zero energy in weight_diff; cannot determine adaptive rank"
        R_cap = int(s2.shape[0])
        if LORA_rank_max is not None:
            R_cap = min(R_cap, int(LORA_rank_max))
        if max_rank is not None:
            R_cap = min(R_cap, int(max_rank))
        R_cap = max(1, R_cap)
        found = R_cap
        # Task-level threshold: when ranks are allowed to differ, use TASK_2_adaptive_rank_min_energy_thresh
        thres_this = TASK_2_adaptive_rank_min_energy_thresh[i] if (not FORCE_SAME_RANK_ACROSS_TASKS) else min_threshold
        for r in range(1, R_cap + 1):
            ratio = s2[:r].sum() / total
            if float(ratio.item()) >= float(thres_this):
                found = r
                break
        ret.append(int(max(int(LORA_rank_min), int(found))))
    return ret

def compute_adaptive_rank_for_linear_diffs(
    weight_diffs: list,
    avg_threshold: float = None,
    min_threshold: float = None,
    max_rank: int = None,
    per_task: bool = None,
):
    # weight_diffs: List[Tensor [out, in]]
    assert isinstance(weight_diffs, (list, tuple)) and len(weight_diffs) > 0
    if per_task is None:
        per_task = not FORCE_SAME_RANK_ACROSS_TASKS
    list_S2 = [_svdvals_squared(M) for M in weight_diffs]
    out0, in0 = weight_diffs[0].shape
    if per_task:
        ranks = _compute_per_task_ranks_from_S2_list(list_S2, min_threshold, max_rank)
        print(f"[AdaptiveRank-Linear per-task] in={in0} out={out0} ranks={ranks}")
        return ranks
    else:
        ret = _compute_adaptive_rank_from_S2_list(list_S2, None, min_threshold, max_rank)
        print(f"[AdaptiveRank-Linear] in={in0} out={out0} rank={ret}")
        return ret


def compute_adaptive_rank_for_conv_diffs(
    weight_diffs: list,
    avg_threshold: float = None,
    min_threshold: float = None,
    max_rank: int = None,
    per_task: bool = None,
):
    # weight_diffs: List[Tensor [out, in, kH, kW]] -> reshape to [out, in*k*k]
    assert isinstance(weight_diffs, (list, tuple)) and len(weight_diffs) > 0
    if per_task is None:
        per_task = not FORCE_SAME_RANK_ACROSS_TASKS
    list_S2 = []
    for W in weight_diffs:
        out_c, in_c, kH, kW = W.shape
        M = W.reshape(out_c, in_c * kH * kW)
        list_S2.append(_svdvals_squared(M))
    out0, in0, kH0, kW0 = weight_diffs[0].shape
    if per_task:
        ranks = _compute_per_task_ranks_from_S2_list(list_S2, min_threshold, max_rank)
        print(f"[AdaptiveRank-Conv per-task] in_ch={in0} out_ch={out0} kernel=({kH0},{kW0}) ranks={ranks}")
        return ranks
    else:
        ret = _compute_adaptive_rank_from_S2_list(list_S2, None, min_threshold, max_rank)
        print(f"[AdaptiveRank-Conv] in_ch={in0} out_ch={out0} kernel=({kH0},{kW0}) rank={ret}")
        return ret


class LoRAAdapterLinearOnly(nn.Module):
    """
    Incremental LoRA (no base Linear) that returns x @ A^T @ B^T + bias_delta.
    """
    def __init__(self, in_features: int, out_features: int, rank: int = None, dropout: float = 0.0, scaling: float = 1.0, use_bias_delta: bool = True):
        super().__init__()
        if rank is None:
            rank = _auto_lora_rank(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = scaling
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.use_bias_delta = use_bias_delta
        if use_bias_delta:
            self.lora_bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('lora_bias', None)
        # init
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @torch.no_grad()
    def init_from_diff(self, weight_diff: torch.Tensor, bias_diff: torch.Tensor = None):
        # weight_diff: [out, in]
        B, A, S = _svd_low_rank(weight_diff.float(), self.rank)
        self.lora_A.copy_(A.to(self.lora_A.dtype).to(self.lora_A.device))
        self.lora_B.copy_(B.to(self.lora_B.dtype).to(self.lora_B.device))
        if self.use_bias_delta and (bias_diff is not None):
            self.lora_bias.copy_(bias_diff)
        if LORA_DEBUG:
            energy_total = (S.float() ** 2).sum().item()
            energy_top = (S[: self.rank].float() ** 2).sum().item()
            energy_ratio = energy_top / max(1e-12, energy_total)
            approx = (B @ A).to(weight_diff.device).to(weight_diff.dtype)
            err = torch.linalg.norm((approx - weight_diff).float()).item()
            base = torch.linalg.norm(weight_diff.float()).item()
            rel_err = err / max(1e-12, base)
            bias_norm = 0.0 if (bias_diff is None) else float(torch.linalg.norm(bias_diff.float()).item())
            print(f"[LoRA-Linear init] shape={tuple(weight_diff.shape)} rank={self.rank} energy={energy_ratio:.4f} rel_err={rel_err:.6f} bias_norm={bias_norm:.6f}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        if self.lora_bias is not None:
            update = update + self.lora_bias
        return update * self.scaling


class LoRAAdapterConv2dOnly(nn.Module):
    """
    Incremental LoRA for Conv2d: convolve with A then 1x1 B, return the delta.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple, padding: tuple, dilation: tuple, groups: int = 1, rank: int = None, dropout: float = 0.0, scaling: float = 1.0, use_bias_delta: bool = True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        kH, kW = kernel_size
        if rank is None:
            # Estimate rank from the flattened in/out dimensions
            rank = _auto_lora_rank(in_channels * kH * kW, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.rank = rank
        self.scaling = scaling
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        # A: [rank, in/groups, kH, kW]
        self.lora_A = nn.Parameter(torch.zeros(rank, in_channels // groups, kH, kW))
        # B: [out, rank, 1, 1]
        self.lora_B = nn.Parameter(torch.zeros(out_channels, rank, 1, 1))
        self.use_bias_delta = use_bias_delta
        if use_bias_delta:
            self.lora_bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('lora_bias', None)
        # init
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @torch.no_grad()
    def init_from_diff(self, weight_diff: torch.Tensor, bias_diff: torch.Tensor = None):
        # weight_diff: [out, in, kH, kW]
        out_c, in_c, kH, kW = weight_diff.shape
        M = weight_diff.reshape(out_c, in_c * kH * kW)
        B, A, S = _svd_low_rank(M.float(), self.rank)  # B:[out,r], A:[r,in*k*k]
        A_reshaped = A.view(self.rank, in_c, kH, kW)
        self.lora_A.copy_(A_reshaped)
        self.lora_B.copy_(B.view(out_c, self.rank, 1, 1))
        if self.lora_bias is not None and (bias_diff is not None):
            self.lora_bias.copy_(bias_diff)
        if LORA_DEBUG:
            energy_total = (S.float() ** 2).sum().item()
            energy_top = (S[: self.rank].float() ** 2).sum().item()
            energy_ratio = energy_top / max(1e-12, energy_total)
            approx = (B @ A).to(M.device).to(M.dtype)
            err = torch.linalg.norm((approx - M).float()).item()
            base = torch.linalg.norm(M.float()).item()
            rel_err = err / max(1e-12, base)
            bias_norm = 0.0 if (bias_diff is None) else float(torch.linalg.norm(bias_diff.float()).item())
            print(f"[LoRA-Conv init] out_in_k=({out_c},{in_c},{kH}x{kW}) rank={self.rank} energy={energy_ratio:.4f} rel_err={rel_err:.6f} bias_norm={bias_norm:.6f}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_d = self.dropout(x)
        u = F.conv2d(x_d, self.lora_A, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        u = F.conv2d(u, self.lora_B)
        if self.lora_bias is not None:
            u = u + self.lora_bias.view(1, -1, 1, 1)
        return u * self.scaling
