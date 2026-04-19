from ldm.modules.attention import *
import global_
import torch
import torch.nn as nn
import torch.nn.functional as F
from my_py_lib.torch_util import custom_repr_v3
from confs import *
import cv2, numpy as np
from lmk_util.lmk_extractor import lmkAll_2_lmkMain, get_lmkMain_indices
from MoE import *
from lora_layers import *
import json
import copy



"""
Global knobs for shared experts and routing (no argparse per user preference)
"""
NUM_SHARED_FFN = 8
GATE_TOPK = 2

# Sparse MoE FFN for all FFN blocks (in addition to shared orig + task LoRA)
# default off to keep behavior unchanged; enable by setting EXTRA_MoE_enable to True
EXTRA_MoE_enable :bool = 1
EXTRA_MoE_num_ep = 8        # number of sparse MoE experts (narrow FFN)
EXTRA_MoE_inner_divisor = 128    # each expert intermediate dim = original FFN intermediate dim * this ratio
EXTRA_MoE_topK = 2              # sparse routing selects top-k experts (k fixed to 2)
EXTRA_MoE_add_noise :bool = 1     # add random noise to routing scores for exploration
EXTRA_MoE_noise_std = 0.1       # noise strength (Gaussian standard deviation)
EXTRA_MoE_en_auxLoss :bool = 0  # compute load-balancing auxiliary loss
EXTRA_MoE_aux_coef = 1e-2       # coefficient for auxiliary loss when adding to total loss
EXTRA_MoE_routing_mode = 'sparse'  # 'sparse' | 'dense'
LMK_PICK_IDX = None
NUM_lmk_pick = len(LMK_PICK_IDX) if LMK_PICK_IDX is not None else len(get_lmkMain_indices(include_face_oval=True))
print(f"{NUM_lmk_pick=}")
IMAGE_SIZE_FOR_LMK_NORM = 512.0

def _log2(orig_modules, lora_modules):
    """Calculate and log parameter statistics for original and LoRA modules"""
    # Calculate original module stats
    orig_params = sum(p.numel() for p in orig_modules.parameters())
    orig_size = sum(p.numel() * p.element_size() for p in orig_modules.parameters())
    # Calculate LoRA stats (handle both single module and tuple/list)
    if isinstance(lora_modules, (list, tuple)):
        lora_params = sum(p.numel() for m in lora_modules for p in m.parameters())
        lora_size = sum(p.numel() * p.element_size() for m in lora_modules for p in m.parameters())
        # Try to get rank from lora modules
        ranks = []
        for m in lora_modules:
            if hasattr(m, 'rank'):
                ranks.append(m.rank)
        if len(ranks) == 2:
            rank_str = f" (rank_in={ranks[0]} rank_out={ranks[1]})"
        elif len(ranks) == 1:
            rank_str = f" (rank={ranks[0]})"
        else:
            rank_str = ""
    else:
        lora_params = sum(p.numel() for p in lora_modules.parameters())
        lora_size = sum(p.numel() * p.element_size() for p in lora_modules.parameters())
        # Try to get rank from lora module
        if hasattr(lora_modules, 'rank'):
            rank_str = f" (rank={lora_modules.rank})"
        else:
            rank_str = ""
    msg1 = f"orig: {orig_params:,} params, {orig_size/1024/1024:.2f}MB"
    msg2 = f"LoRA: {lora_params:,} params, {lora_size/1024/1024:.2f}MB{rank_str}"
    for msg in [msg1, msg2]:
        print(msg)
        continue
        with open(_verify_log_file, 'a') as f:
            f.write(msg + '\n')
def _log1(msg: str):
    """Print message and append to log file"""
    print(msg)
    return
    with open(_verify_log_file, 'a') as f:
        f.write(msg + '\n')

def build_ffn_gate_input_common(x: torch.Tensor, token_pos_grid__cur, tasks: list):
    """Build gate input for FFN routing (reusable across FFN classes)."""
    b, n, d = x.shape
    token_feat = x # token
    avg_feat = x.mean(dim=1, keepdim=True).expand(-1, n, -1) # avg(all tokens)
    len_task = len(tasks) # task one-hot
    task_1h = x.new_zeros(b, len_task)
    task_1h[:, global_.task] = 1
    task_1h = task_1h.unsqueeze(1).expand(-1, n, -1)
    token_pos = token_pos_grid__cur # token-position from global_.token_pos_grid__cur
    assert token_pos.shape[:2] == (b, n), (token_pos.shape, (b, n), )
    rel_flat = x.new_zeros(b, n, 2*NUM_lmk_pick) # token-relative-position to lmks
    lmk = global_.lmk_
    if 1:
        lmk = lmk.to(x.device).float()# TODO to check is it normed already?
        if LMK_PICK_IDX is None:
            assert NUM_lmk_pick==lmk.shape[1]
        else:
            lmk = lmk[:, LMK_PICK_IDX, :]
        rel = token_pos.unsqueeze(2) - lmk.unsqueeze(1)  # [b,n,L,2]
        rel_flat = rel.reshape(b, n, -1)
    token_pos = token_pos * 2.0 - 1.0 # [0,1] -> [-1,1]
    gate_in = torch.cat([token_feat, avg_feat, task_1h, token_pos, rel_flat], dim=-1)
    ctx = {'token_feat': token_feat, 'avg_feat': avg_feat, 'task_1h': task_1h, 'token_pos': token_pos, 'lmk': lmk, 'rel': rel, 'rel_flat': rel_flat}
    return gate_in, ctx

def replace_modules_lossless(
    module: nn.Module,
    src_modules: list,
    l_task: list,
    parent_name: str = "",
    depth :int = 0,
    for_refnet: bool = False,
):
    """
    Apply policy:
    - FFN: shared-plus-task (lossless upcycle)
    - CrossAttention linear projections (to_q, to_k, to_v, to_out.0): shared-plus-task
    - Conv2d: keep task-specific or wrap with shared-plus-task if desired
    - Norms: keep task-specific (LayerNorm/GroupNorm)
    """
    if depth==0:
        CONV2D_PARAM_STATS.clear()
    # Skip modules with no parameters
    if len(list(module.parameters())) == 0:
        # print(f'[replace_modules_lossless] Skipping module with no parameters: {module}')
        return module
    if len(list(module.named_children()))==0:
        print('\n!!!!   len(list(module.named_children()))==0',module)
        assert 0
    for name, child in module.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else f".{name}"
        src_child_modules = [getattr(src_module, name) for src_module in src_modules]
        if len({id(s) for s in src_child_modules}) < len(src_child_modules):
            raise Exception('Duplicate source modules detected!')
            # if sources are the same instance(s), clone to ensure distinct expert modules
            src_child_modules = [copy.deepcopy(src_child_modules[0]) for _ in src_child_modules]

        if isinstance(child, FeedForward):
            if 0:
                setattr(module, name, TaskSpecific_MoE([s for s in src_child_modules], tasks=l_task))
            else:
                # FFN -> shared average + per-task LoRA
                setattr(module, name, upCycle_module(src_child_modules, l_task, module_name=full_name))
            continue

        if isinstance(child, CrossAttention):
            # replace linear projections
            # if for_refnet:
            if 0:
                for proj_name in ["to_q", "to_k", "to_v"]:
                    src_proj_list = [getattr(s, proj_name) for s in src_child_modules]
                    setattr(child, proj_name, upCycle_module(src_proj_list, l_task, module_name=f"{full_name}.{proj_name}"))
                if hasattr(child.to_out, "__getitem__"):
                    src_linear0 = [s.to_out[0] for s in src_child_modules]
                    child.to_out[0] = upCycle_module(src_linear0, l_task, module_name=f"{full_name}.to_out.0")
            else:
                for proj_name in ["to_q", "to_k", "to_v"]:
                    src_proj_list = [getattr(s, proj_name) for s in src_child_modules]
                    setattr(child, proj_name, TaskSpecific_MoE([s for s in src_proj_list], tasks=l_task) )
                if hasattr(child.to_out, "__getitem__"):
                    src_linear0 = [s.to_out[0] for s in src_child_modules]
                    child.to_out[0] = TaskSpecific_MoE([s for s in src_linear0], tasks=l_task)
            continue

        if isinstance(child, nn.Conv2d):
            num_params = sum(p.numel() for p in child.parameters())
            CONV2D_PARAM_STATS.append((num_params, full_name))
            # if num_params > CONV2D_PARAM_MOE_THRES and (not any(full_name.startswith(p) for p in FORCE_TASKSPEC_PREFIXES)):
            if 1:
                printC(f"shared+LoRA Conv2d",f"{full_name}")
                setattr(module, name, upCycle_module(src_child_modules, l_task, module_name=full_name))
            else:
                setattr(module, name, TaskSpecific_MoE([s for s in src_child_modules], tasks=l_task))
            continue
        elif isinstance(child, (nn.LayerNorm, nn.GroupNorm)):
            setattr(module, name, TaskSpecific_MoE([s for s in src_child_modules], tasks=l_task))
            continue
        elif isinstance(child, nn.Linear):
            # default linear: task-specific
            setattr(module, name, TaskSpecific_MoE([s for s in src_child_modules], tasks=l_task))
            continue
        else:
            replace_modules_lossless(child, src_child_modules, l_task, parent_name=full_name, depth=depth+1, for_refnet=for_refnet)

    if depth==0:
        stats_sorted = sorted(CONV2D_PARAM_STATS, key=lambda x: x[0], reverse=True)
        if gate_("[Conv2d param stats] count, name (sorted desc):"):
            for cnt, n in stats_sorted:
                print(f"  {cnt:12d}  {n}")
    return module

def upCycle_module(l_modules, l_task, module_name: str = None):
    assert len(  set(  [type(m) for m in l_modules]  )  ) == 1
    m0 = l_modules[0]
    if isinstance(m0, FeedForward):
        obj = FFN_Shared_Plus_TaskLoRA(l_modules, l_task, module_name=module_name)
    elif isinstance(m0, nn.Linear):
        obj = Linear_Shared_Plus_TaskLoRA(l_modules, l_task, module_name=module_name)
    elif isinstance(m0, nn.Conv2d):
        obj = Conv_Shared_Plus_TaskLoRA(l_modules, l_task, module_name=module_name)
    else:
        raise Exception(module_name,m0)
        return TaskSpecific_MoE([s for s in l_modules], tasks=l_task)
    if obj.dont_lora:
        return TaskSpecific_MoE([s for s in l_modules], tasks=l_task)
    return obj




class ResidualAdapterLinearOnly(nn.Module):
    """
    Full-rank residual adapter returning the linear delta (orig - shared).
    """
    def __init__(self, in_features: int, out_features: int, scaling: float = 1.0, use_bias_delta: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(in_features, out_features)
        self.scaling = scaling
        self.use_bias_delta = use_bias_delta
        self.delta_weight = nn.Parameter(torch.zeros(out_features, in_features))
        if use_bias_delta:
            self.delta_bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('delta_bias', None)
    @torch.no_grad()
    def init_from_diff(self, weight_diff: torch.Tensor, bias_diff: torch.Tensor = None):
        self.delta_weight.copy_(weight_diff)
        if (self.delta_bias is not None) and (bias_diff is not None):
            self.delta_bias.copy_(bias_diff)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = x @ self.delta_weight.T
        if self.delta_bias is not None:
            update = update + self.delta_bias
        return update * self.scaling

class ResidualAdapterConv2dOnly(nn.Module):
    """
    Full-rank residual adapter for Conv2d, returning the convolutional delta (orig - shared).
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple, padding: tuple, dilation: tuple, groups: int = 1, scaling: float = 1.0, use_bias_delta: bool = True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        kH, kW = kernel_size
        self.rank = min(out_channels, in_channels * kH * kW)
        self.scaling = scaling
        self.use_bias_delta = use_bias_delta
        self.delta_weight = nn.Parameter(torch.zeros(out_channels, in_channels // groups, kH, kW))
        if use_bias_delta:
            self.delta_bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('delta_bias', None)
    @torch.no_grad()
    def init_from_diff(self, weight_diff: torch.Tensor, bias_diff: torch.Tensor = None):
        self.delta_weight.copy_(weight_diff)
        if (self.delta_bias is not None) and (bias_diff is not None):
            self.delta_bias.copy_(bias_diff)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = F.conv2d(x, self.delta_weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        if self.delta_bias is not None:
            u = u + self.delta_bias.view(1, -1, 1, 1)
        return u * self.scaling



class Linear_TaskSpecific_Plus_Shared(nn.Module):
    def __init__(self, l_proj: list, l_task: list):
        super().__init__()
        assert len(l_proj) >= 1
        p0 = l_proj[0]
        assert isinstance(p0, nn.Linear)
        in_f, out_f = p0.in_features, p0.out_features
        bias = p0.bias is not None
        self.shared = nn.Linear(in_f, out_f, bias=bias)
        self.shared = zero_module(self.shared)
        self.tasks = l_task
        self.task_proj = ModuleDict_W(l_proj, self.tasks)

    def forward(self, x):
        t = global_.task
        return self.task_proj[t](x) + self.shared(x)


class Conv_TaskSpecific_Plus_Shared(nn.Module):
    def __init__(self, l_conv: list, l_task: list):
        super().__init__()
        assert len(l_conv) >= 1
        c0 = l_conv[0]
        assert isinstance(c0, nn.Conv2d)
        self.shared = nn.Conv2d(c0.in_channels, c0.out_channels, kernel_size=c0.kernel_size, stride=c0.stride, padding=c0.padding, dilation=c0.dilation, groups=c0.groups, bias=(c0.bias is not None), padding_mode=c0.padding_mode)
        self.shared = zero_module(self.shared)
        self.tasks = l_task
        self.task_conv = ModuleDict_W(l_conv, self.tasks)

    def forward(self, x):
        t = global_.task
        return self.task_conv[t](x) + self.shared(x)




def _average_state_dict(modules: list):
    assert len(modules) > 0
    sd0 = modules[0].state_dict()
    avg = {k: torch.zeros_like(v) for k, v in sd0.items()}
    for m in modules:
        msd = m.state_dict()
        for k in avg:
            avg[k] += msd[k]
    for k in avg:
        avg[k] /= len(modules)
    return avg


class FFN_Shared_Plus_TaskLoRA(nn.Module):
    def __init__(self, l_ffn: list, l_task: list, module_name: str = None):
        super().__init__()
        self.module_name = module_name
        # _log1(f"-------- {module_name} --------")
        assert len(l_ffn) >= 1
        self.tasks = l_task
        self.num_tasks = len(l_task)
        self.dont_lora = False
        f0: FeedForward = l_ffn[0]
        # build shared from f0 and load avg
        self.shared_ffn: FeedForward = copy.deepcopy(f0)
        if FOR_upcycle_ckpt_GEN_or_USE:
            avg_sd = _average_state_dict(l_ffn)
            self.shared_ffn.load_state_dict(avg_sd)
        # freeze shared
        for p in self.shared_ffn.parameters():
            p.requires_grad = False
        # discover inner layers
        self.is_glu = isinstance(self.shared_ffn.net[0], GEGLU)
        if self.is_glu:
            in_linear: nn.Linear = self.shared_ffn.net[0].proj
        else:
            assert isinstance(self.shared_ffn.net[0], nn.Sequential)
            in_linear: nn.Linear = self.shared_ffn.net[0][0]
        out_linear: nn.Linear = self.shared_ffn.net[2]
        self.in_features = in_linear.in_features
        self.mid_features = in_linear.out_features
        self.out_features = out_linear.out_features
        if 1: # cal/read adaptive rank across tasks
            if FOR_upcycle_ckpt_GEN_or_USE:
                w_diff_in_list = []
                w_diff_out_list = []
                for f in l_ffn:
                    if self.is_glu:
                        tin: nn.Linear = f.net[0].proj
                    else:
                        tin: nn.Linear = f.net[0][0]
                    tout: nn.Linear = f.net[2]
                    w_diff_in_list.append(tin.weight.data - in_linear.weight.data)
                    w_diff_out_list.append(tout.weight.data - out_linear.weight.data)
                if FORCE_SAME_RANK_ACROSS_TASKS:
                    rank_in = compute_adaptive_rank_for_linear_diffs(w_diff_in_list)
                    rank_out = compute_adaptive_rank_for_linear_diffs(w_diff_out_list)
                    global_.moduleName_2_adaRank[module_name] = [rank_in, rank_out]
                else:
                    ranks_in = compute_adaptive_rank_for_linear_diffs(w_diff_in_list, per_task=True)
                    ranks_out = compute_adaptive_rank_for_linear_diffs(w_diff_out_list, per_task=True)
                    global_.moduleName_2_adaRank[module_name] = [ranks_in, ranks_out]
            else:
                r_info = global_.moduleName_2_adaRank[module_name]
                if FORCE_SAME_RANK_ACROSS_TASKS:  rank_in,  rank_out  = r_info
                else:                             ranks_in, ranks_out = r_info
        if 1:
            # fallback decision: (1) tiny feature dims
            min_dim_in  = min(self.in_features, self.mid_features)
            min_dim_out = min(self.mid_features, self.out_features)
            if (min_dim_in < DONT_lora_if_dim_lt) or (min_dim_out < DONT_lora_if_dim_lt):
                # print(f"[LoRA fallback][FFN] {module_name} {min_dim_in=} {min_dim_out=} {DONT_lora_if_dim_lt=}")
                self.dont_lora = True;  return
        # per-task LoRA adapters
        _l_in = []
        _l_out = []
        for idx, f in enumerate(l_ffn):
            if self.is_glu:
                tin: nn.Linear = f.net[0].proj
            else:
                tin: nn.Linear = f.net[0][0]
            tout: nn.Linear = f.net[2]
            if not FORCE_SAME_RANK_ACROSS_TASKS:
                rank_in = ranks_in[idx]
                rank_out = ranks_out[idx]
            frac_in = float(rank_in)  / min(self.in_features, self.mid_features)
            frac_out = float(rank_out) / min(self.mid_features, self.out_features)
            frac_avg = 0.5 * (frac_in + frac_out)
            if frac_avg > DONT_lora_if_rankFrac_gt:
                lora_in = ResidualAdapterLinearOnly(self.in_features, self.mid_features, scaling=1.0, use_bias_delta=True)
                lora_out = ResidualAdapterLinearOnly(tout.in_features, tout.out_features, scaling=1.0, use_bias_delta=True)
            else:
                lora_in = LoRAAdapterLinearOnly(self.in_features, self.mid_features, rank=rank_in, dropout=0.0, scaling=1.0)
                lora_out = LoRAAdapterLinearOnly(tout.in_features, tout.out_features, rank=rank_out, dropout=0.0, scaling=1.0)
            # init from diffs
            if FOR_upcycle_ckpt_GEN_or_USE:
                with torch.no_grad():
                    w_diff_in = tin.weight.data - in_linear.weight.data
                    b_diff_in = (tin.bias.data - in_linear.bias.data) if tin.bias is not None else None
                    lora_in.init_from_diff(w_diff_in, b_diff_in)
                    w_diff_out = tout.weight.data - out_linear.weight.data
                    b_diff_out = (tout.bias.data - out_linear.bias.data) if tout.bias is not None else None
                    lora_out.init_from_diff(w_diff_out, b_diff_out)
            _l_in.append(lora_in)
            _l_out.append(lora_out)
        self.task_lora_in = ModuleDict_W(_l_in, self.tasks)
        self.task_lora_out = ModuleDict_W(_l_out, self.tasks)
        # reuse dropout and activation behavior
        self.dropout_p = self.shared_ffn.net[1].p if isinstance(self.shared_ffn.net[1], nn.Dropout) else 0.0
        self.dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()

        # Sparse/Dense MoE experts (small inner dim) + gate
        if EXTRA_MoE_enable:
            small_inner = self.mid_features // EXTRA_MoE_inner_divisor
            self.num_moe_expert = EXTRA_MoE_num_ep
            gate_in_dim = self.in_features + self.in_features + len(self.tasks) + 2 + 2*NUM_lmk_pick
            self.moe_gate_mlp = nn.Linear(gate_in_dim, self.num_moe_expert)

            if EXTRA_MoE_routing_mode == 'dense':
                self.moe_experts_batched = BatchedFeedForward(
                    dim=self.in_features, dim_out=self.out_features,
                    glu=self.is_glu, dropout=self.dropout_p,
                    inner_dim=small_inner, num_expert=self.num_moe_expert,
                )
            else:
                mult = small_inner / self.in_features
                experts = []
                for _ in range(self.num_moe_expert):
                    expert = FeedForward(self.in_features, dim_out=self.out_features, mult=mult, glu=self.is_glu, dropout=self.dropout_p)
                    experts.append(expert)
                self.moe_experts_list = nn.ModuleList(experts)

        if FOR_upcycle_ckpt_GEN_or_USE:
            self.verify_approximation(orig_ffn_list=l_ffn)

    def forward(self, x: torch.Tensor, token_pos_grid__cur=None):
        t = global_.task
        # in linear + LoRA
        if self.is_glu:
            base = self.shared_ffn.net[0].proj(x)
            delta = self.task_lora_in[t](x)
            z = base + delta
            v, gate = z.chunk(2, dim=-1)
            h = v * F.gelu(gate)
        else:
            base = self.shared_ffn.net[0][0](x)
            delta = self.task_lora_in[t](x)
            h = F.gelu(base + delta)
        h = self.dropout(h)
        # out linear + LoRA
        y_base = self.shared_ffn.net[2](h)
        y_delta = self.task_lora_out[t](h)
        y = y_base + y_delta
        if EXTRA_MoE_enable:
            # gate input
            gate_in, _ = build_ffn_gate_input_common(x, token_pos_grid__cur, self.tasks)
            scores = self.moe_gate_mlp(gate_in).to(dtype=x.dtype)  # b,n,k
            if EXTRA_MoE_add_noise and self.training:
                scores = scores + torch.randn_like(scores) * EXTRA_MoE_noise_std
            scores = torch.softmax(scores, dim=-1)
            v_topk, idx_topk = scores.topk(k=EXTRA_MoE_topK, dim=-1)

            if EXTRA_MoE_routing_mode == 'dense':
                raise Exception('not carefully checked yet')
            else:  # sparse: forward only the selected experts and aggregate by top-k weights
                if 0:  weights_topk = torch.softmax(v_topk, dim=-1)  # b,n,topk
                else:  weights_topk = v_topk # b,n,topk. use top-k expert scores directly as weights
                b, n, d = x.shape
                dim_out = self.out_features
                y_moe_flat = x.new_zeros(b*n, dim_out) # flattened tensor accumulating outputs from all experts (bs*N, D_out)
                x_flat = x.reshape(b*n, d) # flatten input tensor (bs*N, D_in)
                unique_experts = torch.unique(idx_topk) # set of expert IDs actually selected in this batch
                for j in range(EXTRA_MoE_num_ep): # iterate only over active experts
                    mask_j = (idx_topk == j)  # b,n,topk boolean mask indicating which tokens picked expert j
                    sel_token_mask = mask_j.any(dim=-1)  # b,n boolean mask for tokens that selected expert j
                    flat_pos = sel_token_mask.view(-1).nonzero(as_tuple=False).squeeze(1)  # T_j flattened indices of tokens assigned to expert j
                    if flat_pos.numel() == 0:
                        continue
                    x_sel = x_flat.index_select(0, flat_pos)  # T_j,d select those tokens from flattened input
                    # run expert only on selected tokens (n = T_j)
                    y_sel = self.moe_experts_list[j](x_sel.view(1, x_sel.shape[0], d)).squeeze(0)  # T_j,dim_out expert j handles only its tokens
                    w_tok = (weights_topk * mask_j).sum(dim=-1).view(-1).index_select(0, flat_pos).unsqueeze(-1)  # T_j,1 weights for each token assigned to expert j
                    y_moe_flat.index_add_(0, flat_pos, w_tok * y_sel) # add weighted expert output back into flattened tensor (in-place)
                y = y + y_moe_flat.view(b, n, dim_out) # reshape aggregated MoE output and add back to backbone output
                if EXTRA_MoE_en_auxLoss and self.training:
                    raise Exception('not carefully checked yet')
                    importance = torch.zeros(self.num_moe_expert, device=scores.device, dtype=weights_topk.dtype)
                    importance = importance.scatter_add(0, idx_topk.reshape(-1), weights_topk.reshape(-1))
                    load = torch.zeros(self.num_moe_expert, device=scores.device, dtype=weights_topk.dtype)
                    load = load.scatter_add(0, idx_topk.reshape(-1), torch.ones_like(weights_topk.reshape(-1)))
                    k = importance.shape[0]
                    target_imp = torch.full_like(importance, fill_value=importance.sum() / k)
                    target_load = torch.full_like(load, fill_value=load.sum() / k)
                    aux_imp = F.mse_loss(importance, target_imp)
                    aux_load = F.mse_loss(load, target_load)
                    aux = 0.5 * (aux_imp + aux_load) * EXTRA_MoE_aux_coef
                    global_.moe_aux_loss = aux  # expose aux loss to the training loop for aggregation
        return y

    @torch.no_grad()
    def verify_approximation(self, num_tokens: int = 16, batch_size: int = 2, orig_ffn_list: list = None):
        if EXTRA_MoE_enable:  return
        device = next(self.shared_ffn.parameters()).device
        dtype = next(self.shared_ffn.parameters()).dtype
        x = torch.randn(batch_size, num_tokens, self.in_features, device=device, dtype=dtype)
        old_task = getattr(global_, 'task', None)
        for i,t in enumerate(self.tasks):
            _log2(orig_ffn_list[i], [self.task_lora_in[t], self.task_lora_out[t]])
            global_.task = t
            y_lora = self.forward(x)
            y_avg = self.shared_ffn(x)
            assert orig_ffn_list is not None, "orig_ffn_list must be provided for verification"
            y_orig = orig_ffn_list[i](x)
            d_avg = torch.norm((y_avg - y_orig).float()).item()
            d_lora = torch.norm((y_lora - y_orig).float()).item()
            _log1(f"[FFN verify] task={t} rank_in={self.task_lora_in[t].rank} rank_out={self.task_lora_out[t].rank} L2(avg,orig)={d_avg:.6f}  L2(lora,orig)={d_lora:.6f}")
        global_.task = old_task


class Linear_Shared_Plus_TaskLoRA(nn.Module):
    def __init__(self, l_proj: list, l_task: list, module_name: str = None):
        super().__init__()
        # _log1(f"-------- {module_name} --------")
        assert len(l_proj) >= 1
        self.dont_lora = False
        p0: nn.Linear = l_proj[0]
        # build shared from p0 and load avg
        self.shared: nn.Linear = copy.deepcopy(p0)
        if FOR_upcycle_ckpt_GEN_or_USE:
            avg_sd = _average_state_dict(l_proj)
            self.shared.load_state_dict(avg_sd)
        for p in self.shared.parameters():
            p.requires_grad = False
        self.in_features = self.shared.in_features
        self.out_features = self.shared.out_features
        self.tasks = l_task
        # cal/read adaptive rank across tasks
        if 1:
            if FOR_upcycle_ckpt_GEN_or_USE:
                w_diff_list = []
                for lin in l_proj:
                    w_diff_list.append(lin.weight.data - self.shared.weight.data)
                if FORCE_SAME_RANK_ACROSS_TASKS:
                    rank_lin = compute_adaptive_rank_for_linear_diffs(w_diff_list)
                    global_.moduleName_2_adaRank[module_name] = rank_lin
                else:
                    ranks_lin = compute_adaptive_rank_for_linear_diffs(w_diff_list, per_task=True)
                    global_.moduleName_2_adaRank[module_name] = ranks_lin
            else:
                r_info = global_.moduleName_2_adaRank[module_name]
                if FORCE_SAME_RANK_ACROSS_TASKS:  rank_lin  = r_info
                else:                               ranks_lin = r_info
        if 1:    # fallback decision for Linear
            min_dim = min(self.in_features, self.out_features)
            if min_dim < DONT_lora_if_dim_lt:
                # print(f"[LoRA fallback][Linear] {module_name} {min_dim=} < {DONT_lora_if_dim_lt}")
                self.dont_lora = True;  return
        _l = [] # per-task LoRA adapters
        for idx, lin in enumerate(l_proj):
            if not FORCE_SAME_RANK_ACROSS_TASKS:
                rank_lin = ranks_lin[idx]
            frac = float(rank_lin) / min(self.in_features, self.out_features)
            if frac > DONT_lora_if_rankFrac_gt:
                lora = ResidualAdapterLinearOnly(self.in_features, self.out_features, scaling=1.0, use_bias_delta=True)
            else:
                lora = LoRAAdapterLinearOnly(self.in_features, self.out_features, rank=rank_lin, dropout=0.0, scaling=1.0)
            if FOR_upcycle_ckpt_GEN_or_USE:
                with torch.no_grad():
                    w_diff = lin.weight.data - self.shared.weight.data
                    b_diff = (lin.bias.data - self.shared.bias.data) if (lin.bias is not None and self.shared.bias is not None) else None
                    lora.init_from_diff(w_diff, b_diff)
            _l.append(lora)
        self.task_lora = ModuleDict_W(_l, self.tasks)
        if FOR_upcycle_ckpt_GEN_or_USE:
            self.verify_approximation(orig_linear_list=l_proj)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.shared(x)
        y = y + self.task_lora[global_.task](x)
        return y
    @torch.no_grad()
    def verify_approximation(self, batch_size: int = 2, in_dim_override: int = None, orig_linear_list: list = None):
        device = next(self.shared.parameters()).device
        dtype = next(self.shared.parameters()).dtype
        d_in = self.in_features if in_dim_override is None else in_dim_override
        x = torch.randn(batch_size, d_in, device=device, dtype=dtype)
        old_task = getattr(global_, 'task', None)
        for i,t in enumerate(self.tasks):
            _log2(orig_linear_list[i], self.task_lora[t])
            global_.task = t
            y_lora = self.forward(x)
            y_avg = self.shared(x)
            assert orig_linear_list is not None, "orig_linear_list must be provided for verification"
            y_orig = orig_linear_list[i](x)
            d_avg = torch.norm((y_avg - y_orig).float()).item()
            d_lora = torch.norm((y_lora - y_orig).float()).item()
            _log1(f"[Linear verify] task={t} rank={self.task_lora[t].rank} L2(avg,orig)={d_avg:.6f}  L2(lora,orig)={d_lora:.6f}")
        global_.task = old_task

class Conv_Shared_Plus_TaskLoRA(nn.Module):
    def __init__(self, l_conv: list, l_task: list, module_name: str = None):
        super().__init__()
        # _log1(f"-------- {module_name} --------")
        assert len(l_conv) >= 1
        self.dont_lora = False
        c0: nn.Conv2d = l_conv[0]
        # build shared conv
        self.shared = nn.Conv2d(
            c0.in_channels, c0.out_channels,
            kernel_size=c0.kernel_size, stride=c0.stride,
            padding=c0.padding, dilation=c0.dilation,
            groups=c0.groups, bias=(c0.bias is not None),
            padding_mode=c0.padding_mode,
        )
        if FOR_upcycle_ckpt_GEN_or_USE:
            avg_sd = _average_state_dict(l_conv)
            self.shared.load_state_dict(avg_sd)
        for p in self.shared.parameters():
            p.requires_grad = False
        # per-task LoRA
        self.tasks = l_task
        _l = []
        # cal/read adaptive rank across tasks
        if 1:
            if FOR_upcycle_ckpt_GEN_or_USE:
                w_diff_list = []
                for c in l_conv:
                    w_diff_list.append(c.weight.data - self.shared.weight.data)
                if FORCE_SAME_RANK_ACROSS_TASKS:
                    rank_conv = compute_adaptive_rank_for_conv_diffs(w_diff_list)
                    global_.moduleName_2_adaRank[module_name] = rank_conv
                else:
                    ranks_conv = compute_adaptive_rank_for_conv_diffs(w_diff_list, per_task=True)
                    global_.moduleName_2_adaRank[module_name] = ranks_conv
            else:
                r_info = global_.moduleName_2_adaRank[module_name]
                if FORCE_SAME_RANK_ACROSS_TASKS:  rank_conv  = r_info
                else:                               ranks_conv = r_info
        if 1:    # fallback decision for Conv
            kH, kW = self.shared.kernel_size
            min_dim = min(self.shared.out_channels, self.shared.in_channels * kH * kW )
            if min_dim < DONT_lora_if_dim_lt:
                # print(f"[LoRA fallback][Conv] {module_name} {min_dim=} {DONT_lora_if_dim_lt=} (in={self.shared.in_channels}, out={self.shared.out_channels}, k=({kH},{kW}))")
                self.dont_lora = True;  return
        for idx, c in enumerate(l_conv):
            if not FORCE_SAME_RANK_ACROSS_TASKS:
                rank_conv = ranks_conv[idx]
            frac = float(rank_conv) / min(self.shared.out_channels, self.shared.in_channels * kH * kW)
            if frac > DONT_lora_if_rankFrac_gt:
                lora = ResidualAdapterConv2dOnly(
                    in_channels=c.in_channels, out_channels=c.out_channels,
                    kernel_size=c.kernel_size, stride=c.stride,
                    padding=c.padding, dilation=c.dilation, groups=c.groups,
                    scaling=1.0, use_bias_delta=True,
                )
            else:
                lora = LoRAAdapterConv2dOnly(
                    in_channels=c.in_channels, out_channels=c.out_channels,
                    kernel_size=c.kernel_size, stride=c.stride,
                    padding=c.padding, dilation=c.dilation, groups=c.groups,
                    rank=rank_conv, dropout=0.0, scaling=1.0,
                )
            if FOR_upcycle_ckpt_GEN_or_USE:
                with torch.no_grad():
                    w_diff = c.weight.data - self.shared.weight.data
                    b_diff = (c.bias.data - self.shared.bias.data) if c.bias is not None and self.shared.bias is not None else None
                    lora.init_from_diff(w_diff, b_diff)
            _l.append(lora)
        self.task_lora = ModuleDict_W(_l, self.tasks)

        if FOR_upcycle_ckpt_GEN_or_USE:
            self.verify_approximation(orig_conv_list=l_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.shared(x)
        y = y + self.task_lora[global_.task](x)
        return y

    @torch.no_grad()
    def verify_approximation(self, spatial_hw=(32, 32), batch_size: int = 2, orig_conv_list: list = None):
        device = next(self.shared.parameters()).device
        dtype = next(self.shared.parameters()).dtype
        H, W = spatial_hw
        x = torch.randn(batch_size, self.shared.in_channels, H, W, device=device, dtype=dtype)
        old_task = getattr(global_, 'task', None)
        for i,t in enumerate(self.tasks):
            _log2(orig_conv_list[i], self.task_lora[t])
            global_.task = t
            y_lora = self.forward(x)
            y_avg = self.shared(x)
            assert orig_conv_list is not None, "orig_conv_list must be provided for verification"
            y_orig = orig_conv_list[i](x)
            d_avg = torch.norm((y_avg - y_orig).float()).item()
            d_lora = torch.norm((y_lora - y_orig).float()).item()
            _log1(f"[Conv2d verify] task={t} rank={self.task_lora[t].rank} L2(avg,orig)={d_avg:.6f}  L2(lora,orig)={d_lora:.6f}")
        global_.task = old_task
