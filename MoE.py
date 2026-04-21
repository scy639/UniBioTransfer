from imports import *
import global_
import torch,copy
import torch.nn as nn
from ldm.modules.attention import FeedForward,CrossAttention
from ldm.modules.diffusionmodules.openaimodel import UNetModel,ResBlock,TimestepEmbedSequential
# import torch.nn.functional as F

# ---------------- Configs ----------------
CONV2D_PARAM_STATS = []

def average_module_weight(src_modules: list):
    """Average the weights of multiple modules (similar to init_model.py)."""
    if not src_modules:
        return None
    avg_state_dict = {}
    first_state_dict = src_modules[0].state_dict()
    for key in first_state_dict:
        avg_state_dict[key] = torch.zeros_like(first_state_dict[key])
    for module in src_modules:
        module_state_dict = module.state_dict()
        for key in avg_state_dict:
            avg_state_dict[key] += module_state_dict[key]
    for key in avg_state_dict:
        avg_state_dict[key] /= len(src_modules)
    return avg_state_dict

class ModuleDict_W(nn.Module): # Wrapper of ModuleDict
    def __init__(self, modules: list, keys: list):
        super().__init__()
        assert len(keys) == len(modules), f"{len(keys)=} {len(modules)=}"
        self._keys = [int(k) for k in keys]
        self._moduleDict = nn.ModuleDict({str(int(k)): m for k, m in zip(self._keys, modules)})
    def __getitem__(self, k: int):
        _k = str(int(k))
        return self._moduleDict[_k]
    def keys(self):
        return list(self._keys)
    def forward(self, *args, **kwargs):
        cur_task = global_.task
        assert cur_task in self._keys, f"Current task {cur_task} not in available tasks {self._keys}"
        return self._moduleDict[str(int(cur_task))](*args, **kwargs)
    def offload_unused_tasks(self, unused_tasks, method: str):
        for i in unused_tasks:
            _k = str(int(i))
            if _k in self._moduleDict:
                if method == 'del':
                    # self._moduleDict[_k] = None # should behave the same either way
                    del self._moduleDict[_k]
                elif method == 'cpu':
                    self._moduleDict[_k].to('cpu')
                else:
                    raise

class TaskSpecific_MoE(nn.Module):
    def __init__(
        self,
        module:nn.Module,# or list of Module
        tasks:tuple,
    ):
        super().__init__()
        self.cur_task = None
        self.tasks = tasks
        if isinstance(module, nn.Module):
            modules = [copy.deepcopy(module) for _ in self.tasks]
        elif isinstance(module, list):
            assert len(module) == len(self.tasks), f"got {len(module)} and {len(self.tasks)}"
            modules = module
        else:
            raise ValueError(f"got {type(module)}")
        self.tasks_2_module = ModuleDict_W(modules, self.tasks)
        
    def forward(self, *args, **kwargs) -> torch.Tensor:
        # cur_task = self.cur_task
        cur_task = global_.task
        assert cur_task in self.tasks, f"Current task {cur_task} not in available tasks {self.tasks}"
        return self.tasks_2_module[cur_task](*args, **kwargs)

    def set_task(self, task):
        assert 0, 'set_task is disabled for now; update to gg.task instead'
        # assert task in self.tasks, f"Task {task} not in available tasks {self.tasks}"
        self.cur_task = task

def is_task_specific_(name:str):
    is_task_specific = (
        ('._moduleDict.' in name) or
        ('tasks_2_module' in name) or
        ('task_ffn' in name) or
        ('task_proj' in name) or
        ('task_conv' in name) or
        ('task_gate_mlps' in name) or
        ('task_lora' in name) or
        
        ('encoder_clip_' in name) or
        ('proj_out_source__' in name) or
        ('ID_proj_out' in name) or
        ('landmark_proj_out' in name) or
        ('learnable_vector' in name)
    )
    return is_task_specific
def tp_param_need_sync(name: str, p: torch.nn.Parameter):
    if is_task_specific_(name):
        return False, True
    if 'first_stage_model' in name or 'face_ID_model' in name or 'encoder_clip_face.tokenizer' in name or 'encoder_clip_face.model' in name:
        return False, False
    if not p.requires_grad:
        return False, False
    return True, False
def offload_unused_tasks(parent: nn.Module, active_task: int, method: str, ):
    unused_tasks = [_t for _t in TASKS if _t != active_task] # inactive tasks
    for name, child in parent.named_children():
        if hasattr(child, '__class__') and child.__class__.__name__ in [
            'TaskSpecific_MoE',
            'FFN_TaskSpecific_Plus_Shared',
            'Linear_TaskSpecific_Plus_Shared',
            'Conv_TaskSpecific_Plus_Shared',
            'FFN_Shared_Plus_TaskLoRA',
            'Linear_Shared_Plus_TaskLoRA',
            'Conv_Shared_Plus_TaskLoRA',
        ]:
            for attr_name in [ # normalize attribute handling to avoid repetition
                'tasks_2_module',
                'task_ffn', 'task_proj', 'task_conv',
                'task_lora_in', 'task_lora_out', 'task_lora',
            ]:
                if hasattr(child, attr_name):
                    ml = getattr(child, attr_name)
                    if isinstance(ml, nn.ModuleList):
                        for i in unused_tasks:  # move or delete parameters for inactive tasks
                            if method == 'del':
                                ml[i] = None
                            elif method == 'cpu':
                                ml[i].to('cpu')
                            else:  raise Exception
                    elif isinstance(ml, ModuleDict_W):
                        ml.offload_unused_tasks(unused_tasks,method)
            # recurse(child)
        else:  offload_unused_tasks(child,active_task,method)
def offload_unused_tasks__LD(modelMOE, task_keep: int, method: str, ):
    # Remove or offload inactive task-related parameters to save CUDA memory (method: del|cpu)
    offload_unused_tasks(modelMOE, task_keep, method)
