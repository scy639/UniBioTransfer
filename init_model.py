import sys,os
cur_dir = os.path.dirname(os.path.abspath(__file__))
if __name__=='__main__': sys.path.append(os.path.abspath(os.path.join(cur_dir, '..')))

from confs import *
import json
import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from MoE import *
from multiTask_model import *
from lora_layers import *
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
from my_py_lib.image_util import imgs_2_grid_A,img_paths_2_grid_A
import time
import copy
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import torchvision
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.models.diffusion.bank import Bank
from ldm.util import instantiate_from_config

from ldm.models.diffusion.ddim import DDIMSampler

from transformers import AutoFeatureExtractor

# import clip
from torchvision.transforms import Resize
from fnmatch import fnmatch


from PIL import Image
from torchvision.transforms import PILToTensor
#----------------------------------------------------------------------------


def get_moe():
    if 1:
        seed_everything(42)
        # torch.cuda.set_device(opt.device_ID)
        model :LatentDiffusion = instantiate_from_config(OmegaConf.load(f"LatentDiffusion.yaml").model,)
        if REFNET.ENABLE:
            assert model.model.diffusion_model_refNet.is_refNet

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        device = torch.device("cpu")
        model = model.to(device)
    if FOR_upcycle_ckpt_GEN_or_USE:
        del model.ptsM_Generator

    def average_module_weight(
        src_modules: list,
    ):
        """Average the weights of multiple modules"""
        if not src_modules:
            return None
        # Get the state dict of the first module as template
        avg_state_dict = {}
        first_state_dict = src_modules[0].state_dict()
        # Initialize with zeros
        for key in first_state_dict:
            avg_state_dict[key] = torch.zeros_like(first_state_dict[key])
        # Sum
        for module in src_modules:
            module_state_dict = module.state_dict()
            for key in avg_state_dict:
                avg_state_dict[key] += module_state_dict[key]
        # Average
        for key in avg_state_dict:
            avg_state_dict[key] /= len(src_modules)
        return avg_state_dict
    def recursive_average_module_weight(
        tgt_module: nn.Module,
        src_modules: list,
        cb,
    ):
        """
        Recursively find modules and replace with averaged weights based on callback
        """
        for name, child in tgt_module.named_children():
            if 1:    # Get corresponding modules from source models
                src_child_modules = []
                for src_module in src_modules:
                    src_child = getattr(src_module, name)
                    assert src_child is not None,name
                    src_child_modules.append(src_child)
            # assert not isinstance(child, TaskSpecific_MoE)
            if cb(child, name, tgt_module):
                print(f"[recursive_average_module_weight] {name=} child: {repr(child)[:50]} tgt_module: {repr(tgt_module)[:50]}")
                # Average & load
                avg_weights = average_module_weight(src_child_modules)
                child.load_state_dict(avg_weights)
            else:
                recursive_average_module_weight(child, src_child_modules, cb)
        return tgt_module
    
    def replace_module_with_TaskSpecific(
        tgt_module: nn.Module,# tgt module
        src_modules: list,
        cb,
        parent_name: str = "",
        depth :int = 0,
    ):
        for name, child in tgt_module.named_children():
            if 1:   # Get corresponding modules from source models
                src_child_modules = []
                for src_module in src_modules:
                    src_child = getattr(src_module, name)
                    assert src_child is not None,name
                    src_child_modules.append(src_child)
            assert not isinstance(child, TaskSpecific_MoE)
            full_name = f"{parent_name}.{name}"
            if cb(child, name, full_name, tgt_module):
                print(f"[replace_module_with_TaskSpecific] {name=} child: {repr(child)[:50]} tgt_module: {repr(tgt_module)[:50]}")
                setattr(tgt_module, name, TaskSpecific_MoE(src_child_modules,TASKS))
            else:
                if depth<=0:
                    replace_module_with_TaskSpecific(child, src_child_modules,cb,parent_name=full_name,depth=depth+1)
        return tgt_module
    
    if not FOR_upcycle_ckpt_GEN_or_USE:
        modelMOE :LatentDiffusion = model
        del model
        if 1:  # ensure distinct module instances per task (avoid shared identities)
            with open(PRETRAIN_JSON_PATH, 'r') as f: global_.moduleName_2_adaRank = json.load(f)
            print(f"loaded from {PRETRAIN_JSON_PATH=}")
            _src0 = copy.deepcopy(modelMOE.model.diffusion_model)
            _src1 = copy.deepcopy(modelMOE.model.diffusion_model)
            _src2 = copy.deepcopy(modelMOE.model.diffusion_model)
            _src3 = copy.deepcopy(modelMOE.model.diffusion_model)
            replace_modules_lossless(
                modelMOE.model.diffusion_model,
                [ _src0, _src1, _src2, _src3 ],
                [0,1,2,3],
                parent_name=".model.diffusion_model",
            )
            # Build-time dummy wrapping for task-specific heads so that ckpt keys match
            modelMOE.ID_proj_out = TaskSpecific_MoE([
                copy.deepcopy(modelMOE.ID_proj_out),
                copy.deepcopy(modelMOE.ID_proj_out),
                copy.deepcopy(modelMOE.ID_proj_out),
            ], [0,2,3])
            modelMOE.landmark_proj_out = TaskSpecific_MoE([
                copy.deepcopy(modelMOE.landmark_proj_out),
                copy.deepcopy(modelMOE.landmark_proj_out),
                copy.deepcopy(modelMOE.landmark_proj_out),
            ], [0,2,3])
            modelMOE.proj_out_source__head = TaskSpecific_MoE([
                copy.deepcopy(modelMOE.proj_out_source__head),
                copy.deepcopy(modelMOE.proj_out_source__head),
            ], [2,3])
            # Upcycle single refNet using three source refNets, and keep only one
            if REFNET.ENABLE:
                shared_ref = modelMOE.model.diffusion_model_refNet
                src0 = shared_ref
                src1 = copy.deepcopy(shared_ref)
                src2 = copy.deepcopy(shared_ref)
                src3 = copy.deepcopy(shared_ref)
                replace_modules_lossless(shared_ref, [src0, src1, src2, src3],[0,1,2,3], parent_name=".model.diffusion_model_refNet", for_refnet=True)
        # load from ./modelMOE.ckpt
        time.sleep(20*rank_)
        print(f"ckpt load over. m,u:")
    # Initialize bank here (after model structure is finalized)
    if REFNET.ENABLE :
        modelMOE.model.bank = Bank(reader=modelMOE.model.diffusion_model,writer=modelMOE.model.diffusion_model_refNet)
    if __name__=='__main__':
        for key in sorted( get_representative_moduleNames(modelMOE.state_dict().keys()) ):
            print(f"  - {key}")
    return modelMOE

