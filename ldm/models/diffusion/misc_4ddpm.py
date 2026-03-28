"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""
import global_
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only
from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler
from torchvision.transforms import Resize
import torchvision.transforms.functional as TF  
import torch.nn.functional as F
import math
import time
import random
import copy
from torch.autograd import Variable
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from src.Face_models.encoders.model_irse import Backbone
import dlib
from eval_tool.lpips.lpips import LPIPS
import wandb
from PIL import Image
import argparse
from contextlib import nullcontext
from util_face import *
from util_vis import vis_tensors_A
from my_py_lib.image_util import save_any_A,imgs_2_grid_A
from my_py_lib.torch_util import recursive_to
from my_py_lib.torch_util import custom_repr_v3
from confs import *
from lmk_util.lmk_extractor import LandmarkExtractor,lmkAll_2_lmkMain
from ldm.modules.encoders.modules import FrozenCLIPEmbedder
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from MoE import *


__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}




def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def un_norm_clip(x1):
    x = x1*1.0 # to avoid changing the original tensor or clone() can be used
    reduce=False
    if len(x.shape)==3:
        x = x.unsqueeze(0)
        reduce=True
    x[:,0,:,:] = x[:,0,:,:] * 0.26862954 + 0.48145466
    x[:,1,:,:] = x[:,1,:,:] * 0.26130258 + 0.4578275
    x[:,2,:,:] = x[:,2,:,:] * 0.27577711 + 0.40821073
    
    if reduce:
        x = x.squeeze(0)
    return x

def un_norm(x):
    return (x+1.0)/2.0
 
def save_clip_img(img, path,clip=True):
    if clip:
        img=un_norm_clip(img)
    else:
        img=torch.clamp(un_norm(img), min=0.0, max=1.0)
    img = img.cpu().numpy().transpose((1, 2, 0))
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(path)
    # if clip:
    #     img=TF.normalize(img, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    # else:  
    #     img=TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])



class IDLoss(nn.Module):
    def __init__(self,opts,multiscale=False):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.opts = opts 
        self.multiscale = multiscale
        self.face_pool_1 = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        # self.facenet=iresnet100(pretrained=False, fp16=False) # changed by sanoojan
        
        self.facenet.load_state_dict(torch.load(opts.other_params.arcface_path))
        
        self.face_pool_2 = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        
        self.set_requires_grad(False)
            
    def set_requires_grad(self, flag=True):
        for p in self.parameters():
            p.requires_grad = flag
    
    def extract_feats(self, x,clip_img=True):
        # breakpoint()
        if clip_img:
            x = un_norm_clip(x)
            x = TF.normalize(x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        x = self.face_pool_1(x)  if x.shape[2]!=256 else  x # (1) resize to 256 if needed
        x = x[:, :, 35:223, 32:220]  # (2) Crop interesting region
        x = self.face_pool_2(x) # (3) resize to 112 to fit pre-trained model
        # breakpoint()
        x_feats = self.facenet(x, multi_scale=self.multiscale )
        
        # x_feats = self.facenet(x) # changed by sanoojan
        return x_feats

    

    def forward(self, y_hat, y,clip_img=True,return_seperate=False):
        n_samples = y.shape[0]
        y_feats_ms = self.extract_feats(y,clip_img=clip_img)  # Otherwise use the feature from there

        y_hat_feats_ms = self.extract_feats(y_hat,clip_img=clip_img)
        y_feats_ms = [y_f.detach() for y_f in y_feats_ms]
        
        loss_all = 0
        sim_improvement_all = 0
        seperate_sim=[]
        for y_hat_feats, y_feats in zip(y_hat_feats_ms, y_feats_ms):
 
            loss = 0
            sim_improvement = 0
            count = 0
            # lossess = []
            for i in range(n_samples):
                sim_target = y_hat_feats[i].dot(y_feats[i])
                sim_views = y_feats[i].dot(y_feats[i])

                seperate_sim.append(sim_target)
                loss += 1 - sim_target  # id loss
                sim_improvement +=  float(sim_target) - float(sim_views)
                count += 1
                
            
            loss_all += loss / count
            sim_improvement_all += sim_improvement / count
        if return_seperate:
            return loss_all, sim_improvement_all, seperate_sim
        return loss_all, sim_improvement_all, None

def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2

class LandmarkDetectionModel(nn.Module):
    def __init__(self):
        super(LandmarkDetectionModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(640, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.landmark_predictor = nn.Linear(128 * 32 * 32, 68 * 2)  # Adjust output size as needed

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        landmarks = self.landmark_predictor(x)
        return landmarks
