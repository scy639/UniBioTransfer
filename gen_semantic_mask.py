"""
def:
    tgt: Target image to be edited (face swapped)
    ref: Face ID source image (also called src in REFace)
    swap: Swapped output image, using face ID from ref to replace face in tgt
"""
import os
from pathlib import Path
import dlib
from tqdm import tqdm
from my_py_lib.image_util import print_image_statistics
import torch
import torchvision
from PIL import Image
import numpy as np
from einops import rearrange
from torchvision.transforms import Resize
from torchvision.utils import make_grid
from contextlib import nullcontext
from torch.cuda.amp import autocast
from omegaconf import OmegaConf
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sampling configs
DDIM_STEPS = 50
GUIDANCE_SCALE = 3.0
IMG_SIZE = 512
LATENT_CHANNELS = 4
DOWNSAMPLE_FACTOR = 8
START_NOISE_T = 1000
DDIM_ETA = 0.0
PRECISION = "full"  # or "autocast"
FIXED_CODE = False  # whether to use fixed starting code
SAVE_INTERMEDIATES = False  # whether to save intermediate results
LOG_EVERY_T = 100  # log frequency during sampling


class MaskModel_LazyLoader:
    model = None
    @classmethod
    def get(cls):
        faceParsing_ckpt = "Other_dependencies/face_parsing/79999_iter.pth"
        if cls.model is None:
            from pretrained.face_parsing.face_parsing_demo import init_faceParsing_pretrained_model
            cls.model = init_faceParsing_pretrained_model(
                'default',
                faceParsing_ckpt,
                ''
            )
            print(f"Initialized face parsing model from {faceParsing_ckpt}")
        return cls.model


def gen_semantic_mask(path_img: Path, path_mask_to_save: Path, label_mode:str, path_vis: Path = None):
    """Generate semantic mask for an image using face parsing model"""
    pil_im = Image.open(path_img).convert("RGB")
    w, h = pil_im.size
    # print(f"{pil_im.size=}") # 512,512
    TMP_size = 1024
    if w != TMP_size or h != TMP_size:
        pil_im = pil_im.resize((TMP_size, TMP_size), Image.BILINEAR)
    
    model = MaskModel_LazyLoader.get()
    from pretrained.face_parsing.face_parsing_demo import faceParsing_demo, vis_parsing_maps
    
    # print(f"{pil_im.size=}") # 1024,1024
    # Generate mask with conversion to seg12 format
    mask = faceParsing_demo(
        model, 
        pil_im, 
        label_mode,
        model_name='default'
    )
    
    try:
        Image.fromarray(mask).save(path_mask_to_save)
    except Exception as e:
        print(f"{e=}")
        print(f"{path_mask_to_save=}")
        if path_mask_to_save.exists():
            path_mask_to_save.unlink()
            print(f'path_mask_to_save.unlink()')
    # print(f"Saved mask: {path_mask_to_save}")
    # print(f"{mask.shape=}") # 512,512
    
    if path_vis:
        mask_vis = vis_parsing_maps(pil_im, mask)
        Image.fromarray(mask_vis).save(path_vis)
        print(f"Saved mask vis: {path_vis}")