from imports import *
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as T
from einops import rearrange
import albumentations

from util_face import *
from util_4dataset import *
from util_cv2 import cv2_resize_auto_interpolation
from Mediapipe_Result_Cache import Mediapipe_Result_Cache


def resize_A(img, dataset_name, size=(512, 512), interpolation=None):
    is_pil = isinstance(img, Image.Image)
    if is_pil:
        img = np.array(img)
    if img.shape[:2] != (512, 512):
        img = cv2_resize_auto_interpolation(img, size, interpolation=interpolation)
    if is_pil:
        img = Image.fromarray(img)
    return img


def un_norm_clip(x1):
    x = x1 * 1.0
    reduce = False
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
        reduce = True
    x[:, 0, :, :] = x[:, 0, :, :] * 0.26862954 + 0.48145466
    x[:, 1, :, :] = x[:, 1, :, :] * 0.26130258 + 0.4578275
    x[:, 2, :, :] = x[:, 2, :, :] * 0.27577711 + 0.40821073
    if reduce:
        x = x.squeeze(0)
    return x


def un_norm(x):
    return (x + 1.0) / 2.0


def _dilate(_mask, kernel_size, iterations):
    _mask = _mask.astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    _mask = cv2.dilate(_mask, kernel, iterations=iterations)
    _mask = _mask.astype(bool)
    return _mask


def dilate_4_task0(sm_mask):
    sm_mask = np.array(sm_mask)
    preserve1 = [2, 3, 10, 5]
    mask1 = np.isin(sm_mask, preserve1)
    mask1 = _dilate(mask1, 7, 1)
    preserve2 = [3, 10]
    mask2 = np.isin(sm_mask, preserve2)
    mask2 = _dilate(mask2, 10, 3)
    preserve3 = [1]
    mask3 = np.isin(sm_mask, preserve3)
    mask3 = _dilate(mask3, 7, 2)
    mask = mask1 | mask2 | mask3
    return mask


class Dataset_custom(data.Dataset):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    def get_img4clip(
        self,
        img,
        sm_mask,
        preserve,
        for_clip=True,
        add_semantic_head=False,
        mask_after_npisin=None,
        for_inpaint512=False,
    ):
        sm_mask = np.array(sm_mask)
        if mask_after_npisin is None:
            if self.task == 0 and 0:
                mask = dilate_4_task0(sm_mask)
            else:
                mask = np.isin(sm_mask, preserve)
                if self.task == 0 and 1 and for_inpaint512:
                    forehead_mask = get_forehead_mask(sm_mask)
                    mask = mask & ~forehead_mask
        else:
            mask = mask_after_npisin

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if add_semantic_head:
            mask_before_colorSM = mask
            img, mask = add_colorSM(img, sm_mask, preserve, None)
        mask = mask_after_npisin__2__tensor(mask)

        if for_clip:
            image_tensor = get_tensor_clip()(img)
        else:
            image_tensor = get_tensor(mean=self.mean, std=self.std)(img)
        image_tensor = T.Resize([512, 512])(image_tensor)
        image_tensor = image_tensor * mask
        if for_clip:
            image_tensor = 255.0 * rearrange(un_norm_clip(image_tensor), "c h w -> h w c").cpu().numpy()
            _size = 224
        else:
            image_tensor = 255.0 * rearrange(un_norm(image_tensor), "c h w -> h w c").cpu().numpy()
            _size = 512
        
        image_tensor = albumentations.Resize(height=_size, width=_size)(image=image_tensor)
        image_tensor = Image.fromarray(image_tensor["image"].astype(np.uint8))
        if for_clip:
            image_tensor = get_tensor_clip()(image_tensor)
        else:
            image_tensor = get_tensor(mean=self.mean, std=self.std)(image_tensor)
            image_tensor = image_tensor * mask
        if add_semantic_head:
            mask = mask_after_npisin__2__tensor(mask_before_colorSM)
        return image_tensor, mask

    def __init__(
        self,
        state,
        task,
        paths_tgt,
        paths_ref,
        name="custom",
    ):
        if task == 0:
            USE_filter_mediapipe_fail_swap = 1
            USE_pts = 1
            READ_mediapipe_result_from_cache = 1
        elif task == 1:
            USE_filter_mediapipe_fail_swap = 0
            USE_pts = 0
            READ_mediapipe_result_from_cache = 1
        elif task == 2:
            USE_filter_mediapipe_fail_swap = 1
            USE_pts = 1
            READ_mediapipe_result_from_cache = 1
        elif task == 3:
            USE_filter_mediapipe_fail_swap = 0
            USE_pts = 1
            READ_mediapipe_result_from_cache = 1
        self.READ_mediapipe_result_from_cache = READ_mediapipe_result_from_cache

        assert state == "test"
        self.state = state
        self.image_size = 512
        self.kernel = np.ones((1, 1), np.uint8)
        self.name = name

        assert paths_tgt is not None and paths_ref is not None, "paths_tgt and paths_ref are required"
        assert len(paths_tgt) == len(paths_ref), "paths_tgt and paths_ref must be the same length"
        self.paths_tgt = list(paths_tgt)
        self.paths_ref = list(paths_ref)

        if READ_mediapipe_result_from_cache:
            self.mediapipe_Result_Cache = Mediapipe_Result_Cache()
        self.task = task

    def __getitem__(self, index):
        task = self.task
        path_tgt = self.paths_tgt[index]
        path_ref = self.paths_ref[index]


        img_tgt = Image.open(path_tgt).convert("RGB")
        img_tgt = resize_A(img_tgt, self.name)

        mask_path = path_img_2_path_mask(path_tgt)
        if self.task == 0:
            preserve = [1, 2, 3, 10, 5, 6, 7, 9]
            if 0:
                preserve = [1, 2, 3, 10, 5]
            sm_mask_tgt = Image.open(mask_path).convert("L")
            sm_mask_tgt = np.array(sm_mask_tgt)
            if 0:
                mask_tgt = dilate_4_task0(sm_mask_tgt)
            else:
                mask_tgt = np.isin(sm_mask_tgt, preserve)
                if self.task == 0 and 1:
                    forehead_mask = get_forehead_mask(sm_mask_tgt)
                    mask_tgt = mask_tgt & ~forehead_mask
        elif self.task == 1:
            preserve = [4]
            mask_tgt = path_img_2_mask(path_tgt, preserve)
        elif self.task == 3:
            preserve = [1, 2, 3, 10, 4, 5, 6, 7, 9]
            mask_tgt = path_img_2_mask(path_tgt, preserve)
        elif self.task == 2:
            preserve = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 21]
            sm_mask_tgt = Image.open(mask_path).convert("L")
            sm_mask_tgt = np.array(sm_mask_tgt)
            mask_tgt = np.isin(sm_mask_tgt, preserve)

        converted_mask = np.zeros_like(mask_tgt)
        converted_mask[mask_tgt] = 255
        mask_tgt = Image.fromarray(converted_mask).convert("L")
        mask_tensor = 1 - get_tensor(normalize=False, toTensor=True)(mask_tgt)

        image_tensor = get_tensor(mean=self.mean, std=self.std)(img_tgt)
        image_tensor_resize = T.Resize([self.image_size, self.image_size])(image_tensor)
        mask_tensor_resize = T.Resize([self.image_size, self.image_size])(mask_tensor)

        if task == 2:
            inpaint_tensor_resize = image_tensor_resize
        else:
            inpaint_tensor_resize = image_tensor_resize * mask_tensor_resize
        if 1:
            mask_tensor_resize = 1 - mask_tensor_resize

        if 1:
            mask_path_ref = path_img_2_path_mask(path_ref)
            sm_mask_ref = Image.open(mask_path_ref).convert("L")
            sm_mask_ref = np.array(sm_mask_ref)
            img_ref = cv2.imread(str(path_ref))
            img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
            img_ref = resize_A(img_ref, self.name)

        if task != 2:
            ref_image_tensor, ref_mask_tensor = self.get_img4clip(
                img_ref, sm_mask_ref, preserve, for_clip=True, add_semantic_head=0
            )
            if task == 3:
                ref_image_faceOnly_tensor, _ = self.get_img4clip(
                    img_ref,
                    sm_mask_ref,
                    [1, 2, 3, 10, 5, 6, 7, 9],
                    for_clip=False,
                    add_semantic_head=0,
                )
        else:
            ref_image_tensor = inpaint_tensor_resize

        ret = {
            "inpaint_image": inpaint_tensor_resize,
            "inpaint_mask": mask_tensor_resize,
            "ref_imgs": ref_image_tensor,
            "task": self.task,
        }

        if self.task == 0:
            ret["enInputs"] = {
                "face_ID-in": ref_image_tensor,
                "face-clip-in": ref_image_tensor,
            }
        elif self.task == 1:
            ret["enInputs"] = {
                "hair-clip-in": ref_image_tensor,
            }
        elif self.task == 2:
            tgt_nonBg_tensor, _ = self.get_img4clip(img_tgt, sm_mask_tgt, preserve)
            ret["enInputs"] = {
                "face_ID-in": tgt_nonBg_tensor,
                "head-clip-in": tgt_nonBg_tensor,
            }
        elif self.task == 3:
            ret["enInputs"] = {
                "face_ID-in": ref_image_faceOnly_tensor,
                "head-clip-in": ref_image_tensor,
            }

        if (REFNET.ENABLE and REFNET.task2layerNum[task] > 0) or CH14:
            if task != 2:
                ref_imgs_4unet, ref_mask_4unet = self.get_img4clip(
                    img_ref, sm_mask_ref, preserve, for_clip=False, add_semantic_head=0
                )
            else:
                ref_imgs_4unet, ref_mask_4unet = self.get_img4clip(
                    img_tgt,
                    sm_mask_tgt,
                    "any",
                    for_clip=False,
                    add_semantic_head=0,
                    mask_after_npisin=np.ones_like(sm_mask_tgt).astype(bool),
                )
            ref_imgs_4unet = T.Resize([self.image_size, self.image_size])(ref_imgs_4unet)
            ref_mask_512 = T.Resize([self.image_size, self.image_size])(ref_mask_4unet)
            ret["ref_imgs_4unet"] = ref_imgs_4unet
            ret["ref_mask_512"] = ref_mask_512

        if self.READ_mediapipe_result_from_cache:
            if self.state == "test":
                if task == 2:
                    _p_lmk = path_ref
                else:
                    _p_lmk = path_tgt
            else:
                _p_lmk = path_tgt
            ret["mediapipe_lmkAll"] = self.mediapipe_Result_Cache.get(_p_lmk)
            if ret["mediapipe_lmkAll"] is None:
                raise RuntimeError(
                    f"Missing Mediapipe cache for input image: {_p_lmk}. "
                    "Precompute landmarks and ensure cache exists before inference."
                )

        if self.state == "test":
            prior_image_tensor = "None"
            out_stem = f"{Path(path_tgt).stem}-{Path(path_ref).stem}"
            if task == 2:
                ref512, _ = self.get_img4clip(
                    img_ref, sm_mask_ref, preserve, for_clip=False, add_semantic_head=0
                )
                ref512 = T.Resize([self.image_size, self.image_size])(ref512)
                ret["ref512"] = ref512
            ret = (image_tensor_resize, prior_image_tensor, ret, out_stem)
        return ret

    def __len__(self):
        return len(self.paths_tgt)
