import numpy as np
import cv2
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt
from util_and_constant import *

def has_glasses(path_img):
    mask_path = path_img_2_path_mask(path_img)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    # if 10 in mask: # slow
    if (mask == 10).any():  # vectorized => clearly faster
        return True
    return False

def has_hat(path_img):
    mask_path = path_img_2_path_mask(path_img, label_mode="RF12_")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if (mask == 21).any():
        return True
    return False

def draw_pts70_batch(pts68, gaze, warp_mat256_np, dst_size, im_list=None, return_pt=False):
    import torch
    import torchvision.transforms as transforms
    
    left_eye1 = pts68[:, 36]
    left_eye2 = pts68[:, 39]
    right_eye1 = pts68[:, 42]
    right_eye2 = pts68[:, 45]

    right_eye_length = torch.sqrt(torch.sum((right_eye2-right_eye1)**2, dim=1, keepdim=True))
    left_eye_length = torch.sqrt(torch.sum((left_eye2-left_eye1)**2, dim=1, keepdim=True))
    right_eye_center = (right_eye2 + right_eye1) * 0.5
    left_eye_center = (left_eye2 + left_eye1) * 0.5

    with torch.no_grad():
      left_gaze = gaze[:,:2] * left_eye_length + left_eye_center
      right_gaze = gaze[:,2:] * right_eye_length + right_eye_center
      pts70 = torch.cat([pts68, left_gaze.view(-1,1,2), right_gaze.view(-1,1,2)],dim=1)
      landmarks = pts70.cpu().numpy().round().astype(int)
    
    colors = plt.get_cmap('rainbow')(np.linspace(0, 1, landmarks.shape[1]))
    colors = (255 * colors).astype(int)[:, 0:3].tolist()

    im_pts70_list = []
    if im_list is None:
        im_list = [np.zeros((256, 256, 3), dtype=np.uint8) for idx in range(landmarks.shape[0])]
    else:
        im_list = [np.array(x) for x in im_list]
    for idx in range(landmarks.shape[0]):
        image = im_list[idx]

        for i in range(landmarks.shape[1]):
            x, y = landmarks[idx, i, :]  
            color = colors[i]
            image = cv2.circle(image, (x, y), radius=2, color=(color[2],color[1],color[0]), thickness=-1)
        
        dst_image = cv2.warpAffine(image, warp_mat256_np[idx], (dst_size, dst_size), flags=(cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP), borderMode=cv2.BORDER_CONSTANT)
        im_pts70_list.append(Image.fromarray(dst_image))
        
    if return_pt:
        transform = transforms.Compose([ 
            transforms.ToTensor(), 
            transforms.Normalize(mean=0.5, std=0.5)
        ])
        tensor_list = [transform(x).view(1,3,dst_size,dst_size) for x in im_pts70_list]
        batch = torch.cat(tensor_list,dim=0)
        return batch
    else:
        return im_pts70_list
