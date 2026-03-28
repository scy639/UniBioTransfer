from util_and_constant import *
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

def path_img_2_mask(
    path_img,
    preserve=(1, 2, 3, 5, 6, 7, 9, 10, 11, ), # int | list-liek. Default val represents face
):
    """
    0 bg, 1 mouth, 2 eyebrow, 3 eyes, 4 hair, 5 nose, 6 face (excluding facial parts), 7: ear, 8: neck, 9: tooth
    10: eye_glass, 11: ear_rings
    """
    if isinstance(preserve,int):
        preserve = (preserve,)
    if 1:
        assert isinstance(preserve,tuple) or isinstance(preserve,list)
        assert all(isinstance(p, int) and 0 <= p <= 11 for p in preserve)
    import numpy as np
    from PIL import Image
    mask_path = path_img_2_path_mask(path_img)
    mask = Image.open(mask_path).convert('L')
    mask = np.array(mask) 
    mask = np.isin(mask, preserve)
    return mask



def get_forehead_mask(sm_mask):
    # return mask (np bool) where the forehead (face above eyebrows) is True
    sm_mask = np.array(sm_mask)
    # 6 is face (excluding facial parts); keep only the forehead part
    # First get all face pixels
    face_mask = (sm_mask == 6)
    # Get eyebrow pixels to determine forehead boundary
    # if 2 in sm, ; elif 3(eyes) in ; elif 10(eye_glass) in ; else
    if 2 in sm_mask:
        eyebrow_mask = (sm_mask == 2)
        eyebrow_coords = np.where(eyebrow_mask)
        eyebrow_top = np.min(eyebrow_coords[0])
        # Forehead is face region above eyebrows
        forehead_mask = face_mask & (np.arange(sm_mask.shape[0])[:, None] < eyebrow_top)
    elif 3 in sm_mask:
        eye_mask = (sm_mask == 3)
        eye_coords = np.where(eye_mask)
        eye_top = np.min(eye_coords[0])
        # Estimate forehead as region above eyes with some margin
        forehead_threshold = eye_top - 20  # 20 pixels above eyes as forehead
        forehead_mask = face_mask & (np.arange(sm_mask.shape[0])[:, None] < forehead_threshold)
    elif 10 in sm_mask:
        glass_mask = (sm_mask == 10)
        glass_coords = np.where(glass_mask)
        glass_top = np.min(glass_coords[0])
        # Forehead is face region above glasses
        forehead_mask = face_mask & (np.arange(sm_mask.shape[0])[:, None] < glass_top)
    else:
        # If no eyebrows detected, keep upper portion of face
        face_coords = np.where(face_mask)
        if len(face_coords[0]) > 0:
            face_top = np.min(face_coords[0])
            face_height = np.max(face_coords[0]) - face_top
            forehead_threshold = face_top + face_height * 0.15  # top 15% as forehead
            forehead_mask = face_mask & (np.arange(sm_mask.shape[0])[:, None] < forehead_threshold)
        else:
            forehead_mask = np.zeros_like(face_mask, dtype=bool)
    forehead_mask = forehead_mask & face_mask
    return forehead_mask
