from util_and_constant import *
import cv2
import numpy as np
def auto_interpolation(img:np.ndarray, dst_size:tuple):
    if img.shape[0] > dst_size[0] and img.shape[1] > dst_size[1]:
        interpolation = cv2.INTER_AREA # value:3
    else:
        interpolation = cv2.INTER_LANCZOS4 # value:4
    return interpolation

_DEBUG_interpolation = 0    # if 1, save before resize to 4debug/cv2_resize_auto_interpolation/{str_t_pid()}.png
def cv2_resize_auto_interpolation(src:np.ndarray, dsize:tuple, interpolation:int=None, **kwargs):
    if interpolation is None:
        interpolation = auto_interpolation(src, dsize)
    ret= cv2.resize(src, dsize, interpolation=interpolation, **kwargs)
    if _DEBUG_interpolation and src.shape[0]>1130:
        _p = f"4debug/cv2_resize_auto_interpolation/{str_t_pid()}-before.png"
        cv2.imwrite(_p, src);  print(f"{_p=}")
        _p = f"4debug/cv2_resize_auto_interpolation/{str_t_pid()}-after-{interpolation}.png"
        cv2.imwrite(_p, ret);  print(f"{_p=}")
    return ret