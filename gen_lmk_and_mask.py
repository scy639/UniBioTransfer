ENABLE_lmk_cache = False
ENABLE_mask_cache = False


import cv2
from imports import *
from util_cv2 import cv2_resize_auto_interpolation
from Mediapipe_Result_Cache import Mediapipe_Result_Cache
from lmk_util.lmk_extractor import LandmarkExtractor


def gen_lmk_and_mask(img_paths, size=512, write_cache=True):
    extractor = LandmarkExtractor()
    cache = Mediapipe_Result_Cache()
    seen = set()
    for p in img_paths:
        if not p:
            continue
        p = str(p)
        if p in seen:
            continue
        seen.add(p)

        cache_path = cache.get_path(p)
        if not  ( cache_path.exists() and ENABLE_lmk_cache ):
            img = cv2.imread(p)
            if img is None:
                print(f"cv2.imread failed: {p}")
                raise
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2_resize_auto_interpolation(img, (size, size))
            lmks = extractor.extract_single(img)
            if lmks is None:
                print(f"no lmks: {p}")
                raise
                continue
            if write_cache:
                cache.set(p, lmks)

        path_img_2_path_mask(p, reuse_if_exists=ENABLE_mask_cache, label_mode="RF12_")
