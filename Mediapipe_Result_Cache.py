from confs import *
import json,random,os
import numpy as np



class Mediapipe_Result_Cache:
    """
    Convention: when a cache entry exists, it must not be None.
    In other words, None results should not be cached; get/set guard against historical None values.
    """
    # DIR = Path('/inspurfs/group/mayuexin/suncy/mediapipe_result/A')
    DIR = Path('data/mediapipe_result')
    def __init__(self):
        pass
    def get_path(self, img_path):
        img_path = Path(img_path)
        str_img_folder = str(img_path.parent)
        assert '|' not in str_img_folder
        str_img_folder = str_img_folder.replace('/', '|')
        lmk_folder = self.DIR / str_img_folder
        lmk_folder.mkdir(parents=1, exist_ok=True)
        ret= lmk_folder / (img_path.name+'.npy')
        return ret
    def get(self, img_path):
        path = self.get_path(img_path)
        # print(f"[get] {path=}")
        if path.exists():
            ret = np.load(path)
            assert ret is not None
            return ret
    def set(self, img_path, lmks):
        assert lmks is not None
        path = self.get_path(img_path)
        np.save(path, lmks)
        # print(f"{path=}")
