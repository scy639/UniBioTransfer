from pathlib import Path; import sys, os; from fnmatch import fnmatch
import global_

TASKS = (0,1,2,3,)
TP_enable :bool = 1
world_size_ = int(os.environ.get("WORLD_SIZE", "1"))
rank_ = int(os.environ.get("RANK", "0"))
local_rank_ = int(os.environ.get("LOCAL_RANK", rank_ ))
assert world_size_ >= 1 and 0 <= rank_ < world_size_

USE_filter_mediapipe_fail_swap = 1
CH14 :bool = False
class REFNET:
    ENABLE :bool = 1
    CH9 :bool = 0
    task2layerNum = { # actually used as bool now
        0:9,
        1:9,
        2:9,
        3:9,
    }
USE_pts :bool = 1
READ_mediapipe_result_from_cache = 1
ADAM_or_SGD :bool = False  # 1 => AdamW ; 0 => sgd
N_EPOCHS_TRAIN_REF_AND_MID :int = 1
# ZeRO-1 optimizer sharding (ZeroRedundancyOptimizer).  avoid using FSDP, just ZeroRedundancyOptimizer
ZERO1_ENABLE :bool = 0


NUM_token = 257



if 1:
    SD14_filename = "sd-v1-4.ckpt"
    SD14_localpath = Path("checkpoints") / SD14_filename
    PRETRAIN_CKPT_PATH = f"checkpoints/pretrained.ckpt"
    PRETRAIN_JSON_PATH = f"checkpoints/pretrained.json"

#-------------------------------------------
assert isinstance(TASKS,tuple)
NUM_pts = 95
global_.TP_enable = TP_enable
global_.rank_ = rank_



MERGE_CFG_in_one_batch :bool = 1

FOR_upcycle_ckpt_GEN_or_USE :bool = 0

DEBUG = 0
DEBUG_skip_load_ckpt :bool = 0
DBEUG_skip_most_in_Unet_constructor :bool = 0
# import os; os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
LOG_debug_level = 0


_gate_total_runs = {}
_gate_total_calls = {}
_gate_k2tu = { # id 2 (max_run,interval,prob)
    'vis Dataset_vFrame perspectiveWarp'                   : ( 0, 1, None ),
    'vis LatentDiffusion.get_input'                        : ( 0, 5, None ),
    'vis LatentDiffusion.get_input-before_return True'     : ( 0, 5, None ),
    'vis LatentDiffusion.get_input-before_return False'    : ( 0, 1, None ),
    'vis LatentDiffusion.conditioning_with_feat'           : ( 0, 2, None ),
    'vis LatentDiffusion.p_losses--after-apply_model'      : ( 0, 2, None ),
    'statistics test_batch[0]'      : ( 0, 2, None ),#-------------infer-----------
    "Project config:"               : ( 0, 1, None ),# ------------for printC (arg[0] as id)-----------
    "Lightning config:"             : ( 0, 1, None ),
    "logger_cfg="                   : ( 0, 1, None ),
    "Merged modelckpt-cfg:"         : ( 0, 1, None ),
    'bank get'                      : ( 0, 1, None ),
    'bank set'                      : ( 0, 1, None ),
    'clear'                         : ( 0, 1, None ),
    'mean ct:'                      : ( 0, 1, None ),
    "[__iter__]"                    : ( 1, 1, None ),
    "[_create_batches]"             : ( 1, 1, None ),
    '[set_task_for_MoE]'            : ( 1, 1, None ),
    'len_inter'                     : ( 3, 5, None ),
    'non_paired'                    : ( 3, 5, None ),
    'ddim rec bg'                   : ( 4, 5, None ),
    '[training step]'               : ( 7, 1, None ),
    'LatentDiffusion.configure_optimizers params:' : ( 0, 1, None ),
    'c.shape'                       : ( 2, 6, None ),
    '[conditioning_with_feat return]': ( 0, 6, None ),
    'c for refNet'                   : ( 0, 6, None ),
    'hair _c.shape:'                : ( 0, 1, None ),
    'head _c.shape:'                : ( 0, 1, None ),
    'task'                          : ( 9, 1, None ),
    '_t_norm'                       : ( 9, 1, None ),
    'orig,ID clip,lpips rec lmk:'   : (20, 2, None ),#-------------ddim_losses-----------
    'loss_lpips_1 at 0 0 :'         : (10, 4, None ),
    'loss_lpips_1 at 0 1 :'         : (10, 4, None ),
    'loss_lpips_1 at 0 2 :'         : (10, 4, None ),
    'loss_lpips_1 at 1 0 :'         : (10, 4, None ),
    'loss_lpips_1 at 1 1 :'         : (10, 4, None ),
    'loss_lpips_1 at 1 2 :'         : (10, 4, None ),
    'loss_rec_1 at 0 :'             : (10, 4, None ),
    'loss_rec_1 at 1 :'             : (10, 4, None ),
    'orig, ID clip, lpips rec lmk:' : (10, 4, None ),
    'c_ref True'                    : ( 3, 5, None ),
    'c_ref False'                   : ( 1, 1, None ),
    'ffn_gate_input'                : ( 3, 3, None ),#-------------MoE-----------
    'vis-ffn_gate_input'            : ( 3, 3, None ),
    '[warning]: no param to sync'   : (10,1, None ),#-------------TP-----------
    '[TP] shared sync counts'       : (10,1, None ),
    '[Conv2d param stats] count, name (sorted desc):': (0,1, None ),#-------------upcycle-----------
    'avg full_name='                : ( 0, 1, None ),
}
def gate_(id_, *args, **kw, ): # gate for some vis or print behaviour, just for vis/debug
    # return 0
    if 1 and not ( hasattr(global_,'TP_enable') and global_.TP_enable ):
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank()!=0: return

    global _gate_total_runs, _gate_total_calls
    tu = _gate_k2tu.get(id_, None)
    if tu is None:
        return 0
    max_run, interval, prob = tu
    if max_run==0:
        return 0
    if id_ not in _gate_total_runs: # Initialize counters for this ID if not present
        _gate_total_runs[id_] = 0
        _gate_total_calls[id_] = 0
    if _gate_total_runs[id_] >= max_run: # Check if we've reached the maximum runs
        return False
    _gate_total_calls[id_] += 1
    if _gate_total_calls[id_] % interval != 0:
        return False
    if prob is not None:
        import random
        if random.random() > prob:
            return False
    _gate_total_runs[id_] += 1
    return True

def str_t(): # eg. '0608-17.12.30'
    from datetime import datetime
    now = datetime.now()
    month_day = f"{now.month:02d}{now.day:02d}"
    hour_min_second = f"{now.hour:02d}.{now.minute:02d}.{now.second:02d}"
    ret = f"{month_day}-{hour_min_second}"
    return ret
def str_t_pid(): # eg. '0608-17.12.30-180165'
    if hasattr(global_,'TP_enable') and global_.TP_enable:
        _suffix = global_.rank_
    else:
        _suffix = os.getpid()
    return f"{str_t()}-{_suffix}"

def printC(*args, **kw): # controled print
    if gate_(args[0]):
        return print(*args, **kw)

#--------------------

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # disable tf onednn-related warnings
# from skimage.io import imsave


def path_img_2_path_mask( path_img, check_mask_exists = 1 , reuse_if_exists = True, label_mode="RF12_"):
    assert label_mode=="RF12_"
    assert label_mode in ('RF12',"RF12_",), label_mode
    assert 'semantic_mask' not in str(path_img), path_img
    path_img = Path(path_img)
    if 1:
        _suffix = {
            # "RF12" :'-semantic_mask',
            "RF12_":'-semantic_mask',
        }[label_mode]
        path_mask = path_img.parent / f"{path_img.stem}{_suffix}.png"
    if check_mask_exists or not reuse_if_exists:
        if not path_mask.exists() or not reuse_if_exists:
            from gen_semantic_mask import gen_semantic_mask
            vis_path = None
            # vis_path = person_folder.parent.parent / 'vis_semantic_mask' / f"{person_stem}--{path_img.stem}.png"
            gen_semantic_mask(path_img, path_mask, label_mode, vis_path, )
    return path_mask

from my_py_lib.torchModuleName_util import *
if 0:
    #-------------------- terminal color (only for exceptions/logging/warnings)
    import sys; from IPython.core.ultratb import ColorTB; sys.excepthook = ColorTB()
    class _color:  # ANSI escape
        grey = "\x1b[90m"; green = "\x1b[92m"; yellow = "\x1b[93m"
        red = "\x1b[91m"; orange = "\033[38;5;208m"; orange_light = "\033[38;5;214m"
    if 1:
        import logging
        class _CustomFormatter(logging.Formatter):
            # format = "%(asctime)s %(filename)s:%(lineno)d %(funcName)s [%(levelname)-8s] %(message)s"
            format = "%(asctime)s | %(levelname)-5s | %(message)s"
            reset = "\x1b[0m"
            FORMATS = {
                logging.DEBUG: _color.grey + format + reset,
                logging.INFO: _color.green + format + reset,
                logging.WARNING: _color.yellow + format + reset,
                logging.ERROR: _color.red + format + reset,
                logging.CRITICAL: _color.red + format + reset,
            }
            def format(self, record):
                log_fmt = self.FORMATS.get(record.levelno)
                formatter = logging.Formatter(log_fmt, datefmt='%H:%M:%S') # <= print only time, not date
                return formatter.format(record)
        def setup_colored_logging():
            logger = logging.getLogger(); ch = logging.StreamHandler()
            if LOG_debug_level: logger.setLevel(logging.DEBUG); ch.setLevel(logging.DEBUG)
            else:     logger.setLevel(logging.INFO);  ch.setLevel(logging.INFO)
            ch.setFormatter(_CustomFormatter()); logger.addHandler(ch)
        setup_colored_logging()
    if 1:
        import warnings
        def _custom_showwarning(msg, category, filename, lineno, file=None, line=None):
            reset = "\x1b[0m"; c_file_line = _color.grey; c_cate = _color.orange; c_msg = _color.yellow
            if LOG_debug_level: 
                formatted_message=f"{c_cate}{category.__name__}{reset}: {c_msg}{msg}{reset} {c_file_line}{filename}:{lineno}{reset}"
            else:     formatted_message = f"{c_cate}{category.__name__}{reset}: {c_msg}{msg}{reset}"
            print(formatted_message)
        warnings.showwarning = _custom_showwarning

    if __name__=='__main__':
        logging.warning("This is a warning message in yellow"); logging.error("This is an error message in red")
        warnings.warn("This is a colored warning message")
