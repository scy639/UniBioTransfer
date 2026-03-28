TASK :int = 0


import argparse
parser = argparse.ArgumentParser(description="Custom inference for tgt/ref image pairs.")
parser.add_argument("--task-name", type=str, 
    default=['face','hair','motion','head'][TASK], 
    help="face|hair|motion|head")
parser.add_argument("--out-dir", type=str, default='examples/outputs', help="Output directory")
# option 1: pass 2 paths
parser.add_argument("--tgt", type=str, default=None, help="Path to target image. if None, will use paths read from --pair-list")
parser.add_argument("--ref", type=str, default=None, help="Path to reference image")
# option 2: pass a txt containing paths
parser.add_argument("--pair-list", type=str, default='examples/inputs.txt', help="white-space-separated list file: tgt_path ref_path")
args = parser.parse_args()

#-----------------------------------------set TASK--------------------------------------------------------------------------

task_name :str = args.task_name
if task_name is None:
    pass
else:
    TASK :int = {
        'face': 0,
        'hair': 1,
        'motion': 2,
        'head': 3,
    }[task_name]
print(f'task: {task_name} transfer (ID: {TASK})')

#---------------------------------------------------------------------------------------------------------------------
from util_and_constant import *
from get_mask import *
from util_cv2 import *

