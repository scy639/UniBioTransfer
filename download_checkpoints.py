from pathlib import Path
import os
from imports import *



def _download(repo_id, filename, local_path: Path) -> Path:
    local_path = Path(local_path)
    from huggingface_hub import hf_hub_download
    local_path.parent.mkdir(parents=True, exist_ok=True)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    print(f"downloading to {local_path}")
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(local_path.parent),
        local_dir_use_symlinks=False,
        token=token,
    )



_download("CompVis/stable-diffusion-v-1-4-original",SD14_filename, SD14_localpath)

_download("scy639/UniBioTransfer",PRETRAIN_CKPT_PATH, ".")
_download("scy639/UniBioTransfer",PRETRAIN_JSON_PATH, ".")

_download("scy639/UniBioTransfer","Other_dependencies/arcface/model_ir_se50.pth", ".")
_download("scy639/UniBioTransfer","Other_dependencies/face_parsing/79999_iter.pth", ".")
