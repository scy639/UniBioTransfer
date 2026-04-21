"""
Hugging Face Hub compatible model wrapper for UniBioTransfer.
Provides from_pretrained() and push_to_hub() functionality via PyTorchModelHubMixin.
"""
from pathlib import Path
import torch
import json
import copy
import os
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

import global_
from ldm.models.diffusion.ddpm import LatentDiffusion, LandmarkExtractor
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from MoE import offload_unused_tasks__LD
from multiTask_model import TaskSpecific_MoE, replace_modules_lossless
from my_py_lib.torch_util import cleanup_gpu_memory

TASKS = (0, 1, 2, 3)
TASK_NAME2ID = {"face": 0, "hair": 1, "motion": 2, "head": 3}
TASK_ID2NAME = {v: k for k, v in TASK_NAME2ID.items()}

SD14_FILENAME = "sd-v1-4.ckpt"
SD14_REPO = "CompVis/stable-diffusion-v-1-4-original"
PRETRAIN_REPO = "scy639/UniBioTransfer"


def _load_first_stage_from_sd14(model, sd14_path):
    """Load first_stage_model (VAE) from SD v1.4 checkpoint."""
    print(f"Loading first_stage_model from {sd14_path}")
    sd14 = torch.load(str(sd14_path), map_location="cpu")
    if isinstance(sd14, dict) and "state_dict" in sd14:
        sd14_sd = sd14["state_dict"]
    else:
        sd14_sd = sd14

    prefixes = ["first_stage_model.", "model.first_stage_model."]
    fs_sd = {}
    for prefix in prefixes:
        for k, v in sd14_sd.items():
            if k.startswith(prefix):
                fs_sd[k[len(prefix):]] = v
        if fs_sd:
            break

    if not fs_sd:
        raise RuntimeError("Could not find first_stage_model weights in SD v1-4 checkpoint.")

    model.first_stage_model.load_state_dict(fs_sd, strict=True)


class UniBioTransferModel(LatentDiffusion, PyTorchModelHubMixin):
    """
    Hugging Face Hub compatible wrapper for UniBioTransfer.
    
    Inherits from LatentDiffusion and adds HF Hub integration via PyTorchModelHubMixin.
    
    Usage:
        # Load model from HF Hub
        model = UniBioTransferModel.from_pretrained("scy639/UniBioTransfer", task="face")
        
        # Push to HF Hub
        model.push_to_hub("your-repo/UniBioTransfer")
    
    Args:
        config: Model config dict (handled by PyTorchModelHubMixin)
        task: Task name or ID (face/hair/motion/head)
        **kwargs: Additional arguments passed to LatentDiffusion
    """

    def __init__(self, config=None, task="face", **kwargs):
        self._task_name = task if isinstance(task, str) else TASK_ID2NAME.get(task, "face")
        self._task_id = TASK_NAME2ID.get(self._task_name, 0) if isinstance(task, str) else task
        
        global_.task = self._task_id
        
        if config is None:
            config = {}
        
        super().__init__(**config)
        
        self._hf_config = {
            "task": self._task_name,
            "task_id": self._task_id,
        }

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path=None,
        task="face",
        device="cuda",
        download_sd14=True,
        download_deps=True,
        cache_dir=None,
        **kwargs,
    ):
        """
        Load model from Hugging Face Hub.
        
        Args:
            pretrained_model_name_or_path: HF repo ID or local path. 
                Default: "scy639/UniBioTransfer"
            task: Task name (face/hair/motion/head) or task ID (0/1/2/3)
            device: Device to load model to ("cuda" or "cpu")
            download_sd14: Whether to download SD v1.4 VAE weights
            download_deps: Whether to download other dependencies (ArcFace, DLIB, face_parsing)
            cache_dir: Cache directory for downloads
            **kwargs: Additional arguments
        
        Returns:
            UniBioTransferModel: Loaded model
        """
        task_id = TASK_NAME2ID.get(task, task) if isinstance(task, str) else task
        task_name = TASK_ID2NAME.get(task_id, "face")
        
        global_.task = task_id
        
        if pretrained_model_name_or_path is None:
            pretrained_model_name_or_path = PRETRAIN_REPO
        
        repo_id = pretrained_model_name_or_path
        
        cache_dir = Path(cache_dir) if cache_dir else Path(".")
        
        ckpt_path = cache_dir / "checkpoints" / "pretrained.ckpt"
        json_path = cache_dir / "checkpoints" / "pretrained.json"
        sd14_path = cache_dir / "checkpoints" / SD14_FILENAME
        arcface_path = cache_dir / "Other_dependencies" / "arcface" / "model_ir_se50.pth"
        dlib_path = cache_dir / "Other_dependencies" / "DLIB_landmark_det" / "shape_predictor_68_face_landmarks.dat"
        face_parsing_path = cache_dir / "Other_dependencies" / "face_parsing" / "79999_iter.pth"
        
        def _download_file(repo, filename, local_path):
            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {filename} from {repo}...")
            token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
            hf_hub_download(
                repo_id=repo,
                filename=filename,
                local_dir=str(local_path.parent),
                local_dir_use_symlinks=False,
                token=token,
            )
        
        if not ckpt_path.exists():
            _download_file(repo_id, "checkpoints/pretrained.ckpt", ckpt_path)
        if not json_path.exists():
            _download_file(repo_id, "checkpoints/pretrained.json", json_path)
        
        if download_sd14 and not sd14_path.exists():
            _download_file(SD14_REPO, SD14_FILENAME, sd14_path)
        
        if download_deps:
            if not arcface_path.exists():
                _download_file(repo_id, "Other_dependencies/arcface/model_ir_se50.pth", arcface_path)
            if not dlib_path.exists():
                _download_file(repo_id, "Other_dependencies/DLIB_landmark_det/shape_predictor_68_face_landmarks.dat", dlib_path)
            if not face_parsing_path.exists():
                _download_file(repo_id, "Other_dependencies/face_parsing/79999_iter.pth", face_parsing_path)
        
        seed_everything(42)
        
        cur_dir = Path(__file__).parent
        yaml_path = cur_dir / "LatentDiffusion.yaml"
        if not yaml_path.exists():
            yaml_path = Path("LatentDiffusion.yaml")
        
        model_config = OmegaConf.load(yaml_path).model
        model = instantiate_from_config(model_config)
        
        with open(json_path, 'r') as f:
            global_.moduleName_2_adaRank = json.load(f)
        print(f"Loaded adaptive rank config from {json_path}")
        
        _src0 = copy.deepcopy(model.model.diffusion_model)
        _src1 = copy.deepcopy(model.model.diffusion_model)
        _src2 = copy.deepcopy(model.model.diffusion_model)
        _src3 = copy.deepcopy(model.model.diffusion_model)
        replace_modules_lossless(
            model.model.diffusion_model,
            [_src0, _src1, _src2, _src3],
            [0, 1, 2, 3],
            parent_name=".model.diffusion_model",
        )
        
        model.ID_proj_out = TaskSpecific_MoE([
            copy.deepcopy(model.ID_proj_out),
            copy.deepcopy(model.ID_proj_out),
            copy.deepcopy(model.ID_proj_out),
        ], [0, 2, 3])
        model.landmark_proj_out = TaskSpecific_MoE([
            copy.deepcopy(model.landmark_proj_out),
            copy.deepcopy(model.landmark_proj_out),
            copy.deepcopy(model.landmark_proj_out),
        ], [0, 2, 3])
        model.proj_out_source__head = TaskSpecific_MoE([
            copy.deepcopy(model.proj_out_source__head),
            copy.deepcopy(model.proj_out_source__head),
        ], [2, 3])

        from util_and_constant import REFNET
        if REFNET.ENABLE:
            shared_ref = model.model.diffusion_model_refNet
            src0 = shared_ref
            src1 = copy.deepcopy(shared_ref)
            src2 = copy.deepcopy(shared_ref)
            src3 = copy.deepcopy(shared_ref)
            replace_modules_lossless(shared_ref, [src0, src1, src2, src3], [0, 1, 2, 3], parent_name=".model.diffusion_model_refNet", for_refnet=True)
            from ldm.models.diffusion.bank import Bank
            model.model.bank = Bank(
                reader=model.model.diffusion_model,
                writer=model.model.diffusion_model_refNet
            )

        print(f"Loading model weights from {ckpt_path}")
        pl_sd = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(pl_sd, dict) and "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd

        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0:
            print(f"Missing keys: {len(m)}")
        if len(u) > 0:
            print(f"Unexpected keys: {len(u)}")

        _load_first_stage_from_sd14(model, sd14_path)

        # offload_unused_tasks__LD(model, task_id, method="cpu")
        
        model.ptsM_Generator = LandmarkExtractor(include_visualizer=True, img_256_mode=False)
        cleanup_gpu_memory()
        
        # ZeroGPU 兼容：只在 device 不是 "cpu" 且 CUDA 可用时才移动到 GPU
        # 如果传入 device="cpu"，保持模型在 CPU 上（ZeroGPU 初始化时不碰显卡）
        if device != "cpu" and torch.cuda.is_available():
            model = model.to(torch.device(device))
        else:
            model = model.to(torch.device("cpu"))
        model.eval()
        
        model._task_id = task_id
        model._task_name = task_name
        model._hf_config = {"task": task_name, "task_id": task_id}
        
        return model
