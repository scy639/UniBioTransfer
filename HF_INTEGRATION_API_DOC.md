# UniBioTransfer HF Integration API Documentation

This document describes the key classes and functions for Hugging Face Hub integration.

---

## Table of Contents

1. [Core Model Classes](#core-model-classes)
   - [LatentDiffusion](#latentdiffusion)
   - [DDPM](#ddpm)
   - [DiffusionWrapper](#diffusionwrapper)
2. [MoE Architecture Classes](#moe-architecture-classes)
   - [TaskSpecific_MoE](#taskspecific_moe)
   - [ModuleDict_W](#moduledict_w)
   - [FFN_Shared_Plus_TaskLoRA](#ffn_shared_plus_tasklora)
3. [Dataset & Preprocessing](#dataset--preprocessing)
   - [Dataset_custom](#dataset_custom)
   - [LandmarkExtractor](#landmarkextractor)
4. [Inference Functions](#inference-functions)
   - [load_model_from_config](#load_model_from_config)
   - [gen_lmk_and_mask](#gen_lmk_and_mask)
5. [Configuration Constants](#configuration-constants)
6. [Proposed HF Classes](#proposed-hf-classes)
   - [UniBioTransferModel](#unibiotransfermodel)
   - [UniBioTransferPipeline](#unibiotransferpipeline)

---

## Core Model Classes

### LatentDiffusion

**Location:** `ldm/models/diffusion/ddpm.py:333`

Main diffusion model class. Inherits from `DDPM`.

```python
class LatentDiffusion(DDPM):
    def __init__(
        self,
        first_stage_config: dict,       # VAE config
        cond_stage_config: dict,        # CLIP encoder config
        num_timesteps_cond: int = 1,
        cond_stage_key: str = "image",
        cond_stage_trainable: bool = False,
        concat_mode: bool = True,
        cond_stage_forward: callable = None,
        conditioning_key: str = None,
        scale_factor: float = 1.0,
        scale_by_std: bool = False,
        *args,
        **kwargs
    ):
        """
        Initialize the Latent Diffusion model.
        
        Key components initialized:
        - first_stage_model: VAE (AutoencoderKL) for encoding/decoding
        - model: DiffusionWrapper containing UNet
        - encoder_clip_face/hair/head: CLIP encoders for conditioning
        - face_ID_model: ArcFace for ID loss
        - ptsM_Generator: LandmarkExtractor for face landmarks
        - learnable_vector: Task-specific learnable embeddings
        """
```

#### Key Methods

```python
def set_task(self, batch: dict) -> int:
    """
    Set the current task for MoE routing.
    
    Args:
        batch: Dict containing 'task' key with task ID tensor
        
    Returns:
        task: The task ID (0=face, 1=hair, 2=motion, 3=head)
        
    Side effects:
        - Sets self.task
        - Sets global_.task
        - Configures Landmark_cond, Landmarks_weight based on task
    """

def unset_task(self):
    """
    Unset the current task after inference/training.
    
    Side effects:
        - Sets global_.task = None
        - Sets global_.lmk_ = None
        - Deletes self.task
    """

def get_input_and_conditioning(self, batch: dict, device: torch.device = None) -> Tuple[dict, torch.Tensor]:
    """
    Process batch and prepare model inputs.
    
    Args:
        batch: Raw batch from dataloader
        device: Target device
        
    Returns:
        batch: Processed batch with latents (z9, z_inpaint, etc.)
        c: Conditioning tensor (B, num_tokens, 768)
    """

def decode_first_stage(self, z: torch.Tensor, predict_cids: bool = False, force_not_quantize: bool = False) -> torch.Tensor:
    """
    Decode latent to image.
    
    Args:
        z: Latent tensor (B, 4, 64, 64)
        
    Returns:
        Decoded image tensor (B, 3, 512, 512) in [-1, 1]
    """

def encode_first_stage(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
    """
    Encode image to latent.
    
    Args:
        x: Image tensor (B, 3, 512, 512) in [-1, 1]
        
    Returns:
        Latent distribution
    """

def apply_model(self, x_noisy, t, cond, return_ids=False, return_features=False, z_ref=None) -> torch.Tensor:
    """
    Apply UNet diffusion model.
    
    Args:
        x_noisy: Noisy latent (B, C, 64, 64)
        t: Timestep tensor (B,)
        cond: Conditioning dict with 'c_crossattn' key
        z_ref: Reference latent for refNet (B, 4, 64, 64)
        
    Returns:
        Noise prediction (B, 4, 64, 64)
    """
```

---

### DDPM

**Location:** `ldm/models/diffusion/ddpm.py:5`

Base class for diffusion models. Inherits from `pl.LightningModule`.

```python
class DDPM(pl.LightningModule):
    def __init__(
        self,
        unet_config: dict,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        loss_type: str = "l2",
        ckpt_path: str = None,
        ignore_keys: list = [],
        load_only_unet: bool = False,
        monitor: str = "val/loss",
        use_ema: bool = True,
        first_stage_key: str = "image",
        image_size: int = 256,
        channels: int = 3,
        # ... more params
    ):
        """
        Base DDPM implementation with Gaussian diffusion.
        """
```

#### Key Methods

```python
def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
    """
    Add noise to x_start at timestep t.
    
    Args:
        x_start: Clean data (B, C, H, W)
        t: Timestep (B,)
        noise: Optional noise tensor
        
    Returns:
        Noisy sample x_t
    """

def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    """
    Register diffusion schedule buffers (betas, alphas, etc.).
    """

@contextmanager
def ema_scope(self, context=None):
    """
    Context manager to use EMA weights for inference.
    """
```

---

### DiffusionWrapper

**Location:** `ldm/models/diffusion/ddpm.py:1636`

Wraps UNet with conditioning and optional refNet.

```python
class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config: dict, conditioning_key: str):
        """
        Args:
            diff_model_config: UNet config dict
            conditioning_key: 'concat' | 'crossattn' | 'hybrid' | 'adm'
            
        Components:
            - diffusion_model: Main UNet
            - diffusion_model_refNet: Reference UNet (if REFNET.ENABLE)
        """

def forward(self, x, t, c_concat=None, c_crossattn=None, return_features=False, z_ref=None, task=None, _trainer=None) -> torch.Tensor:
    """
    Forward pass through UNet.
    
    Args:
        x: Noisy latent (B, 9 or 14, 64, 64)
        t: Timestep (B,)
        c_crossattn: List of conditioning tensors
        z_ref: Reference latent for refNet
        task: Task ID for routing
        
    Returns:
        Noise prediction (B, 4, 64, 64)
    """
```

---

## MoE Architecture Classes

### TaskSpecific_MoE

**Location:** `MoE.py:55`

Wraps modules for task-specific routing.

```python
class TaskSpecific_MoE(nn.Module):
    def __init__(self, module: nn.Module | list, tasks: tuple):
        """
        Args:
            module: Single module or list of modules (one per task)
            tasks: Tuple of task IDs, e.g., (0, 2, 3)
            
        Example:
            # Create task-specific FFN for tasks 0, 2, 3
            task_ffn = TaskSpecific_MoE(original_ffn, tasks=(0, 2, 3))
        """

def forward(self, *args, **kwargs) -> torch.Tensor:
    """
    Forward through task-specific module.
    
    Uses global_.task to select which module to use.
    """
```

---

### ModuleDict_W

**Location:** `MoE.py:28`

Wrapper around `nn.ModuleDict` for task-specific modules.

```python
class ModuleDict_W(nn.Module):
    def __init__(self, modules: list, keys: list):
        """
        Args:
            modules: List of nn.Module
            keys: List of task IDs
        """

def __getitem__(self, k: int) -> nn.Module:
    """Get module for task k."""

def offload_unused_tasks(self, unused_tasks: list, method: str):
    """
    Offload or delete unused task modules.
    
    Args:
        unused_tasks: List of task IDs to offload
        method: 'del' | 'cpu'
    """
```

---

### FFN_Shared_Plus_TaskLoRA

**Location:** `multiTask_model.py:331`

Shared FFN with per-task LoRA adapters.

```python
class FFN_Shared_Plus_TaskLoRA(nn.Module):
    def __init__(self, l_ffn: list, l_task: list, module_name: str = None):
        """
        Args:
            l_ffn: List of FeedForward modules (one per task)
            l_task: List of task IDs
            module_name: Name for logging
            
        Components:
            - shared_ffn: Averaged FFN (frozen)
            - task_lora_in: Per-task LoRA for input linear
            - task_lora_out: Per-task LoRA for output linear
            - moe_gate_mlp: Optional sparse MoE gate
            - moe_experts_list: Optional sparse MoE experts
        """

def forward(self, x: torch.Tensor, token_pos_grid__cur=None) -> torch.Tensor:
    """
    Args:
        x: Input tensor (B, N, D)
        token_pos_grid__cur: Token positions for gating (B, N, 2)
        
    Returns:
        Output tensor (B, N, D)
    """
```

---

## Dataset & Preprocessing

### Dataset_custom

**Location:** `Dataset_custom.py:70`

Custom dataset for inference.

```python
class Dataset_custom(data.Dataset):
    def __init__(
        self,
        state: str,              # 'test' only
        task: int,               # 0=face, 1=hair, 2=motion, 3=head
        paths_tgt: list,         # List of target image paths
        paths_ref: list,         # List of reference image paths
        name: str = "custom"
    ):
        """
        Dataset for inference on custom image pairs.
        
        Handles:
        - Image loading and resizing to 512x512
        - Semantic mask generation via face_parsing
        - Mediapipe landmark loading from cache
        - CLIP and ID encoder input preparation
        """

def __getitem__(self, index: int) -> tuple:
    """
    Returns:
        image_tensor_resize: Target image (3, 512, 512)
        prior_image_tensor: "None" (legacy)
        ret: Dict with keys:
            - 'inpaint_image': Masked target (3, 512, 512)
            - 'inpaint_mask': Mask (1, 512, 512)
            - 'ref_imgs': Reference for CLIP (3, 224, 224)
            - 'task': Task ID
            - 'enInputs': Dict of encoder inputs
            - 'mediapipe_lmkAll': Landmarks (468 or 478, 2)
            - 'ref_imgs_4unet': Reference for refNet (3, 512, 512)
            - 'ref_mask_512': Reference mask (1, 512, 512)
        out_stem: Output filename stem
    """
```

---

### LandmarkExtractor

**Location:** `lmk_util/lmk_extractor.py:25`

Wrapper for MediaPipe landmark extraction.

```python
class LandmarkExtractor:
    def __init__(self, include_visualizer: bool = False, fps: int = 25, **kw_of_vis):
        """
        Args:
            include_visualizer: If True, also initialize visualizer
            fps: FPS for video processing
        """

def extract_single(self, image: np.ndarray, only_main_lmk: bool = False) -> np.ndarray:
    """
    Extract landmarks from a single image.
    
    Args:
        image: RGB image (H, W, 3), uint8
        only_main_lmk: If True, return only ~95 main landmarks
        
    Returns:
        landmarks: (N, 2) absolute coordinates, or None if detection failed
    """

class LandmarkVisualizer:
    """Visualize landmarks on images."""
    
    def visualize_landmarks(self, image: np.ndarray, landmarks: np.ndarray, target_size=(512, 512), use_connections=True) -> np.ndarray:
        """
        Draw landmarks on image.
        
        Returns:
            BGR image with landmarks drawn
        """
```

---

## Inference Functions

### load_model_from_config

**Location:** `infer.py:106`

```python
def load_model_from_config(ckpt: str | Path, verbose: int = 1) -> LatentDiffusion:
    """
    Load pretrained model from checkpoint.
    
    Args:
        ckpt: Path to checkpoint file
        verbose: Print missing/unexpected keys if 1
        
    Returns:
        model: LatentDiffusion model on CUDA, in eval mode
        
    Process:
        1. Load state_dict from ckpt
        2. Create model via get_moe()
        3. Load state_dict (strict=False)
        4. Load first_stage_model from SD v1.4
        5. offload_unused_tasks__LD() for memory
        6. model.cuda().eval()
    """
```

---

### gen_lmk_and_mask

**Location:** `gen_lmk_and_mask.py`

```python
def gen_lmk_and_mask(paths: list, force_regenerate: bool = False):
    """
    Generate and cache Mediapipe landmarks and semantic masks for images.
    
    Args:
        paths: List of image paths
        force_regenerate: If True, regenerate even if cache exists
        
    Side effects:
        - Creates {img_path.stem}-semantic_mask.png for each image
        - Creates mediapipe cache file for each image
        
    Cache locations:
        - Semantic mask: Same dir as image, suffix '-semantic_mask.png'
        - Mediapipe: Managed by Mediapipe_Result_Cache
    """
```

---

## Configuration Constants

**Location:** `util_and_constant.py`

```python
# Task IDs
TASKS = (0, 1, 2, 3)  # face, hair, motion, head

# Checkpoint paths
SD14_filename = "sd-v1-4.ckpt"
SD14_localpath = Path("checkpoints") / SD14_filename
PRETRAIN_CKPT_PATH = "checkpoints/pretrained.ckpt"
PRETRAIN_JSON_PATH = "checkpoints/pretrained.json"

# Model architecture
CH14: bool = False  # If True, use 14-channel input

class REFNET:
    ENABLE: bool = True
    CH9: bool = False
    task2layerNum = {0: 9, 1: 9, 2: 9, 3: 9}

# Landmark settings
USE_pts: bool = True
NUM_pts: int = 95
READ_mediapipe_result_from_cache: bool = True

# Training settings
TP_enable: bool = True
ZERO1_ENABLE: bool = False
ADAM_or_SGD: bool = False  # True=AdamW, False=SGD
```

**Location:** `global_.py`

```python
task: int = None           # Current task ID (set during forward)
TP_enable: bool = None     # Task parallelism
rank_: int = None          # Distributed rank
moduleName_2_adaRank: dict = {}  # Adaptive rank config from pretrained.json
lmk_: torch.Tensor = None  # Current batch landmarks for gating
```

---

## Proposed HF Classes

### UniBioTransferModel

**Proposed file:** `ldm/models/hf_model.py`

```python
from huggingface_hub import PyTorchModelHubMixin
from ldm.models.diffusion.ddpm import LatentDiffusion

class UniBioTransferModel(LatentDiffusion, PyTorchModelHubMixin):
    """
    Hugging Face Hub compatible wrapper for UniBioTransfer.
    
    Inherits all functionality from LatentDiffusion and adds:
    - from_pretrained(): Load from HF Hub
    - push_to_hub(): Upload to HF Hub
    - save_pretrained(): Save locally
    
    Example:
        model = UniBioTransferModel.from_pretrained("scy639/UniBioTransfer")
        model.set_task({'task': torch.tensor([0])})  # face transfer
        result = model.decode_first_stage(samples)
    """
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str = "scy639/UniBioTransfer",
        task: str = "face",
        device: str = "cuda",
        **kwargs
    ) -> "UniBioTransferModel":
        """
        Load model from Hugging Face Hub.
        
        Args:
            pretrained_model_name_or_path: HF repo ID or local path
            task: Task name ('face', 'hair', 'motion', 'head')
            device: Target device
            
        Returns:
            model: Loaded and ready for inference
            
        Downloads:
            - pretrained.ckpt (main model)
            - pretrained.json (adaptive rank config)
            - sd-v1-4.ckpt (VAE weights)
            - model_ir_se50.pth (ArcFace)
            - shape_predictor_68_face_landmarks.dat (DLIB)
            - 79999_iter.pth (Face parsing)
        """
        
    def _download_dependencies(self):
        """Download all required pretrained weights."""
        
    def _initialize_model(self, task: int):
        """Initialize model structure and load weights."""
```

---

### UniBioTransferPipeline

**Proposed file:** `infer_hf.py`

```python
class UniBioTransferPipeline:
    """
    High-level inference pipeline for UniBioTransfer.
    
    Handles all preprocessing, model inference, and postprocessing.
    
    Example:
        pipeline = UniBioTransferPipeline.from_pretrained("scy639/UniBioTransfer", task="face")
        result = pipeline("target.jpg", "reference.jpg")
        result.save("output.png")
    """
    
    TASK2ID = {"face": 0, "hair": 1, "motion": 2, "head": 3}
    ID2TASK = {0: "face", 1: "hair", 2: "motion", 3: "head"}
    
    def __init__(self, model, task: str, device: str = "cuda"):
        """
        Args:
            model: LatentDiffusion model instance
            task: Task name
            device: Device for inference
        """
        
    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "scy639/UniBioTransfer",
        task: str = "face",
        device: str = "cuda",
        hf_endpoint: str = None
    ) -> "UniBioTransferPipeline":
        """
        Load pipeline from Hugging Face Hub.
        
        Args:
            repo_id: HF repository ID
            task: Task name ('face', 'hair', 'motion', 'head')
            device: Target device
            hf_endpoint: Custom HF endpoint (e.g., 'https://hf-mirror.com')
            
        Returns:
            pipeline: Ready for inference
        """
        
    def __call__(
        self,
        tgt_image: str | Path | Image.Image | np.ndarray,
        ref_image: str | Path | Image.Image | np.ndarray,
        ddim_steps: int = 50,
        scale: float = 3.0,
        seed: int = 42
    ) -> Image.Image:
        """
        Run inference on a pair of images.
        
        Args:
            tgt_image: Target image (path, PIL Image, or numpy array)
            ref_image: Reference image (path, PIL Image, or numpy array)
            ddim_steps: Number of DDIM sampling steps
            scale: CFG scale
            seed: Random seed
            
        Returns:
            result: PIL Image of generated result
        """
        
    def _preprocess_image(self, image) -> torch.Tensor:
        """Convert input to tensor and resize to 512x512."""
        
    def _generate_landmarks(self, image: np.ndarray) -> np.ndarray:
        """Extract landmarks on-the-fly using MediaPipe."""
        
    def _generate_semantic_mask(self, image: np.ndarray) -> np.ndarray:
        """Generate semantic mask using face_parsing model."""
        
    def _prepare_batch(self, tgt_img, ref_img) -> dict:
        """Prepare batch dict for model input."""
```

---

## Data Flow Summary

```
Input: tgt_image, ref_image (paths or PIL)

1. Preprocessing:
   - Load images -> resize to 512x512
   - gen_lmk_and_mask() or on-the-fly:
     - LandmarkExtractor.extract_single() -> 468/478 landmarks
     - Face parsing -> semantic mask
   - Dataset_custom.__getitem__() logic:
     - Create inpaint_image (masked target)
     - Create inpaint_mask (region to transfer)
     - Prepare CLIP inputs (ref_image masked)
     - Prepare refNet inputs

2. Model Forward:
   - model.set_task(batch)
   - batch, c = model.get_input_and_conditioning(batch)
     - encode_first_stage() -> latents
     - conditioning_with_feat() -> CLIP + ID embeddings
   - sampler.sample() -> DDIM sampling
     - apply_model() at each step
   - model.decode_first_stage(samples) -> image

3. Postprocessing:
   - Clamp to [0, 1]
   - Convert to PIL Image

Output: result_image (PIL Image)
```

---

## Task-Specific Behavior

| Task | ID | Preserve Regions | CLIP Input | ID Loss |
|------|-----|------------------|------------|---------|
| face | 0 | [1,2,3,10,5,6,7,9] (face parts) | ref face | Yes |
| hair | 1 | [4] (hair) | ref hair | No |
| motion | 2 | [1-11,20,21] (full head) | tgt head | Yes |
| head | 3 | [1,2,3,10,4,5,6,7,9] (face+hair) | ref head | Yes |

Semantic mask labels (face_parsing):
- 1: skin
- 2: eyebrow
- 3: eye
- 4: hair
- 5: nose
- 6: mouth
- 7: ear
- 9: neck
- 10: ear_ring
- 11: glasses

---

## Memory Requirements

| Component | VRAM (approx.) |
|-----------|----------------|
| Main UNet | ~2.5 GB |
| refNet (if ENABLE) | ~1.5 GB |
| VAE (first_stage_model) | ~0.5 GB |
| CLIP encoder | ~0.5 GB |
| ID model (ArcFace) | ~0.3 GB |
| Landmark extractor | ~0.2 GB |
| **Total (single task)** | ~5-6 GB |
| **Minimum requirement** | **11 GB** (per README) |

---

## HF Space Requirements

For `requirements.txt`:

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
huggingface_hub>=0.16.0
gradio>=4.0.0
omegaconf>=2.3.0
einops>=0.6.0
albumentations>=1.3.0
opencv-python>=4.8.0
mediapipe>=0.10.0
Pillow>=10.0.0
scikit-image>=0.21.0
natsort>=8.4.0
pytorch-lightning>=2.0.0
```

---

## Appendix: Key File Locations

| File | Purpose |
|------|---------|
| `ldm/models/diffusion/ddpm.py` | Main model classes (LatentDiffusion, DDPM, DiffusionWrapper) |
| `ldm/models/diffusion/ddim.py` | DDIM sampler |
| `MoE.py` | Task-specific MoE modules |
| `multiTask_model.py` | LoRA-based shared+task modules |
| `init_model.py` | Model initialization (get_moe) |
| `infer.py` | Command-line inference script |
| `Dataset_custom.py` | Inference dataset |
| `gen_lmk_and_mask.py` | Landmark and mask generation |
| `lmk_util/lmk_extractor.py` | LandmarkExtractor class |
| `confs.py` | Argument parsing and constants |
| `util_and_constant.py` | Global constants |
| `global_.py` | Global variables (task, rank, etc.) |
| `LatentDiffusion.yaml` | Model config |
| `download_checkpoints.py` | HF download script |
