# ---------------------------------------------------------     Config  -------------------------------------------------
num_workers :int = 1
DDIM_STEPS = 50
BATCH_SIZE = 1
FIXED_CODE = False
# for vis
SAVE_INTERMEDIATES = True
NUM_grid_in_a_column = 5
# ------------------------------------------------------------------------------------------------------------------------
import sys
import os
from pathlib import Path

cur_dir = os.path.dirname(os.path.abspath(__file__))

from confs import *
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from torchvision.utils import make_grid
from my_py_lib.image_util import imgs_2_grid_A,img_paths_2_grid_A
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import torchvision

from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from Dataset_custom import Dataset_custom
from MoE import offload_unused_tasks__LD
from ldm.models.diffusion.ddpm import LandmarkExtractor
from my_py_lib.torch_util import cleanup_gpu_memory
from gen_lmk_and_mask import gen_lmk_and_mask









# ------------------------------------------------------------------------------------------------------------------------
DDIM_ETA = 0.0
SCALE = 3.0
PRECISION = "full"  # "full" or "autocast"
H = 512
W = 512
C = 4
F = 8
# ------------------------------------------------------------------------------------------------------------------------


def load_first_stage_from_sd14(model: LatentDiffusion, sd14_path: Path) -> None:
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


def save_sample_by_decode(x, model, base_path, segment_id, intermediate_num):
    x = model.decode_first_stage(x)
    x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
    x = x.cpu().permute(0, 2, 3, 1).numpy()
    for i in range(len(x)):
        img = Image.fromarray((x[i] * 255).astype(np.uint8))
        save_path = Path(base_path) / segment_id
        save_path.mkdir(parents=True, exist_ok=True)
        img.save(save_path / f"{intermediate_num}.png")


def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]
    if normalize:
        transform_list += [
            torchvision.transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            )
        ]
    return torchvision.transforms.Compose(transform_list)


def load_model_from_config(ckpt, verbose=1):
    if 1:
        ckpt = Path(ckpt)
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(str(ckpt), map_location="cpu")
        if isinstance(pl_sd, dict) and "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
    else:
        print("DEBUG_skip_load_ckpt")
    if 1:
        from init_model import get_moe
        model: LatentDiffusion = get_moe()
        model.ptsM_Generator = LandmarkExtractor(include_visualizer=True, img_256_mode=False)
        cleanup_gpu_memory()
    if 1:
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            pretty_print_torch_module_keys(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            pretty_print_torch_module_keys(u)
        load_first_stage_from_sd14(model, SD14_localpath)

    offload_unused_tasks__LD(model, TASK, method="del") # for save cuda mem
    model.cuda()
    model.eval()
    return model




def load_pairs(pair_list, tgt, ref):
    if tgt and ref:
        pairs = [(tgt, ref), ]
    elif pair_list:
        pairs = []
        with open(pair_list, "r") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(" ")
                if len(parts) != 2:
                    raise ValueError(f"Invalid pair list line {line_num}: expected white-space-separated tgt/ref. got {parts=}")
                pairs.append((parts[0], parts[1]))
    else:
        raise ValueError("No input pairs provided. Use --tgt/--ref or --pair-list.")
    print(f"{pairs=}")
    return pairs


def un_norm(x):
    return (x + 1.0) / 2.0


def un_norm_clip(x1):
    x = x1 * 1.0
    reduce = False
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
        reduce = True
    x[:, 0, :, :] = x[:, 0, :, :] * 0.26862954 + 0.48145466
    x[:, 1, :, :] = x[:, 1, :, :] * 0.26130258 + 0.4578275
    x[:, 2, :, :] = x[:, 2, :, :] * 0.27577711 + 0.40821073
    if reduce:
        x = x.squeeze(0)
    return x


if __name__ == "__main__":
    pairs = load_pairs(args.pair_list, args.tgt, args.ref)

    out_dir = Path(args.out_dir)
    result_path = out_dir / "results"
    grid_path = out_dir / "grid"
    inter_path = out_dir / "intermediates"
    inter_pred_path = inter_path / "pred_x0"
    inter_noised_path = inter_path / "noised"
    out_dir.mkdir(parents=False, exist_ok=True)
    result_path.mkdir(parents=False, exist_ok=True)
    grid_path.mkdir(parents=False, exist_ok=True)
    inter_path.mkdir(parents=False, exist_ok=True)
    if SAVE_INTERMEDIATES:
        inter_pred_path.mkdir(parents=False, exist_ok=True)
        inter_noised_path.mkdir(parents=False, exist_ok=True)
    paths_tgt = [p[0] for p in pairs]
    paths_ref = [p[1] for p in pairs]
    gen_lmk_and_mask(paths_tgt + paths_ref)

    seed_everything(42)

    model: LatentDiffusion = load_model_from_config(PRETRAIN_CKPT_PATH, )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    dataset = Dataset_custom(
        "test",
        task=TASK,
        paths_tgt=paths_tgt,
        paths_ref=paths_ref,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    start_code = None
    if FIXED_CODE:
        start_code = torch.randn([BATCH_SIZE, C, H // F, W // F], device=device)

    precision_scope = autocast if PRECISION == "autocast" else nullcontext
    grids = []
    grid_stems = []

    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for test_batch, prior, test_model_kwargs, out_stem_batch in tqdm(dataloader):
                    model.set_task(test_model_kwargs)
                    bs = test_batch.shape[0]

                    batch_ = {
                        **test_model_kwargs,
                        "GT": torch.zeros_like(test_model_kwargs["inpaint_image"]),
                    }
                    batch_, c = model.get_input_and_conditioning(batch_, device=device)
                    z_inpaint = batch_["z4_inpaint"]
                    z_inpaint_mask = batch_["tgt_mask_64"]
                    z_ref = batch_["z_ref"]
                    z9 = batch_["z9"]

                    uc = None
                    if SCALE != 1.0:
                        uc = model.learnable_vector[TASK].repeat(bs, 1, 1)

                    shape = [C, H // F, W // F]
                    local_start_code = start_code
                    if FIXED_CODE and (local_start_code is None or local_start_code.shape[0] != bs):
                        local_start_code = torch.randn([bs, C, H // F, W // F], device=device)
                    samples_ddim, intermediates = sampler.sample(
                        S=DDIM_STEPS,
                        conditioning=c,
                        batch_size=bs,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=SCALE,
                        unconditional_conditioning=uc,
                        eta=DDIM_ETA,
                        x_T=local_start_code,
                        log_every_t=100,
                        z_inpaint=z_inpaint,
                        z_inpaint_mask=z_inpaint_mask,
                        z_ref=z_ref,
                        z9=z9,
                    )

                    if SAVE_INTERMEDIATES:
                        intermediate_pred_x0 = intermediates["pred_x0"]
                        intermediate_noised = intermediates["x_inter"]
                        for i in range(len(intermediate_pred_x0)):
                            for j in range(bs):
                                stem = f"{out_stem_batch[j]}"
                                save_sample_by_decode(
                                    intermediate_pred_x0[i][j : j + 1],
                                    model,
                                    inter_pred_path,
                                    stem,
                                    i,
                                )
                                save_sample_by_decode(
                                    intermediate_noised[i][j : j + 1],
                                    model,
                                    inter_noised_path,
                                    stem,
                                    i,
                                )

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                    x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
                    for i, x_sample in enumerate(x_checked_image_torch):
                        stem = f"{out_stem_batch[i]}"
                        out_path = result_path / f"{stem}.png"
                        img = Image.fromarray((x_sample.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                        img.save(out_path)
                        print(f"{out_path=}")

                    for i, x_sample in enumerate(x_checked_image_torch):
                        all_img = []
                        all_img.append(un_norm(test_batch[i]).cpu())
                        if TASK != 2:
                            ref_img = test_model_kwargs["ref_imgs"].squeeze(1)
                            ref_img = torchvision.transforms.Resize([512, 512])(ref_img)
                            ref_img = un_norm_clip(ref_img[i]).cpu()
                        else:
                            ref_img = un_norm(test_model_kwargs["ref512"].squeeze(1)[i]).cpu()
                        all_img.append(ref_img)
                        all_img.append(x_sample)

                        grid = torch.stack(all_img, 0)
                        grid = make_grid(grid)
                        grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
                        img = Image.fromarray(grid.astype(np.uint8))
                        stem = f"{out_stem_batch[i]}"
                        path_save_img = grid_path / f"grid-{stem}.jpg"
                        img.save(path_save_img)
                        print(f"{path_save_img=}")
                        grids.append(img)
                        grid_stems.append(stem)
                        if len(grids) >= NUM_grid_in_a_column:
                                stem_start = grid_stems[0]
                                stem_end = grid_stems[-1]
                                grid_column = imgs_2_grid_A(
                                    grids,
                                    grid_layout='column',
                                    grid_path=os.path.join(grid_path, f"{stem_start}--{stem_end}.jpg"),
                                )
                                grids = []
                                grid_stems = []

                    model.unset_task()

    print(f"Your samples are ready and waiting for you here: {out_dir}")


