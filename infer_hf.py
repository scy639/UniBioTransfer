"""
High-level inference pipeline for UniBioTransfer.
Designed for easy use in Hugging Face Spaces and other applications.

ZeroGPU Compatible:
- Supports CPU initialization (device="cpu")
- Dynamically switches to CUDA during inference when called from @spaces.GPU
"""
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import cv2

import global_
from hf_model import UniBioTransferModel, TASK_NAME2ID, TASK_ID2NAME
from ldm.models.diffusion.ddim import DDIMSampler
from pytorch_lightning import seed_everything

DDIM_STEPS_DEFAULT = 50
SCALE_DEFAULT = 3.0


H, W, C, F = 512, 512, 4, 8
class UniBioTransferPipeline:
    """
    High-level pipeline for UniBioTransfer inference.
    """

    def __init__(self, model, task="face", device="cpu"):
        """
        Initialize pipeline with a loaded model.
        """
        self.model = model
        self.task = task
        self.task_id = TASK_NAME2ID.get(task, task) if isinstance(task, str) else task
        self._init_device = device

        global_.task = self.task_id
        self.model.task = self.task_id

        self.sampler = DDIMSampler(model)

    @classmethod
    def from_pretrained(
        cls,
        repo_id="scy639/UniBioTransfer",
        task="face",
        device="cpu",
        cache_dir=None,
        **kwargs,
    ):
        """
        Load pipeline from Hugging Face Hub.
        """
        model = UniBioTransferModel.from_pretrained(
            pretrained_model_name_or_path=repo_id,
            task=task,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
        return cls(model, task=task, device=device)

    def set_task(self, task):
        """Switch to a different task."""
        self.task = task
        self.task_id = TASK_NAME2ID.get(task, task) if isinstance(task, str) else task
        global_.task = self.task_id
        self.model.task = self.task_id

    def __call__(
        self,
        tgt_image,
        ref_image,
        ddim_steps=DDIM_STEPS_DEFAULT,
        scale=SCALE_DEFAULT,
        seed=42,
    ):
        """
        Run inference on a pair of images.
        """
        seed_everything(seed)

        tgt_img = self._load_image(tgt_image)
        ref_img = self._load_image(ref_image)

        tgt_img = self._resize_image(tgt_img, (H, W))
        ref_img = self._resize_image(ref_img, (H, W))

        result = self._run_inference(tgt_img, ref_img, ddim_steps, scale)

        result_img = self._postprocess(result)
        return result_img

    def _load_image(self, img):
        """Load image from various formats."""
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        elif isinstance(img, np.ndarray):
            return Image.fromarray(img).convert("RGB")
        elif isinstance(img, (str, Path)):
            return Image.open(img).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

    def _resize_image(self, img, size):
        """Resize image to target size."""
        if img.size != size:
            img = img.resize(size, Image.LANCZOS)
        return img

    def _run_inference(self, tgt_img, ref_img, ddim_steps, scale):
        """
        Run diffusion sampling.
        完全复用 infer.py 的逻辑，使用 dataloader。
        """
        from Dataset_custom import Dataset_custom
        from gen_lmk_and_mask import gen_lmk_and_mask
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tgt_path = Path(tmpdir) / "tgt.png"
            ref_path = Path(tmpdir) / "ref.png"
            tgt_img.save(tgt_path)
            ref_img.save(ref_path)

            gen_lmk_and_mask([str(tgt_path), str(ref_path)], write_cache=True)

            dataset = Dataset_custom(
                "test",
                task=self.task_id,
                paths_tgt=[str(tgt_path)],
                paths_ref=[str(ref_path)],
            )

            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                num_workers=1,
                pin_memory=True,
                shuffle=False,
                drop_last=False,
            )

            run_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(run_device)

            with torch.no_grad():
                for test_batch, prior, test_model_kwargs, out_stem_batch in dataloader:
                    test_batch = test_batch.to(run_device)
                    for k, v in test_model_kwargs.items():
                        if isinstance(v, torch.Tensor):
                            test_model_kwargs[k] = v.to(run_device)

                    self.model.set_task(test_model_kwargs)
                    bs = test_batch.shape[0]

                    batch_ = {
                        **test_model_kwargs,
                        "GT": torch.zeros_like(test_model_kwargs["inpaint_image"]),
                    }
                    batch_, c = self.model.get_input_and_conditioning(batch_, device=run_device)

                    z_inpaint = batch_["z4_inpaint"]
                    z_inpaint_mask = batch_["tgt_mask_64"]
                    z_ref = batch_["z_ref"]
                    z9 = batch_["z9"]

                    uc = None
                    if scale != 1.0:
                        uc = self.model.learnable_vector[self.task_id].repeat(bs, 1, 1)

                    shape = [C, H // F, W // F]
                    start_code = None

                    samples_ddim, _ = self.sampler.sample(
                        S=ddim_steps,
                        conditioning=c,
                        batch_size=bs,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc,
                        eta=0.0,
                        x_T=start_code,
                        log_every_t=100,
                        z_inpaint=z_inpaint,
                        z_inpaint_mask=z_inpaint_mask,
                        z_ref=z_ref,
                        z9=z9,
                    )

                    x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                    self.model.unset_task()

                    return x_samples_ddim[0]

    def _postprocess(self, tensor):
        """Convert model output tensor to PIL Image."""
        img_array = tensor.cpu().permute(1, 2, 0).numpy()
        img_array = (img_array * 255).astype(np.uint8)
        return Image.fromarray(img_array)


def infer_single(
    tgt_path,
    ref_path,
    task="face",
    output_path=None,
    ddim_steps=DDIM_STEPS_DEFAULT,
    scale=SCALE_DEFAULT,
    device="cuda",
):
    """
    Convenience function for single inference.
    """
    pipeline = UniBioTransferPipeline.from_pretrained(task=task, device=device)
    result = pipeline(tgt_path, ref_path, ddim_steps=ddim_steps, scale=scale)

    if output_path is not None:
        result.save(output_path)
        print(f"Saved result to {output_path}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UniBioTransfer inference")
    parser.add_argument("--task", type=str, default="face", choices=["face", "hair", "motion", "head"])
    parser.add_argument("--tgt", type=str, required=True, help="Path to target image")
    parser.add_argument("--ref", type=str, required=True, help="Path to reference image")
    parser.add_argument("--out", type=str, default="result.png", help="Output path")
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    result = infer_single(
        args.tgt,
        args.ref,
        task=args.task,
        output_path=args.out,
        ddim_steps=args.ddim_steps,
        scale=args.scale,
        device=args.device,
    )

    print(f"Inference complete. Result shape: {result.size}")
