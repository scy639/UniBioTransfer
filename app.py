"""
Hugging Face Space demo for UniBioTransfer.
Gradio interface for face/hair/motion/head transfer.

ZeroGPU Compatible:
- Model initialized on CPU (no GPU memory during startup)
- Inference wrapped with @spaces.GPU decorator
- Thread-safe global variable access with Lock
"""

import threading
import torch
from PIL import Image
import numpy as np

# ==========================================
# 兼容层：处理本地测试 vs HF ZeroGPU 环境
# ==========================================
try:
    import spaces
    print("Detected spaces library (Hugging Face environment).")
except ImportError:
    print("Local environment detected. Mocking spaces.GPU...")
    class spaces:
        @staticmethod
        def GPU(func):
            return func  # 本地测试时，装饰器变为空壳，直接执行原函数

from infer_hf import UniBioTransferPipeline

# 锁和全局单例 Pipeline
inference_lock = threading.Lock()
global_pipeline :UniBioTransferPipeline = None


def get_pipeline(task):
    """
    单例模式：全局只初始化一次模型（放在 CPU），后续只切换任务。
    强制写死 CPU，保证 ZeroGPU 全局初始化时不碰显卡。
    """
    global global_pipeline
    if global_pipeline is None:
        print("Initializing pipeline once on CPU...")
        # 强制写死 CPU，保证 ZeroGPU 全局初始化时不碰显卡
        global_pipeline = UniBioTransferPipeline.from_pretrained(
            repo_id="scy639/UniBioTransfer",
            task=task,
            device="cpu",
        )
    else:
        # 如果模型已经在内存中，只需切换 task ID 即可
        print(f"Switching existing pipeline to task: {task}")
        global_pipeline.set_task(task)
    return global_pipeline


# 核心：将所有会用到 GPU 的前向推理逻辑包裹在这里
@spaces.GPU
def run_gpu_inference(pipeline:UniBioTransferPipeline, tgt_pil, ref_pil, ddim_steps, scale, seed, num_images):
    """
    这里是 ZeroGPU 分配算力的地方。进入此函数时可以安全地 to("cuda")。
    如果是在本地服务器，这个装饰器没用，但内部的 .to("cuda") 同样生效。
    """
    return pipeline(
        tgt_pil,
        ref_pil,
        ddim_steps=ddim_steps,
        scale=scale,
        seed=seed,
        num_images=num_images,
    )


def inference(task, tgt_img, ref_img, ddim_steps, seed, num_images):
    """
    Run inference for the demo.
    """
    if tgt_img is None or ref_img is None:
        return None, "Please upload both target and reference images."

    try:
        # 1. 拿模型 (此时模型在 CPU)
        pipeline = get_pipeline(task)

        tgt_pil = Image.fromarray(tgt_img).convert("RGB")
        ref_pil = Image.fromarray(ref_img).convert("RGB")

        # 2. 加锁，防止并发污染 global_.task，进入 GPU 推理
        with inference_lock:
            results = run_gpu_inference(
                pipeline,
                tgt_pil,
                ref_pil,
                int(ddim_steps),
                float(3),
                int(seed),
                int(num_images)
            )

        return results, f"Success! Task: {task} transfer completed."

    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(f"{error_msg}")
        return None, error_msg


def create_demo():
    """Create Gradio demo interface."""
    import gradio as gr

    with gr.Blocks(title="UniBioTransfer") as demo:
        gr.Markdown(
            """
            # UniBioTransfer

            Perform face transfer, hair transfer, motion transfer (face reenactment), and head transfer.

            - **Face Transfer**: Transfer face identity from reference to target
            - **Hair Transfer**: Transfer hairstyle from reference to target
            - **Motion Transfer**: Transfer motion(expression+head pose) from reference to target
            - **Head Transfer**: Transfer entire head from reference to target

            [Code](https://github.com/scy639/UniBioTransfer)
            [Project Page](https://scy639.github.io/UniBioTransfer.github.io/)
            [Paper](https://arxiv.org/abs/2603.19637)
            """
        )

        with gr.Row():
            with gr.Column():
                task_dropdown = gr.Dropdown(
                    choices=["face", "hair", "motion", "head"],
                    value="face",
                    label="Task",
                    info="Select the transfer type",
                )

                with gr.Row():
                    tgt_image = gr.Image(
                        label="Target Image",
                        type="numpy",
                        height=300,
                    )
                    ref_image = gr.Image(
                        label="Reference Image",
                        type="numpy",
                        height=300,
                    )

                with gr.Row():
                    ddim_steps = gr.Slider(
                        minimum=4,
                        maximum=50,
                        value=50,
                        step=1,
                        label="DDIM Steps",
                        info="More steps = better quality but slower",
                    )
                    # scale = gr.Slider(
                    #     minimum=1.0,
                    #     maximum=10.0,
                    #     value=3.0,
                    #     step=0.5,
                    #     label="CFG Scale",
                    #     info="Guidance scale for conditioning",
                    # )

                seed = gr.Number(
                    value=42,
                    label="Random Seed",
                    info="For reproducibility",
                )

                num_images = gr.Slider(
                    minimum=1,
                    maximum=32,
                    value=4,
                    step=1,
                    label="Number of output images",
                    info="Multi-output with different initial noise",
                )

                run_btn = gr.Button("Run Inference", variant="primary")

            with gr.Column():
                output_gallery = gr.Gallery(
                    label="Results",
                    height=800,
                    columns=2,
                )
                status_text = gr.Textbox(
                    label="Status",
                    lines=3,
                )

        gr.Markdown(
"""
### Usage
1. Upload a **target image** (the person whose face/hair/motion/head will be modified)
2. Upload a **reference image** (the source of the attribute to transfer)
3. Select the **task** type
4. Click "Run Inference"

### Requirements
- Works best when the heads in the two input images have similar sizes.
"""
        )

        run_btn.click(
            fn=inference,
            inputs=[task_dropdown, tgt_image, ref_image, ddim_steps, seed, num_images],
            outputs=[output_gallery, status_text],
        )

        task_dropdown.change(
            fn=lambda t: f"Task switched to: {t} transfer",
            inputs=[task_dropdown],
            outputs=[status_text],
        )

        gr.Examples(
            examples=[
                ["face", "examples/face/tgt.png", "examples/face/ref.png",       20, 42, 4],
                ["hair", "examples/hair/tgt.png", "examples/hair/ref.png",       20, 42, 4],
                ["motion", "examples/motion/tgt.png", "examples/motion/ref.png", 20, 42, 4],
                ["head", "examples/head/tgt.png", "examples/head/ref.png",       20, 42, 4],
            ],
            inputs=[task_dropdown, tgt_image, ref_image, ddim_steps, seed, num_images],
            label="Examples",
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch()
