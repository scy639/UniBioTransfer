# UniBioTransfer


[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.19637)
[![Project Page](https://img.shields.io/badge/Project-Page-blue?logo=googlechrome&logoColor=white)](https://scy639.github.io/UniBioTransfer.github.io/)

## Abstract
>Deepface generation has traditionally followed a task-driven paradigm, where distinct tasks (e.g., face transfer and hair transfer) are addressed by task-specific models. Nevertheless, this single-task setting severely limits model generalization and scalability. A unified model capable of solving multiple deepface generation tasks in a single pass represents a promising and practical direction, yet remains challenging due to data scarcity and cross-task conflicts arising from heterogeneous attribute transformations. To this end, we propose UniBioTransfer, the first unified framework capable of handling both conventional deepface tasks (e.g., face transfer and face reenactment) and shape-varying transformations (e.g., hair transfer and head transfer). Besides, UniBioTransfer naturally generalizes to unseen tasks, like lip, eye, and glasses transfer, with minimal fine-tuning. Generally, UniBioTransfer addresses data insufficiency in multi-task generation through a unified data construction strategy, including a swapping-based corruption mechanism designed for spatially dynamic attributes like hair. It further mitigates cross-task interference via an innovative BioMoE, a mixture-of-experts based model coupled with a novel two-stage training strategy that effectively disentangles task-specific knowledge. Extensive experiments demonstrate the effectiveness, generalization, and scalability of UniBioTransfer, outperforming both existing unified models and task-specific methods across a wide range of deepface generation tasks.

## System requirement
Minimum **11GB** VRAM required e.g., NVIDIA GeForce RTX 2080 Ti


## TODO
- [ ] Inference code & pretrained checkpoint of
    - [x] The 4 main tasks (face/hair/motion/head transfer)
    - [ ] All other tasks
- [ ] Traning code
- [ ] Evaluation (on FFHQ testset) code
- [ ] Support newer base model like FLUX


## Setup

### 1. Environment setup
```
conda create -n "unibio" python=3.10.16 -y
conda activate unibio
sh setup.sh
```


### 2. Download pre-trained weight
Download pre-trained checkpoints via `python download_checkpoints.py`


<small>*Tip*: If you have trouble accessing Hugging Face (e.g., in mainland China), you can firstly set the mirror endpoint: `export HF_ENDPOINT=https://hf-mirror.com`</small>


## Inference


```bash
python infer.py --task-name=face   --tgt=examples/face/tgt.png   --ref=examples/face/ref.png   --out-dir=examples/face
python infer.py --task-name=hair   --tgt=examples/hair/tgt.png   --ref=examples/hair/ref.png   --out-dir=examples/hair
python infer.py --task-name=motion --tgt=examples/motion/tgt.png --ref=examples/motion/ref.png --out-dir=examples/motion
python infer.py --task-name=head   --tgt=examples/head/tgt.png   --ref=examples/head/ref.png   --out-dir=examples/head
```

### Web interface (Gradio demo)

```bash
python app.py
```


## Citing Us
If you find our work valuable, we kindly ask you to consider citing our paper.


```
@article{sun2026unibiotransfer,
  title={UniBioTransfer: A Unified Framework for Multiple Biometrics Transfer},
  author={Sun, Caiyi and Sun, Yujing and Li, Xiangyu and Zheng, Yuhang and Ren, Yiming and Wang, Jiamin and Ma, Yuexin and Yiu, Siu-Ming},
  journal={arXiv preprint arXiv:2603.19637},
  year={2026}
}
```

## Acknowledgements

This code borrows heavily from [REFace](https://github.com/Sanoojan/REFace).


