# UniBioTransfer


### [Paper](https://arxiv.org/abs/2603.19637)
### [Project page](https://scy639.github.io/UniBioTransfer.github.io/)

## Abstract
>Deepface generation has traditionally followed a task-driven paradigm, where distinct tasks (e.g., face transfer and hair transfer) are addressed by task-specific models. Nevertheless, this single-task setting severely limits model generalization and scalability. A unified model capable of solving multiple deepface generation tasks in a single pass represents a promising and practical direction, yet remains challenging due to data scarcity and cross-task conflicts arising from heterogeneous attribute transformations. To this end, we propose UniBioTransfer, the first unified framework capable of handling both conventional deepface tasks (e.g., face transfer and face reenactment) and shape-varying transformations (e.g., hair transfer and head transfer). Besides, UniBioTransfer naturally generalizes to unseen tasks, like lip, eye, and glasses transfer, with minimal fine-tuning. Generally, UniBioTransfer addresses data insufficiency in multi-task generation through a unified data construction strategy, including a swapping-based corruption mechanism designed for spatially dynamic attributes like hair. It further mitigates cross-task interference via an innovative BioMoE, a mixture-of-experts based model coupled with a novel two-stage training strategy that effectively disentangles task-specific knowledge. Extensive experiments demonstrate the effectiveness, generalization, and scalability of UniBioTransfer, outperforming both existing unified models and task-specific methods across a wide range of deepface generation tasks.



## TODO
- [ ] Inference code & pretrained checkpoint of
    - [x] The 4 main tasks (face/hair/motion/head transfer)
    - [ ] All tasks
- [ ] Traning code
- [ ] Evaluation (on FFHQ dataset) code
- [ ] Support newer base model like FLUX


## Setup
```
conda create -n "unibio" python=3.10.16 -y
conda activate unibio
sh setup.sh
```


Download pre-trained checkpoints via `python download_checkpoints.py`





## Citing Us
If you find our work valuable, we kindly ask you to consider citing our paper.


```
TODO
```

## Acknowledgements

This code borrows heavily from [REFace](https://github.com/Sanoojan/REFace).


