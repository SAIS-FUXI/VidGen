# VIDGEN-1M

## VIDGEN-1M: A LARGE-SCALE DATASET FOR TEXT-TO-VIDEO GENERATION

[![arXiv](https://img.shields.io/badge/arXiv-2408.02629-b31b1b.svg)](https://arxiv.org/abs/2408.02629)
[![Project Page](https://img.shields.io/badge/Project-Website-blue)](https://sais-fuxi.github.io/projects/vidgen-1m/)

## Introduction
we present VidGen-1M, a superior training dataset for text-to-video models. Produced through a coarse-to-fine curation strategy, this dataset guarantees high-quality videos and detailed captions with excellent temporal consistency. We trained a video generation model using this data and open-source the model.

## Contents
- [Install](#install)
- [VidGen-1M Datasets](#datasets)
- [Model Weights](#weights)
- [Sampling ](#sampling )

## Install
1. Clone this repository
2. Install Package
```Shell
conda create -n videodiff python=3.10
conda activate videodiff

pip install torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tqdm einops omegaconf bigmodelvis deepspeed tensorboard timm==0.9.16 ninja opencv-python opencv-python-headless ftfy bs4 beartype colossalai accelerate ultralytics webdataset

pip install -U xformers --index-url https://download.pytorch.org/whl/cu118
```

## VidGen-1M Datasets
To assist the community in researching and learning about video generation, we have made public [VidGen-1M](https://huggingface.co/datasets/Fudan-FUXI/VIDGEN-1M) high-quality video data.

## Model Weights
Please download the [Model weight](https://huggingface.co/Fudan-FUXI/VIDGEN-v1.0) from huggingface.

## Sampling 
You can use a single GPU or multiple GPUs for inference. The script has various arguments.
```bash
bash scripts/sample_t2v.sh
```

## Citation
```bibtex
@article{tan2024vdgen-1m,
  title={VIDGEN-1M: A LARGE-SCALE DATASET FOR TEXTTO-VIDEO GENERATION},
  author={Tan, Zhiyu and Yang, Xiaomeng and Qin, Luozheng and Li, Hao},
  journal={arXiv preprint arXiv:2408.02629},
  year={2024},
  institution={Fudan University and Shanghai Academy of AI for Science},
}
```
