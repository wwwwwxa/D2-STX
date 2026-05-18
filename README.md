# [YOUR METHOD NAME]: [YOUR PAPER TITLE]

**[Your Name(s)]**

[[Paper]](#) | [[Project Page]](#)

> **Note**: This repository extends [ESCounts (ACCV 2024)](https://github.com/sinhasaptarshi/EveryShotCounts) with pose-based counting and spatiotemporal cross-modal fusion.

---

## Overview

This codebase provides three training/evaluation pipelines for video repetition counting:

| Script | Modality | Description |
|--------|----------|-------------|
| `video_counting_train.py` | Video only | Trains a VideoMAE-based exemplar counting model |
| `pose_counting_train.py` | Pose only | Trains a PoseMAE-based exemplar counting model |
| `frame_STfusionmodel_Visualization.py` | Video + Pose | Three-stage training of a spatiotemporal cross-modal fusion model, with attention visualization |

---

## Installation

Create and activate a conda environment:

```bash
conda create -n repcount python=3.8
conda activate repcount
```

Install dependencies:

```bash
pip install av==10.0.0
pip install einops==0.3.2
pip install numpy
pip install opencv-python==4.8.1.78
pip install pandas
pip install -e git+https://github.com/facebookresearch/pytorchvideo.git@fae0d89a194a2c1ca99e59eab6eedd40bde38726#egg=pytorchvideo
pip install tqdm==4.59.0
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 \
    -f https://download.pytorch.org/whl/torch_stable.html
pip install simplejson
python -m pip install detectron2 \
    -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
pip install psutil scikit-image timm tensorboardX thop seaborn
pip install -U openmim && mim install mmcv-full
```

> The `mmaction` library is already included in this repository.

---

## Dataset Download

**RepCount**: Download from [SVIP-Lab](https://svip-lab.github.io/dataset/RepCount_dataset.html) and place under `data/RepCount/`

**Countix**: Download from [RepNet](https://sites.google.com/view/repnet) and place under `data/Countix/`

**UCFRep**: Download from [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) and place under `data/UCFRep/`

---

## Feature Extraction

### Video features (VideoMAE)

Download the pretrained VideoMAE-v2 encoder from [here](https://dl.fbaipublicfiles.com/pyslowfast/masked_models/VIT_B_16x4_MAE_PT.pyth) and place it in `pretrained_models/`.

Extract spatio-temporal tokens for videos:

```bash
python save_swim_features.py --dataset RepCount --model VideoMAE \
    --num_gpus 1 --data_path data/RepCount
```

Extract tokens for exemplars:

```bash
python save_swim_features.py --dataset RepCount --model VideoMAE \
    --num_gpus 1 --save_exemplar_encodings True --data_path data/RepCount
```

This creates `saved_VideoMAEtokens_RepCount/` and `exemplar_VideoMAEtokens_RepCount/`.

### Pose features

Extract pose tokens (replace paths with your pose encoder):

```bash
python save_swim_features.py --dataset RepCount --model PoseMAE \
    --num_gpus 1 --data_path data/RepCount
```

This creates `saved_PoseMAEtokens_RepCount/` and `exemplar_PoseMAEtokens_RepCount/`.

---

## Training

### 1. Video-only model

```bash
python video_counting_train.py \
    --dataset RepCount \
    --tokens_dir saved_VideoMAEtokens_RepCount \
    --exemplar_dir exemplar_VideoMAEtokens_RepCount \
    --save_path saved_models_repcount/video_model \
    --multishot --iterative_shots \
    --lr 5e-6 --epochs 100 \
    --token_pool_ratio 0.4 \
    --log_dir logs/video_run \
    --use_tensorboard True
```

### 2. Pose-only model

```bash
python pose_counting_train.py \
    --dataset RepCount \
    --tokens_dir saved_PoseMAEtokens_RepCount \
    --exemplar_dir exemplar_PoseMAEtokens_RepCount \
    --save_path saved_models_repcount/pose_model \
    --multishot --iterative_shots \
    --lr 5e-6 --epochs 100 \
    --token_pool_ratio 0.4 \
    --log_dir logs/pose_run \
    --use_tensorboard True
```

### 3. Spatiotemporal fusion model (three-stage)

**Stage 1 — Train video encoder:**

```bash
python frame_STfusionmodel_Visualization.py \
    --training_stage 1 \
    --dataset RepCount \
    --tokens_dir saved_VideoMAEtokens_RepCount \
    --exemplar_dir exemplar_VideoMAEtokens_RepCount \
    --pose_tokens_dir saved_PoseMAEtokens_RepCount \
    --pose_exemplar_dir exemplar_PoseMAEtokens_RepCount \
    --save_path saved_models_repcount/fusion \
    --stage1_epochs 50 \
    --multishot --iterative_shots \
    --lr 5e-6 --log_dir logs/fusion_stage1
```

**Stage 2 — Train pose encoder:**

```bash
python frame_STfusionmodel_Visualization.py \
    --training_stage 2 \
    --stage1_checkpoint saved_models_repcount/fusion/best_obo_XXX_stage1_epochXXX.pyth \
    --dataset RepCount \
    --tokens_dir saved_VideoMAEtokens_RepCount \
    --exemplar_dir exemplar_VideoMAEtokens_RepCount \
    --pose_tokens_dir saved_PoseMAEtokens_RepCount \
    --pose_exemplar_dir exemplar_PoseMAEtokens_RepCount \
    --save_path saved_models_repcount/fusion \
    --stage2_epochs 35 \
    --multishot --iterative_shots \
    --lr 5e-6 --log_dir logs/fusion_stage2
```

**Stage 3 — Train fusion module:**

```bash
python frame_STfusionmodel_Visualization.py \
    --training_stage 3 \
    --stage1_checkpoint saved_models_repcount/fusion/best_obo_XXX_stage1_epochXXX.pyth \
    --stage2_checkpoint saved_models_repcount/fusion/best_obo_XXX_stage2_epochXXX.pyth \
    --dataset RepCount \
    --tokens_dir saved_VideoMAEtokens_RepCount \
    --exemplar_dir exemplar_VideoMAEtokens_RepCount \
    --pose_tokens_dir saved_PoseMAEtokens_RepCount \
    --pose_exemplar_dir exemplar_PoseMAEtokens_RepCount \
    --save_path saved_models_repcount/fusion \
    --stage3_epochs 50 \
    --fusion_mode pose_as_query \
    --embed_dim 512 --num_heads 8 \
    --multishot --iterative_shots \
    --lr 5e-6 --log_dir logs/fusion_stage3
```

Supported `--fusion_mode` values: `pose_as_query`, `video_as_query`, `bidirectional_gating`.

---

## Testing

### Video / Pose models

```bash
python video_counting_train.py \
    --dataset RepCount \
    --tokens_dir saved_VideoMAEtokens_RepCount \
    --exemplar_dir exemplar_VideoMAEtokens_RepCount \
    --trained_model saved_models_repcount/xxx.pyth \
    --multishot --iterative_shots \
    --get_overlapping_segments \
    --only_test
```

### Fusion model with attention visualization

```bash
python frame_STfusionmodel_Visualization.py \
    --dataset RepCount \
    --tokens_dir saved_VideoMAEtokens_RepCount \
    --exemplar_dir exemplar_VideoMAEtokens_RepCount \
    --pose_tokens_dir saved_PoseMAEtokens_RepCount \
    --pose_exemplar_dir exemplar_PoseMAEtokens_RepCount \
    --trained_model saved_models_repcount/fusion/xxx.pyth \
    --multishot --iterative_shots \
    --only_test \
    --enable_attention_visualization \
    --attention_save_dir attention_visualizations/
```

Visualizations are saved under `attention_visualizations/`, grouped by spatial and temporal attention directions.

---

## Pretrained Models

Download our pretrained checkpoints:

| Model | Dataset | OBO | MAE | Download |
|-------|---------|-----|-----|----------|
| Video model | RepCount | [FILL] | [FILL] | [link] |
| Pose model  | RepCount | [FILL] | [FILL] | [link] |
| Fusion model | RepCount | [FILL] | [FILL] | [link] |

Place downloaded `.pyth` files under `saved_models_repcount/`.

---

## Citation

If you find this work helpful, please cite our paper:

```bibtex
@inproceedings{[yourcitation],
  title   = {[YOUR PAPER TITLE]},
  author  = {[AUTHORS]},
  booktitle = {[VENUE]},
  year    = {[YEAR]},
}
```

This work builds on [ESCounts](https://arxiv.org/abs/2403.18074):

```bibtex
@InProceedings{sinha2024every,
  title     = {Every Shot Counts: Using Exemplars for Repetition Counting in Videos},
  author    = {Sinha, Saptarshi and Stergiou, Alexandros and Damen, Dima},
  booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year      = {2024},
}
```