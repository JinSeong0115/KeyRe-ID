# DukeMTMC-VideoReID Experiment Plan

This repository is now prepared to run KeyRe-ID on DukeMTMC-VideoReID through the dataset name `DukeMTMC-VideoReID`.

Actual dataset path on this machine:

```text
/home/data2/bj-noh/datasets/DukeMTMC-VideoReID
```

H200-compatible training environment:

```text
/home/data2/bj-noh/envs/keyreid-duke-torch2
```

Downloaded ViT-B/16 pretraining checkpoint:

```text
/home/data2/bj-noh/weights/jx_vit_base_p16_224-80ecf9dd.pth
```

## Expected Dataset Layout

The loader accepts common DukeMTMC-VideoReID layouts such as:

```text
DukeMTMC-VideoReID/
  train/
  query/
  gallery/
```

or:

```text
DukeMTMC-VideoReID/
  bbox_train/
  bbox_query/
  bbox_gallery/
```

The dataset root passed to scripts can be either the parent directory or the DukeMTMC-VideoReID directory itself.

## Heatmap Preparation

KeyRe-ID requires six-channel keypoint heatmaps in addition to RGB clips. If the heatmaps do not already exist, generate them as follows:

```bash
python keypoint/extract_keypoint.py \
  --dataset_path /home/data2/bj-noh/datasets/DukeMTMC-VideoReID \
  --output_dir /home/data2/bj-noh/datasets/DukeMTMC-VideoReID/keypoints \
  --skip_existing \
  --log_every 1000

python keypoint/keypoint_to_mask.py \
  --dataset_path /home/data2/bj-noh/datasets/DukeMTMC-VideoReID \
  --output_dir /home/data2/bj-noh/datasets/DukeMTMC-VideoReID/heatmap \
  --skip_existing \
  --log_every 1000
```

The heatmap script now detects common split names, including `train`, `query`, `gallery`, `bbox_train`, `bbox_query`, and `bbox_gallery`.

## Training

```bash
python train.py \
  --dataset_name DukeMTMC-VideoReID \
  --dataset_root /home/data2/bj-noh/datasets/DukeMTMC-VideoReID \
  --ViT_path /home/data2/bj-noh/weights/jx_vit_base_p16_224-80ecf9dd.pth
```

## Evaluation

```bash
python test.py \
  --dataset_name DukeMTMC-VideoReID \
  --dataset_root /home/data2/bj-noh/datasets/DukeMTMC-VideoReID \
  --ViT_path ./weights/Marsbest.pth \
  --device cuda
```

Use the actual Duke-trained checkpoint for final reporting. `Marsbest.pth` is only appropriate for a smoke test or cross-dataset sanity check.

## Current Status

Dataset loader smoke test passed:

```text
train:   702 ids, 2196 tracklets
query:   702 ids, 702 tracklets
gallery: 1110 ids, 2636 tracklets
total images: 927268
```

Full keypoint extraction is running in tmux:

```bash
tmux attach -t duke_keypoints
tail -f /home/data2/bj-noh/duke_keypoint_extraction.log
find /home/data2/bj-noh/datasets/DukeMTMC-VideoReID/keypoints -type f -name '*.pose' | wc -l
```
