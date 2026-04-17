import argparse
import os

import torch

from evaluation import evaluate, test
from heatmap_loader import heatmap_dataloader
from KeyRe_ID_model import KeyRe_ID


def parse_args():
    parser = argparse.ArgumentParser(description="KeyRe-ID Test")
    parser.add_argument("--dataset_name", default="MARS", type=str)
    parser.add_argument("--dataset_root", default="./data", type=str, help="Path to dataset root directory")
    parser.add_argument("--ViT_path", required=True, type=str, help="Path to pretrained ViT weights")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--eval_pool", default="avg", choices=["avg", "max"])
    return parser.parse_args()


def validate_args(args):
    if not os.path.exists(args.ViT_path):
        raise ValueError(f"Invalid ViT_path: {args.ViT_path}")
    if not os.path.exists(args.dataset_root):
        raise ValueError(f"Invalid dataset_root: {args.dataset_root}")


def main():
    args = parse_args()
    validate_args(args)

    use_gpu = args.device == "cuda" and torch.cuda.is_available()
    _, _, num_classes, camera_num, _, q_val_set, g_val_set = heatmap_dataloader(
        args.dataset_name,
        args.dataset_root,
    )

    model = KeyRe_ID(num_classes=num_classes, camera_num=camera_num, pretrainpath=args.ViT_path)
    print("Running load_param")
    model.load_param(args.ViT_path)

    if use_gpu:
        model = model.cuda()

    cmc, mean_ap = test(model, q_val_set, g_val_set, pool=args.eval_pool, use_gpu=use_gpu)
    print("CMC: %.4f, mAP: %.4f" % (cmc, mean_ap))


if __name__ == "__main__":
    main()
