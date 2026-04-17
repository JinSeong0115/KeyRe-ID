import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from heatmap_loader import heatmap_dataloader
from KeyRe_ID_model import KeyRe_ID
import argparse

# --- Constants and Utility Functions ---

IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD = [0.5, 0.5, 0.5]

def denormalize(img_tensor):
    # Denormalizes a tensor image back to a BGR numpy image for OpenCV.
    arr = img_tensor.cpu().numpy().transpose(1, 2, 0)
    arr = (arr * IMAGENET_STD + IMAGENET_MEAN) * 255
    return arr[..., ::-1].astype(np.uint8)

def overlay_heatmap(img_bgr, heatmap, alpha=0.3, cmap=cv2.COLORMAP_JET):
    # Overlays a heatmap onto a BGR image.
    hmap = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    hmap = cv2.normalize(hmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hmap = cv2.applyColorMap(hmap, cmap)
    return cv2.addWeighted(img_bgr, 1 - alpha, hmap, alpha, 0)

# --- Main Logic Function ---

def main(args):
    # The main pipeline for generating and saving attention maps.
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    _, _, num_cls, cam_num, _, q_set, _ = heatmap_dataloader(args.dataset, args.dataset_root)
    loader = DataLoader(
        q_set, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        collate_fn=lambda b: b[0]
    )

    # Load model
    model = KeyRe_ID(num_classes=num_cls, camera_num=cam_num, pretrainpath=None)
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    # Process and visualize attention maps
    for imgs, heatmaps, pid, camids, paths in tqdm(loader, desc="Processing QK attention", unit="query"):
        B, T, C, H, W = imgs.shape
        imgs = imgs.to(device)
        pH, pW = H // 16, W // 16

        imgs_reshaped = imgs.view(B * T, C, H, W)

        with torch.no_grad():
            # Pass through the model to get attention maps
            x = model.base.patch_embed(imgs_reshaped)
            cls_token = model.base.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = x + model.base.pos_embed[:, :x.size(1), :]
            x = model.base.pos_drop(x)

            for blk in model.base.blocks:
                x = blk(x)
        
        # Extract attention from the last block
        attn_map = model.base.blocks[-1].attn.get_attention()
        attn_cls = attn_map[:, :, 0, 1:]
        attn_avg = attn_cls.mean(dim=1)

        pid_str = f"{int(pid):04d}"
        save_pid_dir = os.path.join(args.save_dir, pid_str)
        os.makedirs(save_pid_dir, exist_ok=True)

        for t in range(T):
            frame = imgs[0, t]
            frame_bgr = denormalize(frame)

            attn_2d = attn_avg[t].view(pH, pW).cpu().numpy()
            overlay = overlay_heatmap(frame_bgr, attn_2d)

            filename = f"pid_{pid_str}_frame_{t+1:03d}.jpg"
            save_path = os.path.join(save_pid_dir, filename)
            cv2.imwrite(save_path, overlay)

    print(f"All QK-based part-attention maps saved to {args.save_dir}")

# --- Execution Block ---

if __name__ == "__main__":
    # Handles parsing command-line arguments and calls the main function.
    parser = argparse.ArgumentParser(description="Generate and visualize attention maps from a model.")
    parser.add_argument('--dataset', type=str, default='MARS', help='Name of the dataset (e.g., MARS)')
    parser.add_argument('--dataset_root', type=str, required=True, help='Root directory of the dataset (e.g., ./data)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint file (.pth)')
    parser.add_argument('--save_dir', type=str, default='./visualization/attention_map', help='Directory to save the output images')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for data loading')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    
    args = parser.parse_args()
    main(args)


# python3 -m visualization.attention_map --dataset_root ./data --checkpoint /path/to/your/model.pth --save_dir ./visualization/attention_map