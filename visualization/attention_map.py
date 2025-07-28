import os
import cv2
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from heatmap_loader import heatmap_dataloader
from KeyRe_ID_model import KeyRe_ID
import torch.nn.functional as F

# ───── setting ─────
DATASET = "Mars"
CHECKPOINT = "../weights/Marsbest.pth"
SAVE_DIR = "./visualization/attention_map"
BATCH_SIZE = 1
NUM_WORKERS = 4
IMG_H, IMG_W = 256, 128
ALPHA = 0.3
CMAP = cv2.COLORMAP_JET

IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD = [0.5, 0.5, 0.5]

def denormalize(img_tensor):
    arr = img_tensor.cpu().numpy().transpose(1, 2, 0)
    arr = (arr * IMAGENET_STD + IMAGENET_MEAN) * 255
    return arr[..., ::-1].astype(np.uint8)

def overlay_heatmap(img_bgr, heatmap, alpha=ALPHA):
    hmap = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    hmap = cv2.normalize(hmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hmap = cv2.applyColorMap(hmap, CMAP)
    return cv2.addWeighted(img_bgr, 1 - alpha, hmap, alpha, 0)

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    _, _, num_cls, cam_num, _, q_set, _ = heatmap_dataloader(DATASET)
    loader = DataLoader(q_set, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, collate_fn=lambda b: b[0])

    model = KeyRe_ID(num_classes=num_cls, camera_num=cam_num, pretrainpath=None)
    state_dict = torch.load(CHECKPOINT, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    for imgs, heatmaps, pid, camids, paths in tqdm(loader, desc="Processing QK attention", unit="query"):
        B, T, C, H, W = imgs.shape
        imgs = imgs.to(device)
        pH, pW = H // 16, W // 16

        imgs_reshaped = imgs.view(B * T, C, H, W)

        with torch.no_grad():
            # patch embed + cls token + pos embed
            x = model.base.patch_embed(imgs_reshaped)                    # [BT, 128, C]
            cls_token = model.base.cls_token.expand(x.shape[0], -1, -1) # [BT, 1, C]
            x = torch.cat((cls_token, x), dim=1)                         # [BT, 129, C]
            x = x + model.base.pos_embed[:, :x.size(1), :]              # [BT, 129, C]
            x = model.base.pos_drop(x)

            # Handling attention blocks (visualize only the last block)
            for blk in model.base.blocks[:-1]:
                x = blk(x)

            x = model.base.blocks[-1](x)

        # attention Extraction
        attn_map = model.base.blocks[-1].attn.get_attention()  # [BT, heads, 129, 129]
        attn_cls = attn_map[:, :, 0, 1:]  # [BT, heads, 128]
        attn_avg = attn_cls.mean(dim=1)   # [BT, 128]

        pid_str = f"{int(pid):04d}"
        save_pid_dir = os.path.join(SAVE_DIR)
        os.makedirs(save_pid_dir, exist_ok=True)

        for t in range(T):
            frame = imgs[0, t]
            frame_bgr = denormalize(frame)

            attn_2d = attn_avg[t].view(pH, pW).cpu().numpy()  # [8, 16]
            overlay = overlay_heatmap(frame_bgr, attn_2d)

            filename = f"pid_{pid_str}_f{t+1:03d}.jpg"
            save_path = os.path.join(save_pid_dir, filename)
            cv2.imwrite(save_path, overlay)

    print(f"All QK-based part-attention maps saved to {SAVE_DIR}")
