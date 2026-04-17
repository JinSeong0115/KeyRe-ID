import os, glob, numpy as np
from PIL import Image
import openpifpaf
from tqdm import tqdm

PRID_ROOT = "./data/PRID-2011"
IMG_H, IMG_W = 256, 128
SIGMA = 7.0

PART_KEYPOINTS = {
    0: [0, 1, 2, 3, 4],
    1: [5, 6, 11, 12],
    2: [5, 7, 9],
    3: [6, 8, 10],
    4: [11, 13, 15],
    5: [12, 14, 16],
}

def make_gaussian_heatmap(cx, cy, h, w, sigma=SIGMA):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2)).astype(np.float32)

def keypoints_to_heatmap(predictions, orig_h, orig_w):
    heatmap = np.zeros((6, IMG_H, IMG_W), dtype=np.float32)
    if not predictions:
        return heatmap
    pred = max(predictions, key=lambda p: p.score)
    kps = pred.data
    for part_idx, kp_indices in PART_KEYPOINTS.items():
        for kp_idx in kp_indices:
            x, y, conf = kps[kp_idx]
            if conf > 0.1:
                sx, sy = x * IMG_W / orig_w, y * IMG_H / orig_h
                hm = make_gaussian_heatmap(sx, sy, IMG_H, IMG_W, SIGMA) * conf
                heatmap[part_idx] = np.maximum(heatmap[part_idx], hm)
    return heatmap

def process_cam(cam_name):
    predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16')
    cam_dir = os.path.join(PRID_ROOT, 'multi_shot', cam_name)
    all_images = []
    for pd in sorted(glob.glob(os.path.join(cam_dir, 'person_*'))):
        pid = os.path.basename(pd)
        for img in sorted(glob.glob(os.path.join(pd, '*.png'))):
            all_images.append((img, pid))
    
    print(f"Processing {cam_name}: {len(all_images)} images")
    out_dir = os.path.join(PRID_ROOT, 'heatmap', 'bbox_train')  # same dir for all
    
    for img_path, pid in tqdm(all_images, desc=cam_name):
        fname = os.path.splitext(os.path.basename(img_path))[0] + '.npy'
        save_dir = os.path.join(out_dir, pid)
        save_path = os.path.join(save_dir, fname)
        if os.path.exists(save_path):
            continue
        os.makedirs(save_dir, exist_ok=True)
        img = Image.open(img_path).convert('RGB')
        predictions, _, _ = predictor.pil_image(img)
        np.save(save_path, keypoints_to_heatmap(predictions, img.size[1], img.size[0]))

if __name__ == '__main__':
    process_cam('cam_a')
    process_cam('cam_b')
    # symlink bbox_test -> bbox_train
    bbox_test = os.path.join(PRID_ROOT, 'heatmap', 'bbox_test')
    if not os.path.exists(bbox_test):
        os.symlink(os.path.join(PRID_ROOT, 'heatmap', 'bbox_train'), bbox_test)
    print("Done!")
