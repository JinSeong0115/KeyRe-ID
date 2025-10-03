import cv2
import numpy as np
from collections import OrderedDict
import sys
import os
import json
from scipy.signal.windows import gaussian
import argparse

# --- Keypoint Definitions ---
joints_dict = OrderedDict()
joints_dict['head'] = ['nose', 'Leye', 'Reye', 'LEar', 'REar']
joints_dict['torso'] = ['LS', 'RS', 'LH', 'RH']
joints_dict['left_arm'] = ['LE', 'LW']
joints_dict['right_arm'] = ['RE', 'RW']
joints_dict['left_leg'] = ['LK', 'LA']
joints_dict['right_leg'] = ['RK', 'RA']

pose_keypoints = ['nose', 'Leye', 'Reye', 'LEar', 'REar', 
                  'LS', 'RS', 'LE', 'RE', 'LW', 'RW', 
                  'LH', 'RH', 'LK', 'RK', 'LA', 'RA']

keypoints_dict = {name: idx for idx, name in enumerate(pose_keypoints)}


# --- Utility Functions ---
def gkern(kernlen=21, std=None):
    if std is None:
        std = kernlen / 4
    gkern1d = gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

def rescale_keypoints(rf_keypoints, size, new_size):
    w, h = size
    new_w, new_h = new_size
    rf_keypoints = rf_keypoints.copy()
    rf_keypoints[:, 0] = rf_keypoints[:, 0] * new_w / w
    rf_keypoints[:, 1] = rf_keypoints[:, 1] * new_h / h
    return rf_keypoints


# --- Heatmap Generation Class ---
class KeypointsToMasks:
    def __init__(self, g_scale=11, vis_thresh=0.1, vis_continous=False):
        self.g_scale = g_scale
        self.vis_thresh = vis_thresh
        self.vis_continous = vis_continous
        self.gaussian = None

    def __call__(self, kp_xyc, img_size, output_size):
        kp_xyc_r = rescale_keypoints(kp_xyc, img_size, output_size)
        return self._compute_joints_gaussian_heatmaps(output_size, kp_xyc_r)

    def _compute_joints_gaussian_heatmaps(self, output_size, kp_xyc):
        w, h = output_size
        num_groups = len(joints_dict)
        group_heatmaps = np.zeros((num_groups, h, w))
        kernel = self.get_gaussian_kernel(output_size)
        g_radius = kernel.shape[0] // 2

        for group_idx, (group_name, joint_names) in enumerate(joints_dict.items()):
            temp_heatmap = np.zeros((h, w))
            for joint_name in joint_names:
                idx = keypoints_dict[joint_name]
                kp = kp_xyc[idx]
                if kp[2] <= self.vis_thresh and not self.vis_continous:
                    continue
                kpx, kpy = int(kp[0]), int(kp[1])
                if not (0 <= kpx < w and 0 <= kpy < h):
                    continue
                rt, rb = max(0, kpy - g_radius), min(h, kpy + g_radius + 1)
                rl, rr = max(0, kpx - g_radius), min(w, kpx + g_radius + 1)
                kernel_y_start, kernel_y_end = g_radius - (kpy - rt), g_radius + (rb - kpy - 1)
                kernel_x_start, kernel_x_end = g_radius - (kpx - rl), g_radius + (rr - kpx - 1)
                sub_kernel = kernel[kernel_y_start:kernel_y_end, kernel_x_start:kernel_x_end]
                patch = temp_heatmap[rt:rb, rl:rr]
                if patch.shape == sub_kernel.shape:
                    temp_heatmap[rt:rb, rl:rr] = np.maximum(patch, sub_kernel)
            group_heatmaps[group_idx] = temp_heatmap
        return group_heatmaps

    def get_gaussian_kernel(self, output_size):
        if self.gaussian is None:
            w, h = output_size
            g_radius = int(w / self.g_scale)
            kernel_size = g_radius * 2 + 1
            kernel = gkern(kernel_size)
            kernel = kernel / np.max(kernel)
            self.gaussian = kernel
        return self.gaussian


# --- Main Processing Function ---
def run_heatmap_generation(args):
    # CORRECTED: Define root paths directly from arguments to prevent path duplication.
    image_root = args.dataset_path
    keypoint_root = os.path.join(image_root, "keypoints")
    heatmap_root = args.output_dir
    
    phases = ["bbox_train", "bbox_test"]
    
    kp2mask = KeypointsToMasks(g_scale=11, vis_thresh=0.1, vis_continous=False)
    
    for phase in phases:
        phase_img_dir = os.path.join(image_root, phase)
        phase_kp_dir = os.path.join(keypoint_root, phase)
        phase_heatmap_dir = os.path.join(heatmap_root, phase)
        os.makedirs(phase_heatmap_dir, exist_ok=True)
        
        if not os.path.exists(phase_img_dir):
            print(f"Warning: Image directory not found, skipping: {phase_img_dir}")
            continue
            
        person_ids = sorted(os.listdir(phase_img_dir))
        print(f"Processing phase: {phase} ({len(person_ids)} IDs)")

        for person_id in person_ids:
            person_img_dir = os.path.join(phase_img_dir, person_id)
            if not os.path.isdir(person_img_dir): continue
            
            person_kp_dir = os.path.join(phase_kp_dir, person_id)
            if not os.path.exists(person_kp_dir):
                print(f"Warning: Keypoint folder not found, skipping: {person_kp_dir}")
                continue
            
            person_heatmap_dir = os.path.join(phase_heatmap_dir, person_id)
            os.makedirs(person_heatmap_dir, exist_ok=True)
            
            for img_file in sorted(os.listdir(person_img_dir)):
                if not img_file.endswith(".jpg"): continue
                
                img_path = os.path.join(person_img_dir, img_file)
                kp_file = img_file.replace(".jpg", ".pose")
                kp_path = os.path.join(person_kp_dir, kp_file)
                
                if not os.path.isfile(kp_path): continue
                
                img = cv2.imread(img_path)
                if img is None: continue
                h_img, w_img, _ = img.shape

                with open(kp_path, "r") as f:
                    kp_data = json.load(f)
                kp_array = np.array(kp_data, dtype=np.float32)

                heatmap = kp2mask(kp_array, (w_img, h_img), (w_img, h_img))
                
                npy_save_path = os.path.join(person_heatmap_dir, img_file.replace(".jpg", ".npy"))
                np.save(npy_save_path, heatmap)

    print("🔹 All heatmap generation completed!")


# --- Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate keypoint heatmaps for a dataset.")
    parser.add_argument('--dataset_path', type=str, default='../data/MARS -output_dir', required=True, help='Root path to the image dataset directory')
    parser.add_argument('--output_dir', type=str, default='../data/MARS/heatmaps', required=True, help='Path to the directory where heatmaps will be saved')
    args = parser.parse_args()
    
    run_heatmap_generation(args)


# python ./keypoint/keypoint_to_mask.py --dataset_path ./data/MARS -output_dir ../data/MARS/heatmaps
