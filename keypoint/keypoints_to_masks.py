import cv2
import numpy as np
from collections import OrderedDict
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchreid.utils.imagetools import gkern
from glob import glob
import json
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian

########################################
#       PoseTrack Keypoint Grouping     #
########################################

# PoseTrack joint grouping (joint names included in each group)
joints_dict = OrderedDict()
joints_dict['head'] = ['nose', 'Leye', 'Reye', 'LEar', 'REar']
joints_dict['torso'] = ['LS', 'RS', 'LH', 'RH']
joints_dict['left_arm'] = ['LE', 'LW']
joints_dict['right_arm'] = ['RE', 'RW']
joints_dict['left_leg'] = ['LK', 'LA']
joints_dict['right_leg'] = ['RK', 'RA']

# Gaussian kernel radius for each group
joints_radius = {
    'head': 3,
    'torso': 3,
    'left_arm': 2,
    'right_arm': 2,
    'left_leg': 3,
    'right_leg': 3,
}

# PoseTrack keypoint order (17x3)
pose_keypoints = ['nose', 'Leye', 'Reye', 'LEar', 'REar', 
                  'LS', 'RS', 'LE', 'RE', 'LW', 'RW', 
                  'LH', 'RH', 'LK', 'RK', 'LA', 'RA']

# keypoints_dict: Mapping of joint names to their indices
keypoints_dict = {name: idx for idx, name in enumerate(pose_keypoints)}

# parts_info_per_strat: Variable for use in other modules
parts_info_per_strat = {
    "keypoints": (len(keypoints_dict), list(keypoints_dict.keys())),
    "keypoints_gaussian": (len(keypoints_dict), list(keypoints_dict.keys())),
    "joints": (len(joints_dict), list(joints_dict.keys())),
    "joints_gaussian": (len(joints_dict), list(joints_dict.keys())),
}


########################################
#         Utility Functions            #
########################################
def gkern(kernlen=21, std=None):
    if std is None:
        std = kernlen / 4
    gkern1d = gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

def rescale_keypoints(rf_keypoints, size, new_size):
    """
    Rescale keypoints to a new size.
    Args:
        rf_keypoints (np.ndarray): Keypoints in relative coordinates, shape (K, 3)
        size (tuple): Original image size (w, h)
        new_size (tuple): Target heatmap size (w, h)
    Returns:
        rescaled keypoints (np.ndarray): Shape (K, 3)
    """
    w, h = size
    new_w, new_h = new_size
    rf_keypoints = rf_keypoints.copy()
    rf_keypoints[:, 0] = rf_keypoints[:, 0] * new_w / w
    rf_keypoints[:, 1] = rf_keypoints[:, 1] * new_h / h
    return rf_keypoints

def kp_img_to_kp_bbox(kp_xyc_img, bbox_ltwh):
    """
    Convert keypoints from image coordinates to bounding box coordinates and filter out keypoints 
    outside the bounding box.
    Args:
        kp_xyc_img (np.ndarray): Keypoints in image coordinates, shape (K, 3)
        bbox_ltwh (tuple or np.ndarray): Bounding box as (l, t, w, h)
    Returns:
        kp_xyc_bbox (np.ndarray): Keypoints in bounding box coordinates, shape (K, 3)
    """
    l, t, w, h = bbox_ltwh
    kp_xyc_bbox = kp_xyc_img.copy()
    kp_xyc_bbox[:, 0] = kp_xyc_img[:, 0] - l
    kp_xyc_bbox[:, 1] = kp_xyc_img[:, 1] - t
    mask = (kp_xyc_bbox[:, 0] >= 0) & (kp_xyc_bbox[:, 0] < w) & \
           (kp_xyc_bbox[:, 1] >= 0) & (kp_xyc_bbox[:, 1] < h)
    kp_xyc_bbox[~mask] = 0
    return kp_xyc_bbox

########################################
#    KeypointsToMasks Class (Updated)   #
########################################
class KeypointsToMasks:
    def __init__(self, g_scale=11, vis_thresh=0.1, vis_continous=False):
        """
        Initialize KeypointsToMasks.
        Args:
            vis_thresh (float): Joint confidence threshold (e.g., 0.1)
        """
        self.g_scale = g_scale
        self.vis_thresh = vis_thresh
        self.vis_continous = vis_continous
        self.gaussian = None

    def __call__(self, kp_xyc, img_size, output_size):
        # kp_xyc: (K, 3) keypoint data, img_size and output_size are (w, h)
        kp_xyc_r = rescale_keypoints(kp_xyc, img_size, output_size)
        return self._compute_joints_gaussian_heatmaps(output_size, kp_xyc_r)

    def _compute_joints_gaussian_heatmaps(self, output_size, kp_xyc):
        """
        Generate heatmaps for each group in joints_dict by averaging or summing the heatmaps of the keypoints 
        in the group, then normalize. Returns a heatmap with as many channels as there are groups.
        """
        w, h = output_size
        num_groups = len(joints_dict)
        group_heatmaps = np.zeros((num_groups, h, w))
        count_maps = np.zeros((num_groups, h, w))
        kernel = self.get_gaussian_kernel(output_size)
        g_radius = kernel.shape[0] // 2

        for group_idx, (group_name, joint_names) in enumerate(joints_dict.items()):
            for joint_name in joint_names:
                idx = keypoints_dict[joint_name]
                kp = kp_xyc[idx]
                if kp[2] <= self.vis_thresh and not self.vis_continous:
                    continue
                kpx, kpy = int(kp[0]), int(kp[1])
                rt = max(0, kpy - g_radius)
                rb = min(h, kpy + g_radius + 1)
                rl = max(0, kpx - g_radius)
                rr = min(w, kpx + g_radius + 1)
                kernel_y_start = g_radius - (kpy - rt)
                kernel_y_end = g_radius + (rb - kpy)
                kernel_x_start = g_radius - (kpx - rl)
                kernel_x_end = g_radius + (rr - kpx)
                sub_kernel = kernel[kernel_y_start:kernel_y_end, kernel_x_start:kernel_x_end]
                if self.vis_continous:
                    sub_kernel = sub_kernel * kp[2]
                # Sum heatmaps
                group_heatmaps[group_idx, rt:rb, rl:rr] += sub_kernel
                count_maps[group_idx, rt:rb, rl:rr] += 1
            # Replace count_map==0 with 1 to avoid division by zero
            count_maps[group_idx][count_maps[group_idx]==0] = 1
            group_heatmaps[group_idx] = group_heatmaps[group_idx] / count_maps[group_idx]
        return group_heatmaps

    def get_gaussian_kernel(self, output_size):
        # Generate and normalize a new Gaussian kernel based on output_size
        if self.gaussian is None:
            w, h = output_size
            g_radius = int(w / self.g_scale)
            kernel_size = g_radius * 2 + 1
            kernel = gkern(kernel_size)
            kernel = kernel / np.sum(kernel)
            self.gaussian = kernel
        return self.gaussian

########################################
#         Main Processing Code         #
########################################

if __name__=="__main__":
    """
    A pipeline that processes the entire MARS dataset, reading each image and its corresponding keypoint 
    pose file (.pose) to generate 2D Gaussian heatmaps, then saves the grouped heatmaps as .npy files.
    """
    dataset_root = "/home/user/kim_js/ReID/dataset/MARS"
    phases = ["bbox_train", "bbox_test"]

    keypoint_root = os.path.join(dataset_root, "keypoints", "MARS")
    heatmap_root = "/home/user/data/heatmap"
    
    # Create KeypointsToMasks instance (using joints_gaussian mode)
    kp2mask = KeypointsToMasks(g_scale=11, vis_thresh=0.1, vis_continous=False)
    kp2mask.mode = "joints_gaussian"  # Set mode

    for phase in phases:
        phase_img_dir = os.path.join(dataset_root, phase)
        phase_heatmap_dir = os.path.join(heatmap_root, phase)
        os.makedirs(phase_heatmap_dir, exist_ok=True)
        
        for person_id in sorted(os.listdir(phase_img_dir)):
            person_img_dir = os.path.join(phase_img_dir, person_id)
            if not os.path.isdir(person_img_dir):
                continue
            
            person_kp_dir = os.path.join(keypoint_root, phase, person_id)
            if not os.path.exists(person_kp_dir):
                print(f"Keypoint folder not found: {person_kp_dir}")
                continue
            
            person_heatmap_dir = os.path.join(phase_heatmap_dir, person_id)
            os.makedirs(person_heatmap_dir, exist_ok=True)
            
            for img_file in sorted(os.listdir(person_img_dir)):
                if not img_file.endswith(".jpg"):
                    continue
                
                img_path = os.path.join(person_img_dir, img_file)
                # Corresponding keypoint file: Replace ".jpg" with ".pose"
                kp_file = img_file.replace(".jpg", ".pose")
                kp_path = os.path.join(person_kp_dir, kp_file)
                if not os.path.isfile(kp_path):
                    print(f"Keypoint file not found: {kp_path}")
                    continue
                
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to load image: {img_path}")
                    continue
                h_img, w_img, _ = img.shape

                with open(kp_path, "r") as f:
                    kp_data = json.load(f)  # 17x3 array: [[x,y,s], ..., [x17,y17,s17]]
                kp_array = np.array(kp_data, dtype=np.float32)

                # Generate grouped heatmap (number of channels = len(joints_dict))
                heatmap = kp2mask(kp_array, (w_img, h_img), (w_img, h_img))
                # heatmap shape: (num_groups, h_img, w_img)
                
                heatmap = np.ascontiguousarray(heatmap)
                npy_save_path = os.path.join(person_heatmap_dir, img_file.replace(".jpg", ".npy"))
                np.save(npy_save_path, heatmap)
                print(f"✅ Saved: {npy_save_path}")

    print("🔹 All heatmap generation completed!")
