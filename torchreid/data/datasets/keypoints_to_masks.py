import cv2
import numpy as np
from collections import OrderedDict
from torchreid.utils.imagetools import gkern
import os
from glob import glob
import json
import matplotlib.pyplot as plt

########################################
#       PoseTrack keypoint 그룹핑     #
########################################

# PoseTrack 관절 그룹핑 (각 그룹에 포함될 관절 이름)
joints_dict = OrderedDict()
joints_dict['head'] = ['nose', 'Leye', 'Reye', 'LEar', 'REar']
joints_dict['torso'] = ['LS', 'RS', 'LH', 'RH']
joints_dict['left_arm'] = ['LE', 'LW']
joints_dict['right_arm'] = ['RE', 'RW']
joints_dict['left_leg'] = ['LK', 'LA']
joints_dict['right_leg'] = ['RK', 'RA']

joints_radius = {
    'head': 3,
    'torso': 3,
    'left_arm': 2,
    'right_arm': 2,
    'left_leg': 3,
    'right_leg': 3,
}

# PoseTrack keypoint 순서 (17x3)
pose_keypoints = ['nose', 'Leye', 'Reye', 'LEar', 'REar', 
                  'LS', 'RS', 'LE', 'RE', 'LW', 'RW', 
                  'LH', 'RH', 'LK', 'RK', 'LA', 'RA']

# keypoints_dict: 각 관절 이름과 인덱스 매핑
keypoints_dict = {name: idx for idx, name in enumerate(pose_keypoints)}

# **추가**: parts_info_per_strat 변수 (다른 모듈에서 import될 수 있도록)
parts_info_per_strat = {
    "keypoints": (len(keypoints_dict), list(keypoints_dict.keys())),
    "keypoints_gaussian": (len(keypoints_dict), list(keypoints_dict.keys())),
    "joints": (len(joints_dict), list(joints_dict.keys())),
    "joints_gaussian": (len(joints_dict), list(joints_dict.keys())),
}

########################################
#         Utility Functions            #
########################################
def rescale_keypoints(rf_keypoints, size, new_size):
    """
    Rescale keypoints to new size.
    Args:
        rf_keypoints (np.ndarray): keypoints in relative coordinates, shape (K, 3)
        size (tuple): 원본 이미지 크기 (w, h)
        new_size (tuple): 목표 heatmap 크기 (w, h)
    Returns:
        rescaled keypoints (np.ndarray): shape (K, 3)
    """
    w, h = size
    new_w, new_h = new_size
    rf_keypoints = rf_keypoints.copy()
    rf_keypoints[:, 0] = rf_keypoints[:, 0] * new_w / w
    rf_keypoints[:, 1] = rf_keypoints[:, 1] * new_h / h
    return rf_keypoints

def kp_img_to_kp_bbox(kp_xyc_img, bbox_ltwh):
    """
    Convert keypoints in image coordinates to bounding box coordinates and filter out keypoints 
    that are outside the bounding box.
    Args:
        kp_xyc_img (np.ndarray): keypoints in image coordinates, shape (K, 3)
        bbox_tlwh (tuple or np.ndarray): bounding box as (l, t, w, h)
    Returns:
        kp_xyc_bbox (np.ndarray): keypoints in bounding box coordinates, shape (K, 3)
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
#        KeypointsToMasks Class        #
########################################
class KeypointsToMasks:
    def __init__(self, g_scale=11, vis_thresh=0.1, vis_continous=False):
        """
        vis_thresh: joint confidence threshold (예: 0.1)
        """
        self.g_scale = g_scale
        self.vis_thresh = vis_thresh
        self.vis_continous = vis_continous
        self.gaussian = None

    def __call__(self, kp_xyc, img_size, output_size):
        # kp_xyc: (K, 3) keypoint 데이터, img_size와 output_size는 (w, h)
        kp_xyc_r = rescale_keypoints(kp_xyc, img_size, output_size)
        return self._compute_keypoints_gaussian_heatmaps(output_size, kp_xyc_r)

    def _compute_keypoints_gaussian_heatmaps(self, output_size, kp_xyc):
        w, h = output_size
        keypoints_gaussian_heatmaps = np.zeros((len(kp_xyc), h, w))
        # 매 호출 시, output_size에 맞게 새 Gaussian kernel 계산을 위해 self.gaussian를 초기화
        self.gaussian = None
        kernel = self.get_gaussian_kernel(output_size)
        g_radius = kernel.shape[0] // 2
        for i, kp in enumerate(kp_xyc):
            if kp[2] <= self.vis_thresh and not self.vis_continous:
                continue
            kpx, kpy = int(kp[0]), int(kp[1])
            # 올바른 영역 계산:
            rt = max(0, kpy - g_radius)
            rb = min(h, kpy + g_radius + 1)
            rl = max(0, kpx - g_radius)
            rr = min(w, kpx + g_radius + 1)
            # kernel에서 추출할 인덱스:
            kernel_y_start = g_radius - (kpy - rt)
            kernel_y_end = g_radius + (rb - kpy)
            kernel_x_start = g_radius - (kpx - rl)
            kernel_x_end = g_radius + (rr - kpx)
            sub_kernel = kernel[kernel_y_start:kernel_y_end, kernel_x_start:kernel_x_end]
            if self.vis_continous:
                sub_kernel = sub_kernel * kp[2]
            keypoints_gaussian_heatmaps[i, rt:rb, rl:rr] = sub_kernel
        return keypoints_gaussian_heatmaps

    def get_gaussian_kernel(self, output_size):
        # output_size를 기반으로 새 Gaussian kernel 생성 및 정규화
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
    전체 MARS 데이터셋의 이미지를 대상으로, 
    각 이미지와 대응하는 keypoint pose 파일(.pose)을 읽어 2D Gaussian heatmap을 생성한 후,
    npy 파일로 저장하는 파이프라인.
    
    - 이미지 경로 예시: 
      /home/user/kim_js/ReID/dataset/MARS/bbox_train/0001/0001C1T0001F001.jpg
    - Keypoint 파일 경로 예시: 
      /home/user/kim_js/ReID/dataset/MARS/keypoints/MARS/bbox_train/0001/0001C1T0001F001.pose
    - 저장 경로 예시: 
      /home/user/kim_js/ReID/dataset/MARS/heatmaps/bbox_train/0001/0001C1T0001F001.npy
    """
    dataset_root = "/home/user/kim_js/ReID/dataset/MARS"
    phases = ["bbox_train", "bbox_test"]

    keypoint_root = os.path.join(dataset_root, "keypoints", "MARS")
    heatmap_root = "/home/user/data/heatmap"
    
    # KeypointsToMasks 인스턴스 생성 (PoseTrack 기준, vis_thresh 0.1 적용)
    kp2mask = KeypointsToMasks(g_scale=11, vis_thresh=0.1, vis_continous=False)
    kp2mask.mode = "keypoints_gaussian"  # mode 지정

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
                print(f"Keypoint 폴더 없음: {person_kp_dir}")
                continue
            
            person_heatmap_dir = os.path.join(phase_heatmap_dir, person_id)
            os.makedirs(person_heatmap_dir, exist_ok=True)
            
            for img_file in sorted(os.listdir(person_img_dir)):
                if not img_file.endswith(".jpg"):
                    continue
                
                img_path = os.path.join(person_img_dir, img_file)
                # 대응 keypoint 파일: 이미지 파일명에서 ".jpg"를 ".pose"로 변경
                kp_file = img_file.replace(".jpg", ".pose")
                kp_path = os.path.join(person_kp_dir, kp_file)
                if not os.path.isfile(kp_path):
                    print(f"Keypoint 파일 없음: {kp_path}")
                    continue
                
                img = cv2.imread(img_path)
                if img is None:
                    print(f"이미지 로드 실패: {img_path}")
                    continue
                h_img, w_img, _ = img.shape

                with open(kp_path, "r") as f:
                    kp_data = json.load(f)  # 17x3 배열: [[x,y,s], ..., [x17,y17,s17]]
                kp_array = np.array(kp_data, dtype=np.float32)

                heatmap = kp2mask(kp_array, (w_img, h_img), (w_img, h_img))
                # heatmap shape: (17, h_img, w_img)
                
                # npy 파일로 저장 전에 연속 배열로 변환
                heatmap = np.ascontiguousarray(heatmap)

                npy_save_path = os.path.join(person_heatmap_dir, img_file.replace(".jpg", ".npy"))
                np.save(npy_save_path, heatmap)
                print(f"✅ 저장 완료: {npy_save_path}")

    print("🔹 모든 heatmap 생성 완료!")
