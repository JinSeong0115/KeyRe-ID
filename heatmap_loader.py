import os
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import sys
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
from PIL import Image
from Dataloader import VideoDataset, VideoDataset_inderase
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Datasets"))
from MARS_dataset import Mars
from utility import RandomIdentitySampler

__factory = {
    'Mars':Mars,
}

# Heatmap Preprocessing transform 
class HeatmapResize(object):
    def __init__(self, size):
        self.size = size  # (height, width)
    def __call__(self, tensor):
        tensor = tensor.unsqueeze(0)  # (1, C, H, W)
        resized = F.interpolate(tensor, size=self.size, mode='bilinear', align_corners=False)  # (1, C, new_H, new_W)
        return resized.squeeze(0)  # (C, new_H, new_W)

def Min_Max_Scaling(tensor, eps=1e-6):
    # 각 채널별 최소, 최대값 계산 (keepdim=True로 broadcasting)
    min_val = tensor.amin(dim=(1,2), keepdim=True)
    max_val = tensor.amax(dim=(1,2), keepdim=True)
    scaled = (tensor - min_val) / (max_val - min_val + eps)
    return scaled

class CustomHeatmapTransform(object):
    def __init__(self, size):
        self.size = size  # (height, width)
        self.resize = HeatmapResize(size)
    def __call__(self, tensor):
        # tensor shape: (C, H, W), 여기서 C=6
        tensor = self.resize(tensor) 
        tensor = Min_Max_Scaling(tensor)  # 각 채널별로 min-max scaling 수행 (값 범위 [0, 1])
        tensor = T.Normalize(mean=[0.5]*6, std=[0.5]*6)(tensor)
        return tensor

def Heatmap_collate_fn(batch):
    imgs, heatmaps, pids, camids, erasing_labels = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), torch.stack(heatmaps, dim=0), pids, camids, torch.stack(erasing_labels, dim=0)

def val_collate_fn(batch):
    imgs, pids, camids, img_paths = zip(*batch)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    
    return torch.stack(imgs, dim=0), pids, camids_batch, img_paths


# heatmap_dataloader 함수
def heatmap_dataloader(Dataset_name):
    val_transforms = T.Compose([
        T.Resize([256, 128], interpolation=InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
        
    dataset = __factory[Dataset_name]()
    train_data = dataset.train
    query_data = dataset.query
    gallery_data = dataset.gallery
    
    Heatmap_train_set = Heatmap_Dataset_inderase(
        dataset = train_data,
        seq_len = 4,
        heatmap_transform = CustomHeatmapTransform([256, 128]),
        heatmap_root = "/home/user/data/heatmap/bbox_train"
    )
    train_loader = DataLoader(
        Heatmap_train_set,
        batch_size = 16,
        sampler = RandomIdentitySampler(train_data, 16, 4),
        num_workers = 2,
        collate_fn = Heatmap_collate_fn 
    )
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    num_train = dataset.num_train_vids
    
    q_val_set = Heatmap_Dataset(
        dataset = query_data,
        seq_len = 4,
        transform = val_transforms,
        heatmap_root = "/home/user/data/heatmap/bbox_test",
        heatmap_transform = CustomHeatmapTransform([256, 128])
    )
    g_val_set = Heatmap_Dataset(
        dataset = gallery_data,
        seq_len = 4,
        transform = val_transforms,
        heatmap_root = "/home/user/data/heatmap/bbox_test",
        heatmap_transform = CustomHeatmapTransform([256, 128])
    )
    
    return train_loader, len(query_data), num_classes, cam_num, num_train, q_val_set, g_val_set


# Heatmap_Dataset (Test)
# VideoDataset --> imgs_array, pid, camids, img_paths
class Heatmap_Dataset(VideoDataset): 
    """
    부모 클래스(VideoDataset)의 __getitem__을 호출해서 
    imgs와 img_paths를 받아온 후, 
    imgs의 클립 구조에 맞춰 heatmap을 로드하여 반환하는 클래스입니다.

    반환: (imgs, heatmaps, pid, camid, img_paths)
    """
    def __init__(self, heatmap_transform, heatmap_root, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.heatmap_transform = heatmap_transform
        self.heatmap_root = heatmap_root

    def __getitem__(self, index):
        imgs, pid, camids, img_paths = super().__getitem__(index)
        
        # imgs는 [clips, seq_len, 3, H, W] 형태 
        clips, seq_len = imgs.shape[0], imgs.shape[1]
        
        heatmap_list = []
        for i in range(clips):
            clip_heatmaps = []
            # 각 클립에 해당하는 img_paths의 인덱스 계산
            start_idx = i*seq_len
            end_idx = min(start_idx+seq_len, len(img_paths))
            for frame_path in img_paths[start_idx:end_idx]:
                file_name = os.path.basename(frame_path).replace(".jpg", ".npy")
                person_id = os.path.basename(os.path.dirname(frame_path))
                heatmap_file = os.path.join(self.heatmap_root, person_id, file_name)

                if not os.path.exists(heatmap_file):
                    print(f"⚠️ Heatmap 파일 없음: {heatmap_file}")
                    raise FileNotFoundError(f"Missing heatmap file: {heatmap_file}")
                else:
                    heatmap_np = np.load(heatmap_file)  # shape: (6, H, W)
                    heatmap = torch.tensor(heatmap_np, dtype=torch.float32)

                # 채널별 정규화
                for c in range(heatmap.shape[0]):
                    max_val = heatmap[c].max()
                    if max_val > 0:
                        heatmap[c] = heatmap[c] / max_val
                    else:
                        heatmap[c] = torch.zeros_like(heatmap[c])  # 명시적 0 설정

                # transform 적용
                if self.heatmap_transform is not None:
                    heatmap = self.heatmap_transform(heatmap)
                    if heatmap.shape[1:] != self.heatmap_transform.size:
                        heatmap = T.Resize(self.heatmap_transform.size)(heatmap)
                
                clip_heatmaps.append(heatmap)
            
            # 클립 내 프레임 수가 seq_len보다 적을 경우 패딩
            pad_size = seq_len - len(clip_heatmaps)
            if pad_size > 0:
                padding = torch.zeros(pad_size, *clip_heatmaps[0].shape)
                clip_heatmaps = torch.cat([torch.stack(clip_heatmaps), padding], dim=0)
            else:
                clip_heatmaps = torch.stack(clip_heatmaps)
            
            heatmap_list.append(clip_heatmaps)
        
        heatmaps = torch.stack(heatmap_list, dim=0)  # [clips, seq_len, 6, H, W]

        return imgs, heatmaps, pid, camids


# Heatmap_Dataset_inderase (Train)
# VideoDataset_inderase --> imgs, pid, camids, labels, selected_img_paths, erased_regions, transform_params_list
class Heatmap_Dataset_inderase(VideoDataset_inderase):
    def __init__(self, heatmap_transform, heatmap_root, dataset, *args, **kwargs):
        super().__init__(dataset=dataset, *args, **kwargs)
        self.heatmap_transform = heatmap_transform  # 히트맵 변환 함수
        self.heatmap_root = heatmap_root  # 히트맵 파일이 저장된 루트 경로
        self.heatmap_cache = {}  # 히트맵 캐시로 성능 최적화

    def load_heatmap(self, img_path):
        if img_path not in self.heatmap_cache:
            file_name = os.path.basename(img_path).replace(".jpg", ".npy")
            person_id = os.path.basename(os.path.dirname(img_path))
            heatmap_file = os.path.join(self.heatmap_root, person_id, file_name)

            # 파일이 없으면 에러
            if not os.path.exists(heatmap_file):
                raise FileNotFoundError(f"히트맵 파일 없음: {heatmap_file}")

            # 히트맵 로드 및 변환
            heatmap = np.load(heatmap_file)
            heatmap = torch.tensor(heatmap, dtype=torch.float32)
            self.heatmap_cache[img_path] = heatmap

        return self.heatmap_cache[img_path]

    def __getitem__(self, index):
        imgs, pid, camids, labels, selected_img_paths, erased_regions, transform_params_list = super().__getitem__(index)

        # selected_img_paths로 히트맵 로드
        heatmap_list = []
        for label, img_path, erased_region, transform_params in zip(labels, selected_img_paths, erased_regions, transform_params_list):
            heatmap = self.load_heatmap(img_path)
            heatmap = self.heatmap_transform(heatmap)
            # 이미지와 동일한 변환 적용
            if transform_params['flipped']:
                heatmap = torch.flip(heatmap, dims=[-1])
            heatmap = F.pad(heatmap, (10, 10, 10, 10), mode='constant', value=0)  # 패딩
            if transform_params['crop_params'] is not None:
                heatmap = T.functional.crop(heatmap, *transform_params['crop_params'])
            if label == 1:  # erasing 적용된 경우
                x, y, w, h = erased_region
                heatmap_h, heatmap_w = heatmap.shape[1], heatmap.shape[2]
                y_end = min(y + h, heatmap_h)
                x_end = min(x + w, heatmap_w)
                heatmap[:, y:y_end, x:x_end] = 0

            heatmap = heatmap.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
            heatmap_list.append(heatmap)

        heatmaps = torch.cat(heatmap_list, dim=0)  # (seq_len, C, H, W)
        return imgs, heatmaps, pid, camids, labels
    



# def custom_collate_fn(batch):
#     """
#     각 항목은 (imgs, heatmaps, pid, camid, img_paths) 형태로 반환된다고 가정합니다.
#     imgs: [num_clips, seq_len, 3, H, W]
#     heatmaps: [num_clips, seq_len, 6, H, W]
#     """
#     # 각 샘플의 클립 수 중 최대값 계산
#     max_clips = max(item[0].shape[0] for item in batch)
    
#     imgs_list, heatmaps_list = [], []
#     pids_list, camids_list, img_paths_list = [], [], []
#     clip_masks = [] 

#     for imgs, heatmaps, pid, camid, img_paths in batch:
#         num_clips = imgs.shape[0]
#         pad_clips = max_clips - num_clips

#         # imgs와 heatmaps에 대해 패딩 적용
#         if pad_clips > 0:
#             # imgs: 패딩할 텐서는 동일한 타입과 디바이스로 생성
#             pad_shape_imgs = (pad_clips, *imgs.shape[1:])
#             pad_tensor_imgs = imgs.new_zeros(pad_shape_imgs)
#             imgs_padded = torch.cat([imgs, pad_tensor_imgs], dim=0)

#             pad_shape_heat = (pad_clips, *heatmaps.shape[1:])
#             pad_tensor_heat = heatmaps.new_zeros(pad_shape_heat)
#             heatmaps_padded = torch.cat([heatmaps, pad_tensor_heat], dim=0)
            
#             # 유효한 클립은 1, 패딩은 0인 마스크 생성
#             mask = torch.cat([torch.ones(num_clips), torch.zeros(pad_clips)])
#         else:
#             imgs_padded = imgs
#             heatmaps_padded = heatmaps
#             mask = torch.ones(num_clips)
        
#         imgs_list.append(imgs_padded)
#         heatmaps_list.append(heatmaps_padded)
#         clip_masks.append(mask)
#         pids_list.append(pid)
#         camids_list.append(camid)
#         img_paths_list.append(img_paths)
    
#     # 배치 차원으로 스택 (결과 shape: [B, max_clips, ...])
#     imgs_batch = torch.stack(imgs_list, dim=0)
#     heatmaps_batch = torch.stack(heatmaps_list, dim=0)
#     masks_batch = torch.stack(clip_masks, dim=0)  # [B, max_clips]
    
#     # pids, camids, img_paths는 리스트로 그대로 반환하거나 필요에 맞게 처리
#     return imgs_batch, heatmaps_batch, pids_list, camids_list, img_paths_list, masks_batch# Update 2025. 04. 12. (토) 23:39:10 KST
