import os
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import sys
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F

from Dataloader import VideoDataset, VideoDataset_inderase
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Datasets"))
from MARS_dataset import Mars
from utility import RandomIdentitySampler


# Heatmap 전처리 transform 
class HeatmapResize(object):
    def __init__(self, size):
        self.size = size  # (height, width)
    def __call__(self, tensor):
        tensor = tensor.unsqueeze(0)  # (1, C, H, W)
        resized = F.interpolate(tensor, size=self.size, mode='bilinear', align_corners=True)
        return resized.squeeze(0)  # (C, new_H, new_W)

def Min_Max_Scaling(tensor, eps=1e-6):
    # tensor: (C, H, W)
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
        tensor = self.resize(tensor)  # torch  g연산으로 리사이즈
        tensor = Min_Max_Scaling(tensor)  # 각 채널별로 min-max scaling 수행 (값 범위 [0, 1])
        tensor = tensor * 2 - 1  # [-1, 1]로 조정
        return tensor

# Collate 함수 (Heatmap_Dataset)
def Heatmap_collate_fn(batch):
    if len(batch[0]) == 6:  # train
        imgs, heatmaps, pids, camids, labels, img_paths = zip(*batch)
        imgs = torch.stack(imgs, dim=0)
        heatmaps = torch.stack(heatmaps, dim=0)
        labels = torch.stack(labels, dim=0)
        return imgs, heatmaps, pids, camids, labels, img_paths
    elif len(batch[0]) == 5:  # test
        imgs, heatmaps, pids, camids, img_paths = zip(*batch)
        imgs = torch.stack(imgs, dim=0)
        heatmaps = torch.stack(heatmaps, dim=0)
        
        return imgs, heatmaps, pids, camids, img_paths
    else:
        raise ValueError("Unexpected number of elements in batch: {}".format(len(batch[0])))

def custom_collate_fn(batch):
    """
    각 항목은 (imgs, heatmaps, pid, target_cam, img_paths) 형태로 반환된다고 가정합니다.
    imgs: [num_clips, seq_len, 3, H, W]
    heatmaps: [num_clips, seq_len, 6, H, W]
    """
    # 각 샘플의 클립 수 중 최대값 계산
    max_clips = max(item[0].shape[0] for item in batch)
    
    imgs_list, heatmaps_list = [], []
    pids_list, camids_list, img_paths_list = [], [], []
    clip_masks = []  # 유효한 클립 위치(1)와 패딩된 클립(0)을 표시하는 마스크

    for imgs, heatmaps, pid, target_cam, img_paths in batch:
        num_clips = imgs.shape[0]
        pad_clips = max_clips - num_clips

        # imgs와 heatmaps에 대해 패딩 적용
        if pad_clips > 0:
            # imgs: 패딩할 텐서는 동일한 타입과 디바이스로 생성
            pad_shape_imgs = (pad_clips, *imgs.shape[1:])
            pad_tensor_imgs = imgs.new_zeros(pad_shape_imgs)
            imgs_padded = torch.cat([imgs, pad_tensor_imgs], dim=0)

            pad_shape_heat = (pad_clips, *heatmaps.shape[1:])
            pad_tensor_heat = heatmaps.new_zeros(pad_shape_heat)
            heatmaps_padded = torch.cat([heatmaps, pad_tensor_heat], dim=0)
            
            # 유효한 클립은 1, 패딩은 0인 마스크 생성
            mask = torch.cat([torch.ones(num_clips), torch.zeros(pad_clips)])
        else:
            imgs_padded = imgs
            heatmaps_padded = heatmaps
            mask = torch.ones(num_clips)
        
        imgs_list.append(imgs_padded)
        heatmaps_list.append(heatmaps_padded)
        clip_masks.append(mask)
        pids_list.append(pid)
        camids_list.append(target_cam)
        img_paths_list.append(img_paths)
    
    # 배치 차원으로 스택 (결과 shape: [B, max_clips, ...])
    imgs_batch = torch.stack(imgs_list, dim=0)
    heatmaps_batch = torch.stack(heatmaps_list, dim=0)
    masks_batch = torch.stack(clip_masks, dim=0)  # [B, max_clips]
    
    # pids, camids, img_paths는 리스트로 그대로 반환하거나 필요에 맞게 처리
    return imgs_batch, heatmaps_batch, pids_list, camids_list, img_paths_list, masks_batch


# heatmap_dataloader 함수
def heatmap_dataloader(Dataset_name):
    train_transforms = T.Compose([
        T.Resize([256, 128], interpolation=InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    val_transforms = T.Compose([
        T.Resize([256, 128], interpolation=InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
        
    dataset = Mars()
    train_data = dataset.train
    query_data = dataset.query
    gallery_data = dataset.gallery
    
    Heatmap_train_set = Heatmap_Dataset_inderase(
        dataset=train_data,
        seq_len=4,
        sample='intelligent',
        transform=train_transforms,
        heatmap_transform=CustomHeatmapTransform([256, 128]),
        heatmap_root="/home/user/data/heatmap/bbox_train"
    )
    train_loader = DataLoader(
        Heatmap_train_set,
        batch_size=16,
        sampler=RandomIdentitySampler(train_data, 16, 4),
        num_workers=2,
        collate_fn=Heatmap_collate_fn
    )
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    num_train = dataset.num_train_vids
    
    q_val_set = Heatmap_Dataset(
        dataset=query_data,
        seq_len=4,
        sample='dense',
        transform=val_transforms,
        heatmap_root="/home/user/data/heatmap/bbox_test",
        heatmap_transform=CustomHeatmapTransform([256, 128])
    )
    g_val_set = Heatmap_Dataset(
        dataset=gallery_data,
        seq_len=4,
        sample='dense',
        transform=val_transforms,
        heatmap_root="/home/user/data/heatmap/bbox_test",
        heatmap_transform=CustomHeatmapTransform([256, 128])
    )
    
    return train_loader, len(query_data), num_classes, cam_num, num_train, q_val_set, g_val_set


# Heatmap_Dataset (Test)
class Heatmap_Dataset(VideoDataset):
    """
    부모 클래스(VideoDataset)의 __getitem__을 호출해서 
    imgs와 img_paths를 받아온 후, 
    imgs의 클립 구조에 맞춰 heatmap을 로드하여 반환하는 클래스입니다.

    반환: (imgs, heatmaps, pid, target_cam, img_paths)
    """
    def __init__(self, heatmap_transform, heatmap_root, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.heatmap_transform = heatmap_transform
        self.heatmap_root = heatmap_root

    def __getitem__(self, index):
        # 부모 클래스에서 이미지와 경로를 가져옴
        result = super().__getitem__(index)
        imgs, pid, target_cam, img_paths = result

        # target_cam이 리스트일 경우 단일 값으로 변환
        if isinstance(target_cam, list):
            target_cam = target_cam[0]  # 첫 번째 카메라 ID 사용 (모든 프레임 동일 가정)
        if isinstance(pid, list):
            pid = pid[0]  # 첫 번째 카메라 ID 사용 (모든 프레임 동일 가정)
        
        # imgs는 [clips, seq_len, 3, H, W] 형태 
        clips, seq_len = imgs.shape[0], imgs.shape[1]
        # target_cam = [target_cam for _ in range(clips * seq_len)]
        
        heatmap_list = []
        for i in range(clips):
            clip_heatmaps = []
            # 각 클립에 해당하는 img_paths의 인덱스 계산
            start_idx = i * seq_len
            end_idx = min(start_idx + seq_len, len(img_paths))
            for frame_path in img_paths[start_idx:end_idx]:
                file_name = os.path.basename(frame_path).replace(".jpg", ".npy")
                person_id = os.path.basename(os.path.dirname(frame_path))
                heatmap_file = os.path.join(self.heatmap_root, person_id, file_name)

                if not os.path.exists(heatmap_file):
                    print(f"⚠️ Heatmap 파일 없음: {heatmap_file}")
                    heatmap = torch.zeros(
                        (6, self.heatmap_transform.size[0], self.heatmap_transform.size[1])
                    )
                else:
                    heatmap_np = np.load(heatmap_file)  # shape: (6, H, W)
                    heatmap = torch.tensor(heatmap_np, dtype=torch.float32)

                # 채널별 정규화
                for c in range(heatmap.shape[0]):
                    max_val = heatmap[c].max()
                    if max_val > 0:
                        heatmap[c] = heatmap[c] / max_val

                # transform 적용
                if self.heatmap_transform is not None:
                    heatmap = self.heatmap_transform(heatmap)  # [6, H, W]
                
                clip_heatmaps.append(heatmap)
            
            # 클립 내 프레임 수가 seq_len보다 적을 경우 패딩
            while len(clip_heatmaps) < seq_len:
                clip_heatmaps.append(torch.zeros_like(clip_heatmaps[0]))
            
            heatmap_stack = torch.stack(clip_heatmaps, dim=0)  # [seq_len, 6, H, W]
            heatmap_list.append(heatmap_stack)
        
        heatmaps = torch.stack(heatmap_list, dim=0)  # [clips, seq_len, 6, H, W]

        return imgs, heatmaps, pid, target_cam, img_paths


# Heatmap_Dataset_inderase (Train)
class Heatmap_Dataset_inderase(VideoDataset_inderase):
    """
    VideoDataset_inderase를 상속받아, 이미지 로드 후 image path를 기반으로 heatmap 파일(.npy)을 추가로 로드합니다.
    반환: (imgs, heatmaps, pid, target_cam, labels, selected_img_paths)
    """
    def __init__(self, heatmap_transform, heatmap_root, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.heatmap_transform = heatmap_transform
        self.heatmap_root = heatmap_root

    def __getitem__(self, index):
        imgs, pid, target_cam, labels, selected_img_paths = super().__getitem__(index)
        heatmap_list = []
        for img_path in selected_img_paths:
            file_name = os.path.basename(img_path).replace(".jpg", ".npy")
            person_id = os.path.basename(os.path.dirname(img_path))
            heatmap_file = os.path.join(self.heatmap_root, person_id, file_name)
            if not os.path.exists(heatmap_file):
                print(f"⚠️ Heatmap 파일 없음: {heatmap_file}")
            else:
                heatmap = np.load(heatmap_file)  # (6, H, W)
                heatmap = torch.tensor(heatmap, dtype=torch.float32)
            # 각 채널별로 max값으로 나누어 [0, 1]로 정규화 (기존 방식과 동일한 역할)
            for c in range(heatmap.shape[0]):
                max_val = heatmap[c].max()
                if max_val > 0:
                    heatmap[c] = heatmap[c] / max_val
            if self.heatmap_transform is not None:
                heatmap = self.heatmap_transform(heatmap)
            heatmap_list.append(heatmap)
        heatmaps = torch.stack(heatmap_list, dim=0)  # [seq_len, 6, H, W]
        return imgs, heatmaps, pid, target_cam, labels, selected_img_paths


if __name__ == "__main__":
    # heatmap_dataloader를 통해 DataLoader와 데이터셋 생성
    train_loader, num_query, num_classes, cam_num, num_train, q_val_set, g_val_set = heatmap_dataloader("Mars")
    
    # Train DataLoader의 collate_fn 확인
    if train_loader.collate_fn is not None:
        print("Train DataLoader collate_fn:", train_loader.collate_fn.__name__)
    else:
        print("Train DataLoader에 collate_fn이 지정되지 않았습니다.")
    
    # Train 배치의 항목 개수 및 상세 정보 확인
    print("=== Train Loader 확인 ===")
    for batch in train_loader:
        print("Train 배치 항목 개수:", len(batch))
        for i, item in enumerate(batch):
            if torch.is_tensor(item):
                print(f"항목 {i}의 shape: {item.shape}")
            else:
                print(f"항목 {i}의 타입: {type(item)}")
        break
    
    # Query 샘플의 항목 개수 및 상세 정보 확인
    print("=== Query Set 확인 ===")
    q_sample = q_val_set[0]
    print("Query 샘플 항목 개수:", len(q_sample))
    for i, item in enumerate(q_sample):
        if torch.is_tensor(item):
            print(f"항목 {i}의 shape: {item.shape}")
        else:
            print(f"항목 {i}의 타입: {type(item)}")
    
    # Gallery 샘플의 항목 개수 및 상세 정보 확인
    print("=== Gallery Set 확인 ===")
    g_sample = g_val_set[0]
    print("Gallery 샘플 항목 개수:", len(g_sample))
    for i, item in enumerate(g_sample):
        if torch.is_tensor(item):
            print(f"항목 {i}의 shape: {item.shape}")
        else:
            print(f"항목 {i}의 타입: {type(item)}")


# if __name__ == "__main__":
#     train_loader, num_query, num_classes, cam_num, num_train, q_val_set, g_val_set = heatmap_dataloader("Mars")
    
#     print("=== Train (Image & Heatmap) Loader ===")
#     for batch in train_loader:
#         imgs, heatmaps, pids, camids, labels, img_paths = batch
#         print("Image batch shape:", imgs.shape)
#         print("Heatmap batch shape:", heatmaps.shape)
#         print("Labels shape:", labels.shape)
#         print("예시 이미지 경로 (Train sample):", img_paths[0])
#         imgs_min = imgs.min().item()
#         imgs_max = imgs.max().item()
#         imgs_mean = imgs.mean().item()
#         print(f"Train Images - min: {imgs_min:.4f}, max: {imgs_max:.4f}, mean: {imgs_mean:.4f}")
#         heatmaps_min = heatmaps.min().item()
#         heatmaps_max = heatmaps.max().item()
#         heatmaps_mean = heatmaps.mean().item()
#         print(f"Train Heatmaps - min: {heatmaps_min:.4f}, max: {heatmaps_max:.4f}, mean: {heatmaps_mean:.4f}")
#         break

#     print("=== Query Sample ===")
#     q_sample = q_val_set[0]
#     imgs, heatmaps, pid, target_cam, img_paths = q_sample
#     print("Query image shape:", imgs.shape)
#     print("Query heatmap shape:", heatmaps.shape)
#     print("Query PID:", pid)
#     print("Query Camera ID:", target_cam)
#     imgs_min = imgs.min().item()
#     imgs_max = imgs.max().item()
#     imgs_mean = imgs.mean().item()
#     print(f"Query Images - min: {imgs_min:.4f}, max: {imgs_max:.4f}, mean: {imgs_mean:.4f}")
#     heatmaps_min = heatmaps.min().item()
#     heatmaps_max = heatmaps.max().item()
#     heatmaps_mean = heatmaps.mean().item()
#     print(f"Query Heatmaps - min: {heatmaps_min:.4f}, max: {heatmaps_max:.4f}, mean: {heatmaps_mean:.4f}")

#     print("=== Gallery Sample ===")
#     g_sample = g_val_set[0]
#     imgs, heatmaps, pid, target_cam, img_paths = g_sample
#     print("Gallery image shape:", imgs.shape)
#     print("Gallery heatmap shape:", heatmaps.shape)
#     print("Gallery PID:", pid)
#     print("Gallery Camera ID:", target_cam)
#     imgs_min = imgs.min().item()
#     imgs_max = imgs.max().item()
#     imgs_mean = imgs.mean().item()
#     print(f"Gallery Images - min: {imgs_min:.4f}, max: {imgs_max:.4f}, mean: {imgs_mean:.4f}")
#     heatmaps_min = heatmaps.min().item()
#     heatmaps_max = heatmaps.max().item()
#     heatmaps_mean = heatmaps.mean().item()
#     print(f"Gallery Heatmaps - min: {heatmaps_min:.4f}, max: {heatmaps_max:.4f}, mean: {heatmaps_mean:.4f}")
