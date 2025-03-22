import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import sys
import random
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F

# Dataloader.py에서 사용한 VideoDataset, VideoDataset_inderase 임포트
from Dataloader import VideoDataset, VideoDataset_inderase

# MARS 데이터셋 경로를 맞추기 위해 Datasets 폴더 경로 추가
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Datasets"))
from MARS_dataset import Mars


# Custom Transform for Heatmap
class HeatmapResize(object):
    """
    torch.Tensor (C, H, W)를 입력받아, 지정된 크기로 리사이즈합니다.
    """
    def __init__(self, size):
        self.size = size  # (height, width)
    def __call__(self, tensor):
        # tensor shape: (C, H, W)
        tensor = tensor.unsqueeze(0)  # (1, C, H, W)
        resized = F.interpolate(tensor, size=self.size, mode='bilinear', align_corners=False)
        return resized.squeeze(0)  # (C, new_H, new_W)

# heatmap에 적용할 transform: 리사이즈 후 정규화.
# heatmap 채널 수는 17이므로, mean과 std의 길이도 17이어야 합니다.
heatmap_transforms = T.Compose([
    HeatmapResize([256, 128]),
    T.Normalize(mean=[0.5]*17, std=[0.5]*17)
])

# HeatmapDataset_inderase (Train)
class HeatmapDataset_inderase(Dataset):
    """
    학습용 heatmap 데이터셋: 
    이미지 데이터셋(VideoDataset_inderase)에서 선택한 image path 리스트를 기반으로,
    bbox_train 폴더에서 heatmap npy 파일을 불러옵니다.
    
    반환: (heatmaps, pid, camid, selected_img_paths)
           heatmaps: [seq_len, 17, H, W] 형태의 텐서
    """
    def __init__(self, dataset, seq_len=4, sample='intelligent', transform=None, heatmap_root=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform  # heatmap_transforms를 전달
        self.heatmap_root = heatmap_root

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # 원본 이미지 데이터셋의 샘플: (img_paths, pid, camid)
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)
        if self.sample == 'intelligent':
            indices = []
            each = max(num // self.seq_len, 1)
            for i in range(self.seq_len):
                if i != self.seq_len - 1:
                    indices.append(random.randint(min(i*each, num-1), min((i+1)*each-1, num-1)))
                else:
                    indices.append(random.randint(min(i*each, num-1), num-1))
            indices = np.array(indices)
        else:
            indices = np.linspace(0, num-1, self.seq_len, dtype=int)
        
        # 선택된 image path 리스트 추출
        selected_img_paths = [img_paths[int(i)] for i in indices]
        
        # 각 선택된 image path에 해당하는 heatmap 로드 (bbox_train)
        heatmap_list = []
        heatmaps_path = []
        for img_path in selected_img_paths:
            file_name = os.path.basename(img_path).replace(".jpg", ".npy")
            person_id = file_name[:4]
            heatmap_file = os.path.join(self.heatmap_root, person_id, file_name)
            if not os.path.exists(heatmap_file):
                print(f"⚠️ Heatmap 파일 없음: {heatmap_file}")
            heatmaps_path.append(heatmap_file)
            heatmap = np.load(heatmap_file)  # (17, H, W)
            heatmap = torch.tensor(heatmap, dtype=torch.float32)
            if self.transform:
                heatmap = self.transform(heatmap)
            heatmap_list.append(heatmap)
        heatmaps = torch.stack(heatmap_list, dim=0)  # [seq_len, 17, H, W]
        return heatmaps, pid, camid, selected_img_paths, heatmaps_path

# HeatmapDataset (Query, Gallery)
class HeatmapDataset(Dataset):
    """
    평가용 heatmap 데이터셋:
    이미지 데이터셋(VideoDataset)에서 반환한 image path 리스트를 기반으로,
    bbox_test 폴더에서 heatmap npy 파일을 로드합니다.
    
    반환: (heatmaps, pid, camid, selected_img_paths)
    """
    def __init__(self, dataset, seq_len=4, sample='dense', transform=None, heatmap_root=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform  # heatmap_transforms
        self.heatmap_root = heatmap_root

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)
        if self.sample == 'dense':
            cur_index = 0
            frame_indices = [i for i in range(num)]
            indices_list = []
            while num - cur_index >= self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
                cur_index += self.seq_len
            if cur_index < num:
                last_seq = frame_indices[cur_index:]
                while len(last_seq) < self.seq_len:
                    last_seq.append(last_seq[-1])
                indices_list.append(last_seq)
            # 평가용에서는 첫 번째 클립만 사용 (또는 필요에 따라 여러 클립 선택)
            selected_indices = indices_list[0]
        else:
            selected_indices = np.linspace(0, num-1, self.seq_len, dtype=int)
        selected_img_paths = [img_paths[int(i)] for i in selected_indices]
        
        heatmap_list = []
        for img_path in selected_img_paths:
            file_name = os.path.basename(img_path).replace(".jpg", ".npy")
            person_id = file_name[:4]
            heatmap_file = os.path.join(self.heatmap_root, person_id, file_name)
            if not os.path.exists(heatmap_file):
                print(f"⚠️ Heatmap 파일 없음: {heatmap_file}")
            heatmap = np.load(heatmap_file)
            heatmap = torch.tensor(heatmap, dtype=torch.float32)
            if self.transform:
                heatmap = self.transform(heatmap)
            heatmap_list.append(heatmap)
        heatmaps = torch.stack(heatmap_list, dim=0)
        return heatmaps, pid, camid, selected_img_paths

# Collate 함수 (Heatmap)
def heatmap_collate_fn(batch):
    heatmaps, pids, camids, img_paths, heatmap_file = zip(*batch)
    max_T = max(h.shape[0] for h in heatmaps)
    padded_batch = []
    for h in heatmaps:
        T_val, C, H, W = h.shape
        if T_val < max_T:
            pad = torch.zeros((max_T - T_val, C, H, W), dtype=h.dtype)
            h = torch.cat([h, pad], dim=0)
        padded_batch.append(h)
    return torch.stack(padded_batch, dim=0), pids, camids, img_paths, heatmap_file

# dataloader 함수
def heatmap_dataloader(Dataset_name):
    # 평가용 이미지 transform (Dataloader.py와 동일)
    val_transforms = T.Compose([
        T.Resize([256, 128], interpolation=InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])

    # Mars 데이터셋 로드
    dataset = Mars()

    # -------------------------
    # 이미지 데이터셋 구성 (Dataloader.py와 동일)
    # -------------------------
    # q_val_set = VideoDataset(dataset.query, seq_len=4, sample='dense', transform=val_transforms)
    # g_val_set = VideoDataset(dataset.gallery, seq_len=4, sample='dense', transform=val_transforms)
    train_set = VideoDataset_inderase(dataset.train, seq_len=4, sample='intelligent', transform=val_transforms)
    
    # -------------------------
    # Heatmap 데이터셋 구성
    # -------------------------
    # Train: HeatmapDataset_inderase 사용, heatmap_root = bbox_train
    heatmap_train_set = HeatmapDataset_inderase(
        dataset = dataset.train,
        seq_len = 4,
        sample = 'intelligent',
        transform = heatmap_transforms,
        heatmap_root = "/home/user/data/heatmap/bbox_train"
    )
    # Query, Gallery: HeatmapDataset 사용, heatmap_root = bbox_test
    # heatmap_query_set = HeatmapDataset(
    #     dataset=dataset.query,
    #     seq_len=4,
    #     sample='dense',
    #     transform=heatmap_transforms,
    #     heatmap_root="/home/user/data/heatmap/bbox_test"
    # )
    # heatmap_gallery_set = HeatmapDataset(
    #     dataset=dataset.gallery,
    #     seq_len=4,
    #     sample='dense',
    #     transform=heatmap_transforms,
    #     heatmap_root="/home/user/data/heatmap/bbox_test"
    # )
    
    # -------------------------
    # DataLoader 구성 (Heatmap DataLoader)
    # -------------------------
    heatmap_train_loader = DataLoader(
        heatmap_train_set,
        batch_size=16,
        num_workers=4,
        collate_fn=heatmap_collate_fn
    )
    # heatmap_query_loader = DataLoader(
    #     heatmap_query_set,
    #     batch_size=16,
    #     num_workers=4,
    #     collate_fn=heatmap_collate_fns
    # )
    # heatmap_gallery_loader = DataLoader(
    #     heatmap_gallery_set,
    #     batch_size=16,
    #     num_workers=4,
    #     collate_fn=heatmap_collate_fn
    # )
    
    return heatmap_train_loader
#, heatmap_query_loader, heatmap_gallery_loader


# 테스트용 main 코드
if __name__ == "__main__":
    ht_loader = heatmap_dataloader("Mars")  # , hq_loader, hg_loader
    
    print("=== Train Heatmap Loader ===")
    for batch in ht_loader:
        heatmaps, pids, camids, img_paths, heatmaps_path = batch
        print("Train Heatmap batch shape:", heatmaps.shape)  # 예: [B, seq_len, 17, 256, 128]
        print("Selected image paths (Train sample):", img_paths[0])
        print("Selected heatmap paths (Train sample):",heatmaps_path[0])
        print("Selected image paths (Train sample):", img_paths[1])
        print("Selected heatmap paths (Train sample):",heatmaps_path[1])
        break

    # print("=== Query Heatmap Loader ===")
    # for batch in hq_loader:
    #     heatmaps, pids, camids, img_paths = batch
    #     print("Query Heatmap batch shape:", heatmaps.shape)
    #     print("Image paths (Query sample):", img_paths[0])
    #     break

    # print("=== Gallery Heatmap Loader ===")
    # for batch in hg_loader:
    #     heatmaps, pids, camids, img_paths = batch
    #     print("Gallery Heatmap batch shape:", heatmaps.shape)
    #     print("Image paths (Gallery sample):", img_paths[0])
    #     break
