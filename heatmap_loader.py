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
from Datasets.MARS_dataset import Mars
from Datasets.iLDSVID import iLIDSVID
from utility import RandomIdentitySampler

__factory = {
    'Mars':Mars,
    'iLIDSVID':iLIDSVID
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
    # Calculate min and max values per channel (keepdim=True for broadcasting)
    min_val = tensor.amin(dim=(1,2), keepdim=True)
    max_val = tensor.amax(dim=(1,2), keepdim=True)
    scaled = (tensor - min_val) / (max_val - min_val + eps)
    return scaled

class CustomHeatmapTransform(object):
    def __init__(self, size):
        self.size = size  # (height, width)
        self.resize = HeatmapResize(size)
    def __call__(self, tensor):
        # tensor shape: (C, H, W), where C=6
        tensor = self.resize(tensor) 
        tensor = Min_Max_Scaling(tensor)  # Perform min-max scaling per channel (value range [0, 1])
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

# heatmap_dataloader function
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
        dataset=train_data,
        seq_len=4,
        heatmap_transform=CustomHeatmapTransform([256, 128]),
        heatmap_root="./dataset/heatmap/bbox_train"
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
        transform=val_transforms,
        heatmap_transform=CustomHeatmapTransform([256, 128]),
        heatmap_root="./dataset/heatmap/bbox_test"
    )
    g_val_set = Heatmap_Dataset(
        dataset=gallery_data,
        seq_len=4,
        transform=val_transforms,
        heatmap_transform=CustomHeatmapTransform([256, 128]),
        heatmap_root="./dataset/heatmap/bbox_test"
    )
    
    return train_loader, len(query_data), num_classes, cam_num, num_train, q_val_set, g_val_set

# Heatmap_Dataset (Test)
# VideoDataset --> imgs_array, pid, camids, img_paths
class Heatmap_Dataset(VideoDataset): 
    """
    A class that calls the __getitem__ method of the parent class (VideoDataset)
    to retrieve imgs and img_paths, then loads and returns heatmaps
    aligned with the clip structure of imgs.

    Returns: (imgs, heatmaps, pid, camid, img_paths)
    """
    def __init__(self, heatmap_transform, heatmap_root, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.heatmap_transform = heatmap_transform
        self.heatmap_root = heatmap_root

    def __getitem__(self, index):
        imgs, pid, camids, img_paths = super().__getitem__(index)
        
        # imgs shape: [clips, seq_len, 3, H, W]
        clips, seq_len = imgs.shape[0], imgs.shape[1]
        
        heatmap_list = []
        for i in range(clips):
            clip_heatmaps = []
            # Calculate indices for img_paths corresponding to each clip
            start_idx = i*seq_len
            end_idx = min(start_idx+seq_len, len(img_paths))
            for frame_path in img_paths[start_idx:end_idx]:
                file_name = os.path.basename(frame_path).replace(".jpg", ".npy")
                person_id = os.path.basename(os.path.dirname(frame_path))
                heatmap_file = os.path.join(self.heatmap_root, person_id, file_name)

                if not os.path.exists(heatmap_file):
                    print(f"⚠️ Heatmap file not found: {heatmap_file}")
                    raise FileNotFoundError(f"Missing heatmap file: {heatmap_file}")
                else:
                    heatmap_np = np.load(heatmap_file)  # shape: (6, H, W)
                    heatmap = torch.tensor(heatmap_np, dtype=torch.float32)

                # Normalize per channel
                for c in range(heatmap.shape[0]):
                    max_val = heatmap[c].max()
                    if max_val > 0:
                        heatmap[c] = heatmap[c] / max_val
                    else:
                        heatmap[c] = torch.zeros_like(heatmap[c])  # Explicitly set to 0

                # Apply transform
                if self.heatmap_transform is not None:
                    heatmap = self.heatmap_transform(heatmap)
                    if heatmap.shape[1:] != self.heatmap_transform.size:
                        heatmap = T.Resize(self.heatmap_transform.size)(heatmap)
                
                clip_heatmaps.append(heatmap)
            
            # Pad if the number of frames in the clip is less than seq_len
            pad_size = seq_len - len(clip_heatmaps)
            if pad_size > 0:
                padding = torch.zeros(pad_size, *clip_heatmaps[0].shape)
                clip_heatmaps = torch.cat([torch.stack(clip_heatmaps), padding], dim=0)
            else:
                clip_heatmaps = torch.stack(clip_heatmaps)
            
            heatmap_list.append(clip_heatmaps)
        
        heatmaps = torch.stack(heatmap_list, dim=0)  # [clips, seq_len, 6, H, W]

        return imgs, heatmaps, pid, camids, img_paths

# Heatmap_Dataset_inderase (Train)
# VideoDataset_inderase --> imgs, pid, camids, labels, selected_img_paths, erased_regions, transform_params_list
class Heatmap_Dataset_inderase(VideoDataset_inderase):
    def __init__(self, heatmap_transform, heatmap_root, dataset, *args, **kwargs):
        super().__init__(dataset=dataset, *args, **kwargs)
        self.heatmap_transform = heatmap_transform  # Heatmap transformation function
        self.heatmap_root = heatmap_root  # Root path where heatmap files are stored
        self.heatmap_cache = {}  # Heatmap cache for performance optimization

    def load_heatmap(self, img_path):
        if img_path not in self.heatmap_cache:
            file_name = os.path.basename(img_path).replace(".jpg", ".npy")
            person_id = os.path.basename(os.path.dirname(img_path))
            heatmap_file = os.path.join(self.heatmap_root, person_id, file_name)

            # Raise error if file does not exist
            if not os.path.exists(heatmap_file):
                raise FileNotFoundError(f"Heatmap file not found: {heatmap_file}")

            # Load and transform heatmap
            heatmap = np.load(heatmap_file)
            heatmap = torch.tensor(heatmap, dtype=torch.float32)
            self.heatmap_cache[img_path] = heatmap

        return self.heatmap_cache[img_path]

    def __getitem__(self, index):
        imgs, pid, camids, labels, selected_img_paths, erased_regions, transform_params_list = super().__getitem__(index)

        # Load heatmaps using selected_img_paths
        heatmap_list = []
        for label, img_path, erased_region, transform_params in zip(labels, selected_img_paths, erased_regions, transform_params_list):
            heatmap = self.load_heatmap(img_path)
            heatmap = self.heatmap_transform(heatmap)
            # Apply the same transformations as the image
            if transform_params['flipped']:
                heatmap = torch.flip(heatmap, dims=[-1])
            heatmap = F.pad(heatmap, (10, 10, 10, 10), mode='constant', value=0)  # Padding
            if transform_params['crop_params'] is not None:
                heatmap = T.functional.crop(heatmap, *transform_params['crop_params'])
            if label == 1:  # If erasing is applied
                x, y, w, h = erased_region
                heatmap_h, heatmap_w = heatmap.shape[1], heatmap.shape[2]
                y_end = min(y + h, heatmap_h)
                x_end = min(x + w, heatmap_w)
                heatmap[:, y:y_end, x:x_end] = 0

            heatmap = heatmap.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
            heatmap_list.append(heatmap)

        heatmaps = torch.cat(heatmap_list, dim=0)  # (seq_len, C, H, W)
        return imgs, heatmaps, pid, camids, labels