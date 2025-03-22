import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFile
import os.path as osp
import random
import torch
import numpy as np
import math
from utility import RandomIdentitySampler,RandomErasing3
from Datasets.MARS_dataset import Mars
from torchvision.transforms import InterpolationMode

__factory = {
    'Mars':Mars,
}

def train_collate_fn(batch):
    imgs, pids, camids, erasing_labels, img_paths  = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    
    return torch.stack(imgs, dim=0), pids, camids, torch.stack(erasing_labels , dim=0), img_paths

def val_collate_fn(batch):
    imgs, pids, camids, img_paths = zip(*batch)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    
    return torch.stack(imgs, dim=0), pids, camids_batch, img_paths

def dataloader(Dataset_name):
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

    dataset = __factory[Dataset_name]()
    train_set = VideoDataset_inderase(dataset.train, seq_len=4, sample='intelligent', transform=train_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    num_train = dataset.num_train_vids

    train_loader = DataLoader(train_set, batch_size=64, sampler=RandomIdentitySampler(dataset.train, 64, 4), num_workers=4, collate_fn=train_collate_fn)
    q_val_set = VideoDataset(dataset.query, seq_len=4, sample='dense', transform=val_transforms)
    g_val_set = VideoDataset(dataset.gallery, seq_len=4, sample='dense', transform=val_transforms)

    return train_loader, len(dataset.query), num_classes, cam_num, num_train, q_val_set, g_val_set

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class VideoDataset(Dataset):  # test dataset
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=4, sample='evenly', transform=None , max_length=40):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]
            if len(indices) < self.seq_len:
                indices=np.array(indices)
                indices = np.append(indices , [indices[-1] for i in range(self.seq_len - len(indices))])
            else:
                indices=np.array(indices)
            imgs = []
            target_cam=[]
            for index in indices:
                index=int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                target_cam.append(camid)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            return imgs, pid, target_cam

        elif self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            cur_index=0
            frame_indices = [i for i in range(num)]
            indices_list=[]
            while num-cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
                cur_index+=self.seq_len
            last_seq=frame_indices[cur_index:]
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            
            indices_list.append(last_seq)
            imgs_list=[]
            target_cam=[]
            for indices in indices_list:
                if len(imgs_list) > self.max_length:
                    break 
                imgs = []
                for index in indices:
                    index=int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                    target_cam.append(camid)
                imgs = torch.cat(imgs, dim=0)
                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, target_cam, img_paths

        elif self.sample == 'dense_subset':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.max_length - 1)
            begin_index = random.randint(0, rand_end)
            
            cur_index=begin_index
            frame_indices = [i for i in range(num)]
            indices_list=[]
            while num-cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
                cur_index+=self.seq_len
            last_seq=frame_indices[cur_index:]
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            
            indices_list.append(last_seq)
            imgs_list=[]
            for indices in indices_list:
                if len(imgs_list) > self.max_length:
                    break 
                imgs = []
                for index in indices:
                    index=int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)
                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid
        
        elif self.sample == 'intelligent_random':
            indices = []
            each = max(num//seq_len,1)
            for  i in range(seq_len):
                if i != seq_len -1:
                    indices.append(random.randint(min(i*each , num-1), min( (i+1)*each-1, num-1)) )
                else:
                    indices.append(random.randint(min(i*each , num-1), num-1) )
            print(len(indices))
            imgs = []
            for index in indices:
                index=int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            return imgs, pid, camid
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

class VideoDataset_inderase(Dataset):  # train dataset
    """
    Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=4, sample='evenly', transform=None , max_length=40):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.max_length = max_length
        self.erase = RandomErasing3(probability=0.5, mean=[0.485, 0.456, 0.406])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)
        if self.sample != "intelligent":
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices1 = frame_indices[begin_index:end_index]
            indices = []
            for index in indices1:
                if len(indices1) >= self.seq_len:
                    break
                indices.append(index)
            indices=np.array(indices)
        else:
            indices = []
            each = max(num//self.seq_len,1)
            for  i in range(self.seq_len):
                if i != self.seq_len -1:
                    indices.append(random.randint(min(i*each , num-1), min( (i+1)*each-1, num-1)))
                else:
                    indices.append(random.randint(min(i*each , num-1), num-1))
        imgs = []
        labels = []
        target_cam=[]
        selected_img_paths = []
        
        for index in indices:
            index=int(index)
            img_path = img_paths[index]
            selected_img_paths.append(img_path)
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img, temp = self.erase(img)  # random erasing
            labels.append(temp)
            img = img.unsqueeze(0)
            imgs.append(img)
            target_cam.append(camid)
        labels = torch.tensor(labels)
        imgs = torch.cat(imgs, dim=0)
        
        return imgs, pid, target_cam, labels, selected_img_paths

if __name__ == "__main__":
    # 테스트할 데이터셋 이름 (PRID, Mars, iLIDSVID 중 선택 가능)
    dataset_name = "Mars"
    
    # 데이터 로더 실행
    print(f"🔍 Testing dataloader for dataset: {dataset_name}")
    train_loader, num_query, num_classes, cam_num, num_train, q_val_set, g_val_set = dataloader(dataset_name)
    batch = next(iter(train_loader))
    imgs, pid, target_cam, labels, img_path = batch  # 배치 데이터 언패킹

    print(img_path[0])
    print(pid[:4])
    print(target_cam[0])
    
    
    
# if __name__ == "__main__":
#     # 테스트할 데이터셋 이름 (PRID, Mars, iLIDSVID 중 선택 가능)
#     dataset_name = "Mars"
    
#     # 데이터 로더 실행
#     print(f"🔍 Testing dataloader for dataset: {dataset_name}")
#     train_loader, num_query, num_classes, cam_num, num_train, q_val_set, g_val_set = dataloader(dataset_name)

#     # 데이터 로더 크기 출력
#     print(f"✅ Train loader size: {len(train_loader)}")
#     print(f"✅ Query set size: {num_query}, Gallery set size: {len(g_val_set)}")
#     print(f"✅ Number of classes: {num_classes}, Camera count: {cam_num}, View count: {num_train}")

#     # 첫 번째 배치 가져오기
#     for idx, (imgs, pids, camids, labels) in enumerate(train_loader):
#         print(f"\n🔹 Batch {idx}:")
#         print(f"   📌 Image batch shape: {imgs.shape} (batch_size, seq_len, C, H, W)")
#         print(f"   📌 PIDs: {pids.shape} -> {pids[:5]}")
#         print(f"   📌 Camera IDs: {camids.shape} -> {camids[:5]}")
#         print(f"   📌 Labels (Random Erasing): {labels.shape} -> {labels[:5]}")
#         break  # 첫 번째 배치만 출력

#     # Query 데이터셋 샘플 출력
#     print(f"\n🔍 Query dataset sample:")
#     q_sample = q_val_set[0]
#     print(f"   📌 Query sample image shape: {q_sample[0].shape}")
#     print(f"   📌 Query PID: {q_sample[1]}, Camera ID: {q_sample[2]}")

#     # Gallery 데이터셋 샘플 출력
#     print(f"\n🔍 Gallery dataset sample:")
#     g_sample = g_val_set[0]
#     print(f"   📌 Gallery sample image shape: {g_sample[0].shape}")
#     print(f"   📌 Gallery PID: {g_sample[1]}, Camera ID: {g_sample[2]}")

#     print("\n✅ Dataloader test completed successfully!")