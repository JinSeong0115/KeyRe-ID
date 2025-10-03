import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import random
import torch
from utility import RandomIdentitySampler, RandomErasing3
from Datasets.MARS import MARS
from Datasets.iLIDS_VID import iLIDSVID
from torchvision.transforms import InterpolationMode


__factory = {
    'MARS': MARS,
    'iLIDSVID': iLIDSVID
}

def train_collate_fn(batch):
    imgs, pids, camids, labels, _, _, _  = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    
    return torch.stack(imgs, dim=0), pids, camids, torch.stack(labels, dim=0)

def val_collate_fn(batch):
    imgs, pids, camids, img_paths = zip(*batch)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    
    return torch.stack(imgs, dim=0), pids, camids_batch, img_paths

def dataloader(Dataset_name):
    val_transforms = T.Compose([
        T.Resize([256, 128], interpolation=InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = __factory[Dataset_name]()
    train_set = VideoDataset_inderase(dataset.train, seq_len=4)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    num_train = dataset.num_train_vids

    train_loader = DataLoader(train_set, batch_size=64,sampler=RandomIdentitySampler(dataset.train, 64, 4), num_workers=4, collate_fn=train_collate_fn)
    q_val_set = VideoDataset(dataset.query, seq_len=4, transform=val_transforms)
    g_val_set = VideoDataset(dataset.gallery, seq_len=4, transform=val_transforms)

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

def process_image(img_path, erase):
    img = read_image(img_path)
    transform_params = {'flipped': False, 'crop_params': None}

    # Resize
    img = T.Resize([256, 128], interpolation=InterpolationMode.BICUBIC)(img)
    # Random Horizontal Flip
    if random.random() < 0.5:
        img = T.functional.hflip(img)
        transform_params['flipped'] = True
    # Pad
    img_padded = T.Pad(10)(img)
    # Random Crop
    crop_params = T.RandomCrop.get_params(img_padded, output_size=[256, 128])
    img = T.functional.crop(img_padded, *crop_params)
    transform_params['crop_params'] = crop_params
    # ToTensor and Normalize
    img = T.ToTensor()(img)
    img = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(img)
    
    # img, label, erased_region = erase(img)  # img, 0, (0, 0, 0, 0) or img, 1, (x1, y1, w, h)  
    result = erase(img)
    img, label, erased_region = result
    img = img.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
    
    return img, label, erased_region, transform_params        


class VideoDataset(Dataset):  # test dataset
    """
    Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    def __init__(self, dataset, seq_len=4, transform=None , max_length=40):
        self.dataset = dataset
        self.seq_len = seq_len
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)
        
        cur_index=0
        frame_indices = [i for i in range(num)]
        indices_list=[]
        while num-cur_index > self.seq_len:
            indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
            cur_index += self.seq_len
        last_seq = frame_indices[cur_index:]
        for index in last_seq:
            if len(last_seq) >= self.seq_len:
                break
            last_seq.append(index)
        indices_list.append(last_seq)
        
        imgs_list=[]   
        camids = []         
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
                camids.append(camid)
            imgs = torch.cat(imgs, dim=0)
            imgs_list.append(imgs)
        imgs_array = torch.stack(imgs_list)
        return imgs_array, pid, camids, img_paths

class VideoDataset_inderase(Dataset):  # train dataset
    """
    Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    def __init__(self, dataset, seq_len=4, max_length=40):
        self.dataset = dataset
        self.seq_len = seq_len
        self.max_length = max_length
        self.erase = RandomErasing3(probability=0.5, mean=[0.485, 0.456, 0.406])
        print(f"self.erase type: {type(self.erase)}")     

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)
        indices = []
        each = max(num//self.seq_len, 1)
        
        for i in range(self.seq_len):
            if i != self.seq_len-1:
                indices.append(random.randint(min(i*each, num-1), min((i+1)*each-1, num-1)))
            else:
                indices.append(random.randint(min(i*each, num-1), num-1))
        
        imgs = []
        labels = []
        camids = []
        selected_img_paths = []
        erased_regions = []
        transform_params_list = []
        
        for index in indices:
            index = int(index)
            img_path = img_paths[index]
            img, label, erased_region, transform_params = process_image(img_path, self.erase)
            
            imgs.append(img)
            camids.append(camid)
            labels.append(label)
            selected_img_paths.append(img_path)
            erased_regions.append(erased_region)
            transform_params_list.append(transform_params)
            
        imgs = torch.cat(imgs, dim=0)
        labels = torch.tensor(labels)

        return imgs, pid, camids, labels, selected_img_paths, erased_regions, transform_params_list
    
