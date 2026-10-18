"""Video clip datasets, identity sampler and data loaders."""
import copy
import math
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms import InterpolationMode

from .reference_maps import RegionMapRenderer, ReferenceMapTransform, keypoint_path, load_keypoints

SEQ_LEN = 4


def read_image(img_path):
    while True:
        try:
            return Image.open(img_path).convert('RGB')
        except IOError:
            print(f'IOError when reading {img_path}; retrying')


class RandomErasing(object):
    """Random erasing (Zhong et al., AAAI 2020). Returns (image, erased flag delta_t of Eq. (15), region)."""

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) >= self.probability:
            return img, 0, (0, 0, 0, 0)
        for _ in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[2] - w)
                y1 = random.randint(0, img.size()[1] - h)
                if img.size()[0] == 3:
                    img[0, y1:y1 + h, x1:x1 + w] = self.mean[0]
                    img[1, y1:y1 + h, x1:x1 + w] = self.mean[1]
                    img[2, y1:y1 + h, x1:x1 + w] = self.mean[2]
                else:
                    img[0, y1:y1 + h, x1:x1 + w] = self.mean[0]
                return img, 1, (x1, y1, w, h)
        return img, 0, (0, 0, 0, 0)


def _train_image(img, erase):
    """Resize, horizontal flip, pad + random crop, normalise, random erasing."""
    params = {'flipped': False, 'crop_params': None}
    img = T.Resize([256, 128], interpolation=InterpolationMode.BICUBIC)(img)
    if random.random() < 0.5:
        img = T.functional.hflip(img)
        params['flipped'] = True
    img_padded = T.Pad(10)(img)
    crop_params = T.RandomCrop.get_params(img_padded, output_size=[256, 128])
    img = T.functional.crop(img_padded, *crop_params)
    params['crop_params'] = crop_params
    img = T.ToTensor()(img)
    img = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(img)
    img, label, erased_region = erase(img)
    return img.unsqueeze(0), label, erased_region, params


class TrainClipDataset(Dataset):
    """Training clips: the tracklet is divided into T equal segments and one frame is sampled per segment (Section 3.1).

    Returns the augmented clip [T, 3, 256, 128], the reference part maps [T, 6, 256, 128]
    (same geometric augmentation as the frames), the identity, the camera of every frame
    and the random-erasing flag of every frame.
    """

    def __init__(self, tracklets, data_root, keypoint_root, seq_len=SEQ_LEN):
        self.dataset = tracklets
        self.data_root = data_root
        self.keypoint_root = keypoint_root
        self.seq_len = seq_len
        self.erase = RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
        self.renderer = RegionMapRenderer()
        self.map_transform = ReferenceMapTransform([256, 128])

    def __len__(self):
        return len(self.dataset)

    def _sample_indices(self, num):
        indices = []
        each = max(num // self.seq_len, 1)
        for i in range(self.seq_len):
            if i != self.seq_len - 1:
                indices.append(random.randint(min(i * each, num - 1), min((i + 1) * each - 1, num - 1)))
            else:
                indices.append(random.randint(min(i * each, num - 1), num - 1))
        return indices

    def _reference_map(self, img_path, size, params, label, erased_region):
        kp = load_keypoints(keypoint_path(img_path, self.data_root, self.keypoint_root))
        maps = torch.tensor(self.renderer(kp, size[0], size[1]), dtype=torch.float32)
        maps = self.map_transform(maps)
        if params['flipped']:
            maps = torch.flip(maps, dims=[-1])
        maps = F.pad(maps, (10, 10, 10, 10), mode='constant', value=0)
        if params['crop_params'] is not None:
            maps = T.functional.crop(maps, *params['crop_params'])
        if label == 1:
            x, y, w, h = erased_region
            maps[:, y:min(y + h, maps.shape[1]), x:min(x + w, maps.shape[2])] = 0
        return maps.unsqueeze(0)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        indices = self._sample_indices(len(img_paths))

        imgs, labels, camids, frames = [], [], [], []
        for i in indices:
            img_path = img_paths[int(i)]
            raw = read_image(img_path)
            img, label, erased_region, params = _train_image(raw, self.erase)
            imgs.append(img)
            labels.append(label)
            camids.append(camid)
            frames.append((img_path, raw.size, params, label, erased_region))

        maps = [self._reference_map(*fr) for fr in frames]
        return torch.cat(imgs, dim=0), torch.cat(maps, dim=0), pid, camids, torch.tensor(labels)


def train_collate_fn(batch):
    imgs, maps, pids, camids, erasing = zip(*batch)
    return (torch.stack(imgs, 0), torch.stack(maps, 0), torch.tensor(pids, dtype=torch.int64),
            torch.tensor(camids, dtype=torch.int64), torch.stack(erasing, 0))


class TestClipDataset(Dataset):
    """Inference clips: the tracklet is cut into consecutive non-overlapping clips of T frames, the last one
    padded by repeating its own frames (Section 3.1).

    Returns the clips [num_clips, T, 3, 256, 128], the identity, the camera of every frame
    and the frame paths.
    """

    def __init__(self, tracklets, seq_len=SEQ_LEN, max_length=40):
        self.dataset = tracklets
        self.seq_len = seq_len
        self.max_length = max_length
        self.transform = T.Compose([
            T.Resize([256, 128], interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)
        cur = 0
        frame_indices = list(range(num))
        indices_list = []
        while num - cur > self.seq_len:
            indices_list.append(frame_indices[cur:cur + self.seq_len])
            cur += self.seq_len
        last = frame_indices[cur:]
        for i in last:
            if len(last) >= self.seq_len:
                break
            last.append(i)
        indices_list.append(last)

        clips, camids = [], []
        for indices in indices_list:
            if len(clips) > self.max_length:
                break
            frames = []
            for i in indices:
                frames.append(self.transform(read_image(img_paths[int(i)])).unsqueeze(0))
                camids.append(camid)
            clips.append(torch.cat(frames, dim=0))
        return torch.stack(clips), pid, camids, img_paths


class RandomIdentitySampler(Sampler):
    """Samples N identities with K tracklets each per batch (batch size N*K)."""

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.length = 0
        for pid in self.pids:
            num = len(self.index_dic[pid])
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []
        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)
        return iter(final_idxs)

    def __len__(self):
        return self.length


def build_train_loader(dataset, data_root, keypoint_root, batch_size=16, num_instances=4, num_workers=4):
    train_set = TrainClipDataset(dataset.train, data_root, keypoint_root)
    return DataLoader(train_set, batch_size=batch_size,
                      sampler=RandomIdentitySampler(dataset.train, batch_size, num_instances),
                      num_workers=num_workers, collate_fn=train_collate_fn)


def build_test_sets(dataset):
    return TestClipDataset(dataset.query), TestClipDataset(dataset.gallery)
