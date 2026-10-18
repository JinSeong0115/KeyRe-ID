from __future__ import print_function, absolute_import
from scipy.io import loadmat
import os.path as osp
import numpy as np
import json
import os
import glob

def read_json(fpath):
    with open(fpath, 'r') as f:
        return json.load(f)

def write_json(obj, fpath):
    os.makedirs(osp.dirname(fpath), exist_ok=True)
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4)

class PRID(object):
    """
    PRID-2011
    Reference:
    Hirzer et al. Person Re-Identification by Descriptive and Discriminative Classification. SCIA 2011.
    
    Dataset statistics:
    # identities: 200
    # tracklets: 400
    # cameras: 2
    """
    def __init__(self, root, split_id=0):
        self.root = osp.abspath(osp.expanduser(root))
        self.cam_a_path = osp.join(self.root, 'multi_shot', 'cam_a')
        self.cam_b_path = osp.join(self.root, 'multi_shot', 'cam_b')
        self.split_mat_path = osp.join(self.root, 'train_test_splits_prid.mat')
        self.split_path = osp.join(self.root, 'splits.json')
        
        self._check_before_run()
        self._prepare_split()
        
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError("split_id exceeds range")
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']
        
        train, num_train_tracklets, num_train_pids, num_imgs_train = \
            self._process_data(train_dirs, cam1=True, cam2=True)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
            self._process_data(test_dirs, cam1=True, cam2=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
            self._process_data(test_dirs, cam1=False, cam2=True)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery

        print("=> PRID-2011 loaded")
        print("  subset   | # ids | # tracklets")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))

        self.train = train
        self.query = query
        self.gallery = gallery
        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids
        self.num_train_cams = 2
        self.num_query_cams = 1
        self.num_gallery_cams = 1
        self.num_train_vids = num_train_tracklets
        self.num_query_vids = num_query_tracklets
        self.num_gallery_vids = num_gallery_tracklets

    def _check_before_run(self):
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.cam_a_path):
            raise RuntimeError("'{}' is not available".format(self.cam_a_path))

    def _prepare_split(self):
        if osp.exists(self.split_path):
            return
        print("Creating splits from mat file")
        mat = loadmat(self.split_mat_path)['ls_set']
        num_splits = mat.shape[0]
        num_total_ids = mat.shape[1]
        # num_total_ids can vary (178 or 200)
        num_ids_each = num_total_ids // 2
        
        cam_a_dirs = sorted(os.listdir(self.cam_a_path))
        cam_b_dirs = sorted(os.listdir(self.cam_b_path))
        # Only use persons appearing in both cameras (first 200)
        common_dirs = sorted(list(set(cam_a_dirs) & set(cam_b_dirs)))[:num_total_ids]
        
        splits = []
        for i in range(num_splits):
            train_idxs = sorted([int(x)-1 for x in mat[i, num_ids_each:]])
            test_idxs = sorted([int(x)-1 for x in mat[i, :num_ids_each]])
            train_dirs = [common_dirs[j] for j in train_idxs]
            test_dirs = [common_dirs[j] for j in test_idxs]
            splits.append({'train': train_dirs, 'test': test_dirs})
        
        write_json(splits, self.split_path)
        print("Created {} splits".format(len(splits)))

    def _process_data(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        num_imgs_per_tracklet = []
        dirname2pid = {d: i for i, d in enumerate(dirnames)}
        
        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_a_path, dirname)
                img_names = sorted(glob.glob(osp.join(person_dir, '*.png')))
                if len(img_names) > 0:
                    tracklets.append((tuple(img_names), dirname2pid[dirname], 0))
                    num_imgs_per_tracklet.append(len(img_names))
            if cam2:
                person_dir = osp.join(self.cam_b_path, dirname)
                img_names = sorted(glob.glob(osp.join(person_dir, '*.png')))
                if len(img_names) > 0:
                    tracklets.append((tuple(img_names), dirname2pid[dirname], 1))
                    num_imgs_per_tracklet.append(len(img_names))
        
        num_tracklets = len(tracklets)
        num_pids = len(dirnames)
        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet
