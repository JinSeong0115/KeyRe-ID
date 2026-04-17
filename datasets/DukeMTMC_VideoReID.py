from __future__ import absolute_import, print_function

import glob
import os
import os.path as osp
import re
from collections import defaultdict

import numpy as np


class DukeMTMCVideoReID(object):
    """
    DukeMTMC-VideoReID dataset loader.

    The public copies of DukeMTMC-VideoReID are not always organized with
    exactly the same directory names. This loader accepts common layouts such as
    train/query/gallery, bbox_train/bbox_query/bbox_gallery, and
    bounding_box_train/query/bounding_box_test. Tracklets are inferred from
    leaf directories when available; otherwise images are grouped by person and
    camera id parsed from filenames.
    """

    dataset_dir = "DukeMTMC-VideoReID"
    image_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")

    train_candidates = (
        "train",
        "bbox_train",
        "bounding_box_train",
        "DukeMTMC-VideoReID/train",
    )
    query_candidates = (
        "query",
        "bbox_query",
        "bounding_box_query",
        "DukeMTMC-VideoReID/query",
    )
    gallery_candidates = (
        "gallery",
        "bbox_gallery",
        "bbox_test",
        "bounding_box_test",
        "DukeMTMC-VideoReID/gallery",
    )

    def __init__(self, root, min_seq_len=0):
        self.root = osp.abspath(osp.expanduser(root))
        self.train_dir = self._resolve_split_dir(self.train_candidates)
        self.query_dir = self._resolve_split_dir(self.query_candidates)
        self.gallery_dir = self._resolve_split_dir(self.gallery_candidates)

        self._check_before_run()

        train, num_train_tracklets, num_train_pids, num_train_imgs = \
            self._process_dir(self.train_dir, relabel=True, min_seq_len=min_seq_len)
        query, num_query_tracklets, num_query_pids, num_query_imgs = \
            self._process_dir(self.query_dir, relabel=False, min_seq_len=min_seq_len)
        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = \
            self._process_dir(self.gallery_dir, relabel=False, min_seq_len=min_seq_len)

        num_imgs_per_tracklet = num_train_imgs + num_query_imgs + num_gallery_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> DukeMTMC-VideoReID loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids
        self.num_train_cams = self._count_cameras(train)
        self.num_query_cams = self._count_cameras(query)
        self.num_gallery_cams = self._count_cameras(gallery)
        self.num_train_vids = num_train_tracklets
        self.num_query_vids = num_query_tracklets
        self.num_gallery_vids = num_gallery_tracklets

    def _resolve_split_dir(self, candidates):
        roots = [self.root]
        nested_root = osp.join(self.root, self.dataset_dir)
        if osp.isdir(nested_root):
            roots.append(nested_root)

        for base in roots:
            for relpath in candidates:
                split_dir = osp.join(base, relpath)
                if osp.isdir(split_dir):
                    return split_dir
        return None

    def _check_before_run(self):
        if not osp.isdir(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        for split_name, split_dir in (
            ("train", self.train_dir),
            ("query", self.query_dir),
            ("gallery", self.gallery_dir),
        ):
            if split_dir is None or not osp.isdir(split_dir):
                raise RuntimeError(
                    "DukeMTMC-VideoReID {} split is not available under '{}'. "
                    "Expected one of the common split directory names.".format(split_name, self.root)
                )

    def _process_dir(self, split_dir, relabel=False, min_seq_len=0):
        grouped = self._collect_tracklets(split_dir)
        pid_container = sorted({pid for pid, _, _ in grouped.keys() if pid >= 0})
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        tracklets = []
        num_imgs_per_tracklet = []
        for (pid, camid, tracklet_key), img_paths in sorted(grouped.items()):
            if pid < 0 or len(img_paths) < min_seq_len:
                continue
            pid_out = pid2label[pid] if relabel else pid
            tracklets.append((tuple(sorted(img_paths)), pid_out, camid))
            num_imgs_per_tracklet.append(len(img_paths))

        if not tracklets:
            raise RuntimeError("No valid tracklets found in '{}'".format(split_dir))

        return tracklets, len(tracklets), len(pid_container), num_imgs_per_tracklet

    def _collect_tracklets(self, split_dir):
        image_paths = []
        for ext in self.image_exts:
            image_paths.extend(glob.glob(osp.join(split_dir, "**", ext), recursive=True))

        grouped = defaultdict(list)
        for img_path in image_paths:
            pid = self._parse_pid(img_path, split_dir)
            camid = self._parse_camid(img_path)
            tracklet_key = self._parse_tracklet_key(img_path, split_dir, pid, camid)
            grouped[(pid, camid, tracklet_key)].append(img_path)
        return grouped

    @staticmethod
    def _parse_pid(img_path, split_dir):
        basename = osp.basename(img_path)
        match = re.match(r"(-?\d+)", basename)
        if match:
            return int(match.group(1))

        rel_parts = osp.relpath(img_path, split_dir).split(os.sep)
        for part in rel_parts[:-1]:
            match = re.match(r"(-?\d+)", part)
            if match:
                return int(match.group(1))
        raise RuntimeError("Cannot parse person id from '{}'".format(img_path))

    @staticmethod
    def _parse_camid(img_path):
        basename = osp.basename(img_path)
        match = re.search(r"[cC](\d+)", basename)
        if match:
            return int(match.group(1)) - 1

        for part in osp.dirname(img_path).split(os.sep):
            match = re.search(r"[cC](\d+)", part)
            if match:
                return int(match.group(1)) - 1
        return 0

    @staticmethod
    def _parse_tracklet_key(img_path, split_dir, pid, camid):
        rel_dir = osp.dirname(osp.relpath(img_path, split_dir))
        if rel_dir and rel_dir != ".":
            return rel_dir
        return "{}_c{}".format(pid, camid)

    @staticmethod
    def _count_cameras(tracklets):
        return max(camid for _, _, camid in tracklets) + 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DukeMTMC-VideoReID Dataset Loader")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the DukeMTMC-VideoReID dataset directory")
    args = parser.parse_args()

    DukeMTMCVideoReID(root=args.dataset_path)
