"""Reference part maps for training supervision (Section 3.2).

Human keypoints are extracted once, offline, with OpenPifPaf (tools/extract_keypoints.py).
Each of the 17 joints is turned into a 2D Gaussian heatmap of unit peak centred at the joint
(Eq. 3), with a kernel set to 1/11 of the crop width (sigma ~ 5.8 px on a 128-px crop); joints
whose detection confidence falls below 0.1 are discarded and contribute nothing. The joints
are grouped into R = 6 anatomical regions (Table 1), and each region heatmap is the pixel-wise
maximum of the heatmaps of its joints (Eq. 4). Each region map is rescaled so that its largest
entry is one; the model then average-pools it onto the patch grid and averages it over the
T frames of the clip (Eq. 5). The maps are used during training only.
"""
import json
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from scipy.signal.windows import gaussian

JOINT_NAMES = ['nose', 'Leye', 'Reye', 'LEar', 'REar', 'LS', 'RS', 'LE', 'RE', 'LW', 'RW',
               'LH', 'RH', 'LK', 'RK', 'LA', 'RA']
REGIONS = OrderedDict([
    ('head', ['nose', 'Leye', 'Reye', 'LEar', 'REar']),
    ('torso', ['LS', 'RS', 'LH', 'RH']),
    ('left_arm', ['LE', 'LW']),
    ('right_arm', ['RE', 'RW']),
    ('left_leg', ['LK', 'LA']),
    ('right_leg', ['RK', 'RA']),
])
_JOINT_INDEX = {name: i for i, name in enumerate(JOINT_NAMES)}


def _gaussian_kernel(width, g_scale=11):
    radius = int(width / g_scale)
    size = radius * 2 + 1
    g1d = gaussian(size, std=size / 4).reshape(size, 1)
    kernel = np.outer(g1d, g1d)
    return kernel / np.max(kernel)


class RegionMapRenderer:
    """Keypoints [17, (x, y, confidence)] -> region heatmaps [R=6, H, W] (Eqs. 3-4)."""

    def __init__(self, g_scale=11, vis_thresh=0.1):
        self.g_scale = g_scale
        self.vis_thresh = vis_thresh
        self._kernels = {}

    def kernel(self, width):
        if width not in self._kernels:
            self._kernels[width] = _gaussian_kernel(width, self.g_scale)
        return self._kernels[width]

    def __call__(self, keypoints, width, height):
        kernel = self.kernel(width)
        r = kernel.shape[0] // 2
        maps = np.zeros((len(REGIONS), height, width))
        for ri, joints in enumerate(REGIONS.values()):
            region = np.zeros((height, width))
            for name in joints:
                x, y, c = keypoints[_JOINT_INDEX[name]]
                if c <= self.vis_thresh:
                    continue
                kx, ky = int(x), int(y)
                if not (0 <= kx < width and 0 <= ky < height):
                    continue
                rt, rb = max(0, ky - r), min(height, ky + r + 1)
                rl, rr = max(0, kx - r), min(width, kx + r + 1)
                sub = kernel[r - (ky - rt):r + (rb - ky), r - (kx - rl):r + (rr - kx)]
                patch = region[rt:rb, rl:rr]
                if patch.shape == sub.shape:
                    region[rt:rb, rl:rr] = np.maximum(patch, sub)
            maps[ri] = region
        return maps


def keypoint_path(img_path, data_root, keypoint_root):
    rel = os.path.relpath(img_path, data_root)
    return os.path.join(keypoint_root, os.path.splitext(rel)[0] + '.pose')


def load_keypoints(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'keypoint file not found: {path} (run tools/extract_keypoints.py)')
    with open(path) as f:
        return np.array(json.load(f), dtype=np.float32)


class ReferenceMapTransform:
    """Resize to the 256x128 input, rescale each region map so that its largest entry is one, map to [-1, 1]."""

    def __init__(self, size=(256, 128)):
        self.size = size
        self.normalize = T.Normalize(mean=[0.5] * 6, std=[0.5] * 6)

    def __call__(self, maps):
        maps = F.interpolate(maps.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False).squeeze(0)
        lo = maps.amin(dim=(1, 2), keepdim=True)
        hi = maps.amax(dim=(1, 2), keepdim=True)
        maps = (maps - lo) / (hi - lo + 1e-6)
        return self.normalize(maps)
