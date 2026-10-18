"""Evaluate a KeyRe-ID checkpoint with the 3,840-d retrieval descriptor.

    python tools/test.py --dataset MARS --data_root /path/to/MARS --weights weights/MARS/keyreid_mars_seed1234.pth
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from keyreid import KeyReID, load_checkpoint  # noqa: E402
from keyreid.data import build_dataset, build_test_sets  # noqa: E402
from keyreid.evaluation import evaluate  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['MARS', 'DukeMTMC-VideoReID', 'iLIDS-VID', 'PRID-2011'])
    p.add_argument('--data_root', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--split_id', type=int, default=0, help='iLIDS-VID / PRID-2011 split')
    p.add_argument('--num_parts', type=int, default=4, choices=[3, 4, 6],
                   help='K parts (Table 1): 3 = head / torso + arms / legs, 4 = head / torso / arms / legs, 6 = one per region')
    p.add_argument('--cam_lambda', type=float, default=3.0)
    p.add_argument('--fusion_weight', type=float, default=0.5, help='w in Eq. (17)')
    p.add_argument('--num_workers', type=int, default=4)
    args = p.parse_args()

    dataset = build_dataset(args.dataset, args.data_root, args.split_id)
    query_set, gallery_set = build_test_sets(dataset)
    model = KeyReID(num_classes=dataset.num_train_pids, camera_num=dataset.num_train_cams,
                    num_parts=args.num_parts, cam_lambda=args.cam_lambda)
    load_checkpoint(model, args.weights)
    model.cuda().eval()

    r1, r5, mAP = evaluate(model, query_set, gallery_set, args.num_parts, args.fusion_weight,
                           num_workers=args.num_workers)
    print(f'[{args.dataset}] Rank-1 {r1 * 100:.2f}  Rank-5 {r5 * 100:.2f}  mAP {mAP * 100:.2f}')


if __name__ == '__main__':
    with torch.no_grad():
        main()
