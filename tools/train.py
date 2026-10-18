"""Train KeyRe-ID on MARS or DukeMTMC-VideoReID from ImageNet-21K ViT-B/16 weights.

    python tools/train.py --dataset MARS --data_root /path/to/MARS \
        --keypoint_root /path/to/MARS_keypoints --output_dir outputs/mars_seed1234 --seed 1234
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch
from torch.cuda import amp
from torch_ema import ExponentialMovingAverage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from keyreid import KeyReID  # noqa: E402
from keyreid.data import build_dataset, build_train_loader, build_test_sets  # noqa: E402
from keyreid.evaluation import evaluate  # noqa: E402
from keyreid.losses import make_loss  # noqa: E402
from keyreid.solver import AverageMeter, build_optimizer, build_scheduler  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='MARS', choices=['MARS', 'DukeMTMC-VideoReID'])
    p.add_argument('--data_root', required=True)
    p.add_argument('--keypoint_root', required=True, help='keypoint files written by tools/extract_keypoints.py')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--pretrained', default='weights/pretrained/jx_vit_base_p16_224-80ecf9dd.pth')
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--eval_interval', type=int, default=40)
    p.add_argument('--num_parts', type=int, default=4, choices=[3, 4, 6],
                   help='K parts (Table 1): 3 = head / torso + arms / legs, 4 = head / torso / arms / legs, 6 = one per region')
    p.add_argument('--alpha', type=float, default=0.875, help='alpha in Eq. (13): global / local loss balance')
    p.add_argument('--beta', type=float, default=0.0005, help='beta in Eqs. (14), (16): center loss weight')
    p.add_argument('--gamma', type=float, default=5.0, help='gamma in Eq. (13): weight of L_part (Eq. 10)')
    p.add_argument('--cam_lambda', type=float, default=3.0, help='lambda in Eq. (1): camera embedding scale')
    p.add_argument('--num_workers', type=int, default=4)
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = build_dataset(args.dataset, args.data_root)
    train_loader = build_train_loader(dataset, args.data_root, args.keypoint_root, num_workers=args.num_workers)
    query_set, gallery_set = build_test_sets(dataset)

    model = KeyReID(num_classes=dataset.num_train_pids, camera_num=dataset.num_train_cams,
                    num_parts=args.num_parts, cam_lambda=args.cam_lambda, pretrained=args.pretrained)
    model.load_pretrained_vit(args.pretrained)
    model.cuda()

    loss_func, center_criterion = make_loss(dataset.num_train_pids, alpha=args.alpha)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=0.5)
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer, num_epochs=args.epochs)
    scaler = amp.GradScaler()
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

    log = open(os.path.join(args.output_dir, 'log.txt'), 'a')
    best_rank1 = best_map = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        scheduler.step(epoch)
        model.train()
        meter = AverageMeter()
        for imgs, ref_maps, pid, cam, erasing in train_loader:
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            imgs, ref_maps = imgs.cuda(), ref_maps.cuda()
            pid, cam, erasing = pid.cuda(), cam.cuda().view(-1), erasing.cuda()

            with amp.autocast(enabled=True):
                scores, feats, attn_scores, part_loss = model(imgs, cam_label=cam, ref_maps=ref_maps)
                attn_loss = (attn_scores * erasing).sum(dim=1).mean()          # L_attn, Eq. (15)
                id_tri_loss, center_loss = loss_func(scores, feats, pid)
                loss = id_tri_loss + args.beta * center_loss + attn_loss + args.gamma * part_loss   # Eq. (13)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update()
            for prm in center_criterion.parameters():
                if prm.grad is not None:
                    prm.grad.data *= 1.0 / args.beta
            scaler.step(optimizer_center)
            scaler.update()
            meter.update(loss.item(), imgs.size(0))

        msg = f'Epoch {epoch} ({time.time() - t0:.0f}s) loss {meter.avg:.3f}'
        if epoch % args.eval_interval == 0:
            r1, r5, mAP = evaluate(model, query_set, gallery_set, args.num_parts, num_workers=args.num_workers)
            msg += f' | Rank-1 {r1:.4f} Rank-5 {r5:.4f} mAP {mAP:.4f}'
            if r1 > best_rank1:
                best_rank1 = r1
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_rank1.pth'))
            if mAP > best_map:
                best_map = mAP
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_mAP.pth'))
        print(msg, flush=True)
        log.write(msg + '\n')
        log.flush()
    print(f'Best Rank-1 {best_rank1:.4f}  best mAP {best_map:.4f}', flush=True)


if __name__ == '__main__':
    main()
