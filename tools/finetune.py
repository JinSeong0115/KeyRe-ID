"""Fine-tune KeyRe-ID on iLIDS-VID or PRID-2011, starting from a MARS-trained model.

iLIDS-VID : three stages (lr 0.008 -> 0.001 -> 0.0003) with early stopping on Rank-1;
            each stage starts from the best model of the previous one.
PRID-2011 : 120 epochs with the MARS schedule, evaluated every 10 epochs.

    python tools/finetune.py --dataset iLIDS-VID --data_root /path/to/iLIDS-VID \
        --keypoint_root /path/to/iLIDS-VID_keypoints --init outputs/mars_seed1234/best_rank1.pth \
        --output_dir outputs/ilids_seed1234 --seed 1234
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
from keyreid.solver import AverageMeter, CosineLRScheduler, build_optimizer, build_scheduler  # noqa: E402

# (name, learning rate, max epochs, early-stopping patience)
ILIDS_STAGES = [('Stage1', 0.008, 60, 10), ('Stage2', 0.001, 60, 10), ('Stage3', 0.0003, 120, 15)]
PRID_EPOCHS, PRID_EVAL_INTERVAL = 120, 10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['iLIDS-VID', 'PRID-2011'])
    p.add_argument('--data_root', required=True)
    p.add_argument('--keypoint_root', required=True)
    p.add_argument('--init', required=True, help='MARS-trained KeyRe-ID weights (best_rank1.pth from tools/train.py)')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--split_id', type=int, default=0)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--num_parts', type=int, default=4, choices=[3, 4, 6],
                   help='K parts (Table 1): 3 = head / torso + arms / legs, 4 = head / torso / arms / legs, 6 = one per region')
    p.add_argument('--alpha', type=float, default=0.875, help='alpha in Eq. (13)')
    p.add_argument('--beta', type=float, default=0.0005, help='beta in Eqs. (14), (16)')
    p.add_argument('--gamma', type=float, default=5.0, help='gamma in Eq. (13)')
    p.add_argument('--cam_lambda', type=float, default=3.0, help='lambda in Eq. (1)')
    p.add_argument('--num_workers', type=int, default=4)
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def build_model(args, dataset):
    return KeyReID(num_classes=dataset.num_train_pids, camera_num=dataset.num_train_cams,
                   num_parts=args.num_parts, cam_lambda=args.cam_lambda)


def train_step(model, batch, loss_func, center_criterion, optimizer, optimizer_center, scaler, ema, args,
               double_update):
    imgs, ref_maps, pid, cam, erasing = batch
    optimizer.zero_grad()
    optimizer_center.zero_grad()
    imgs, ref_maps = imgs.cuda(), ref_maps.cuda()
    pid, cam, erasing = pid.cuda(), cam.cuda().view(-1), erasing.cuda()
    with amp.autocast(enabled=True):
        scores, feats, attn_scores, part_loss = model(imgs, cam_label=cam, ref_maps=ref_maps)
        id_tri_loss, center_loss = loss_func(scores, feats, pid)
        attn_loss = (attn_scores * erasing).sum(dim=1).mean()
        loss = id_tri_loss + args.beta * center_loss + attn_loss + args.gamma * part_loss
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    if double_update:
        scaler.update()
    ema.update()
    for prm in center_criterion.parameters():
        if prm.grad is not None:
            prm.grad.data *= 1.0 / args.beta
    scaler.step(optimizer_center)
    scaler.update()
    return loss.item()


# ---------------------------------------------------------------- iLIDS-VID
def ilids_stage(model, args, dataset, loader, query_set, gallery_set, lr, max_epochs, patience, name, save_path):
    loss_func, center_criterion = make_loss(dataset.num_train_pids, alpha=args.alpha)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=0.5 if name == 'Stage1' else 0.1)
    optimizer = build_optimizer(model, base_lr=lr)
    scheduler = CosineLRScheduler(optimizer, t_initial=max_epochs, lr_min=lr * 0.02, warmup_lr_init=lr * 0.1,
                                  warmup_t=3, cycle_limit=1, t_in_epochs=True)
    scaler = amp.GradScaler()
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

    best = (0.0, 0.0, 0.0)
    no_improve = 0
    for epoch in range(1, max_epochs + 1):
        scheduler.step(epoch)
        model.train()
        for batch in loader:
            train_step(model, batch, loss_func, center_criterion, optimizer, optimizer_center, scaler, ema, args,
                       double_update=False)
        r1, r5, mAP = evaluate(model, query_set, gallery_set, args.num_parts, num_workers=args.num_workers)
        if r1 > best[0] or (r1 == best[0] and r5 > best[1]):
            best = (r1, r5, mAP)
            torch.save(model.state_dict(), save_path)
            no_improve = 0
        else:
            no_improve += 1
        print(f'  {name} ep{epoch}: Rank-1 {r1:.4f} Rank-5 {r5:.4f} mAP {mAP:.4f}', flush=True)
        if no_improve >= patience:
            break
    print(f'  {name} best: Rank-1 {best[0]:.4f} Rank-5 {best[1]:.4f} mAP {best[2]:.4f}', flush=True)
    return best


def finetune_ilids(args, dataset, loader, query_set, gallery_set):
    prev = None
    for name, lr, max_epochs, patience in ILIDS_STAGES:
        model = build_model(args, dataset)
        if prev is None:
            model.load_finetune_init(args.init, keep_camera_embedding=False)
        else:
            model.load_state_dict(torch.load(prev, map_location='cpu'), strict=True)
        model.cuda()
        path = os.path.join(args.output_dir, f'{name.lower()}_best.pth')
        best = ilids_stage(model, args, dataset, loader, query_set, gallery_set, lr, max_epochs, patience, name, path)
        prev = path
        del model
        torch.cuda.empty_cache()
    print(f'FINAL Rank-1 {best[0]:.4f} Rank-5 {best[1]:.4f} mAP {best[2]:.4f} ({prev})', flush=True)


# ---------------------------------------------------------------- PRID-2011
def finetune_prid(args, dataset, loader, query_set, gallery_set):
    epochs, eval_interval = PRID_EPOCHS, PRID_EVAL_INTERVAL
    model = build_model(args, dataset)
    model.load_finetune_init(args.init, keep_camera_embedding=True)
    model.cuda()
    loss_func, center_criterion = make_loss(dataset.num_train_pids, alpha=args.alpha)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=0.5)
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer, num_epochs=epochs)
    scaler = amp.GradScaler()
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

    best_rank1 = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        scheduler.step(epoch)
        model.train()
        meter = AverageMeter()
        for batch in loader:
            meter.update(train_step(model, batch, loss_func, center_criterion, optimizer, optimizer_center, scaler,
                                    ema, args, double_update=True), batch[0].size(0))
        msg = f'Epoch {epoch} ({time.time() - t0:.0f}s) loss {meter.avg:.3f}'
        if epoch % eval_interval == 0:
            r1, r5, mAP = evaluate(model, query_set, gallery_set, args.num_parts, num_workers=args.num_workers)
            msg += f' | Rank-1 {r1:.4f} Rank-5 {r5:.4f} mAP {mAP:.4f}'
            if r1 > best_rank1:
                best_rank1 = r1
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_rank1.pth'))
        print(msg, flush=True)
    print(f'Best Rank-1 {best_rank1:.4f}', flush=True)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = build_dataset(args.dataset, args.data_root, args.split_id)
    loader = build_train_loader(dataset, args.data_root, args.keypoint_root, num_workers=args.num_workers)
    query_set, gallery_set = build_test_sets(dataset)
    if args.dataset == 'iLIDS-VID':
        finetune_ilids(args, dataset, loader, query_set, gallery_set)
    else:
        finetune_prid(args, dataset, loader, query_set, gallery_set)


if __name__ == '__main__':
    main()
