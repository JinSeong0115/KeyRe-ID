import os
import argparse
import time
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.cuda import amp
from torch_ema import ExponentialMovingAverage
from heatmap_loader import heatmap_dataloader
from keyreid import KeyReID
from losses import make_loss
from utils import AverageMeter, optimizer as build_optimizer, scheduler as build_scheduler


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=21):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    all_cmc, all_AP, num_valid_q = [], [], 0.
    for q_idx in range(num_q):
        q_pid, q_camid = q_pids[q_idx], q_camids[q_idx]
        order = indices[q_idx]
        keep = np.invert((g_pids[order] == q_pid) & (g_camids[order] == q_camid))
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = np.asarray([x / (i+1.) for i, x in enumerate(tmp_cmc)]) * orig_cmc
        all_AP.append(tmp_cmc.sum() / num_rel)
    assert num_valid_q > 0
    return np.asarray(all_cmc).astype(np.float32).sum(0) / num_valid_q, np.mean(all_AP)


def test(model, queryloader, galleryloader, pool='avg', use_gpu=True):
    model.eval()
    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for imgs, heatmaps, pids, camids, _ in queryloader:
            if use_gpu:
                imgs, heatmaps = imgs.cuda(), heatmaps.cuda()
            features = model(imgs, heatmaps, pids, cam_label=camids)
            features = features.view(imgs.size(0), -1).mean(0).cpu()
            qf.append(features); q_pids.append(pids); q_camids.extend(camids)
        qf = torch.stack(qf); q_pids = np.asarray(q_pids); q_camids = np.asarray(q_camids)
        gf, g_pids, g_camids = [], [], []
        for imgs, heatmaps, pids, camids, _ in galleryloader:
            if use_gpu:
                imgs, heatmaps = imgs.cuda(), heatmaps.cuda()
            features = model(imgs, heatmaps, pids, cam_label=camids)
            features = features.view(imgs.size(0), -1).mean(0).cpu()
            gf.append(features); g_pids.append(pids); g_camids.extend(camids)
    gf = torch.stack(gf); g_pids = np.asarray(g_pids); g_camids = np.asarray(g_camids)
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(1, keepdim=True).expand(n, m).t()
    distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    cmc, mean_ap = evaluate(distmat.numpy(), q_pids, g_pids, q_camids, g_camids)
    print("Results -- mAP: {:.1%}, CMC r1: {:.4f}".format(mean_ap, cmc[0]))
    return cmc[0], mean_ap


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="KeyRe-ID2 Multi-Seed")
    parser.add_argument("--dataset_name", default="DukeMTMC-VideoReID", type=str)
    parser.add_argument('--ViT_path', required=True, type=str)
    parser.add_argument("--dataset_root", required=True, type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--epochs", default=120, type=int)
    parser.add_argument("--eval_interval", default=10, type=int)
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Output dirs
    eval_dir = os.path.join(args.output_dir, "evaluate")
    weight_dir = os.path.join(args.output_dir, "weights")
    loss_dir = os.path.join(args.output_dir, "loss")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(loss_dir, exist_ok=True)

    # Data & Model (identical to original train.py)
    heatmap_train_loader, _, num_classes, camera_num, _, q_val_set, g_val_set = heatmap_dataloader(args.dataset_name, args.dataset_root)
    model = KeyReID(num_classes=num_classes, camera_num=camera_num, pretrainpath=args.ViT_path)
    model.load_param(args.ViT_path)

    loss_fun, center_criterion = make_loss(num_classes=num_classes)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=0.5)
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)
    scaler = amp.GradScaler()

    device = "cuda"
    model = model.to(device)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    cmc_rank1 = 0
    best_map = 0
    loss_history = []

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        scheduler.step(epoch)
        model.train()

        for Epoch_n, (imgs, heatmaps, pid, target_cam, erasing_labels) in enumerate(heatmap_train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()

            imgs = imgs.to(device)
            heatmaps = heatmaps.to(device)
            pid = pid.to(device)
            target_cam = target_cam.to(device)
            erasing_labels = erasing_labels.to(device)

            with amp.autocast(enabled=True):
                target_cam = target_cam.view(-1)
                score, feat, a_vals = model(imgs, heatmaps, pid, cam_label=target_cam)
                attn_noise = a_vals * erasing_labels
                attn_loss = attn_noise.sum(dim=1).mean()
                loss_id, center = loss_fun(score, feat, pid)
                loss = loss_id + 0.0005 * center + attn_loss

            # Exactly matching original train.py order
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update()

            for param in center_criterion.parameters():
                param.grad.data *= (1. / 0.0005)
            scaler.step(optimizer_center)
            scaler.update()

            if isinstance(score, list):
                acc = (score[0].max(1)[1] == pid).float().mean()
            else:
                acc = (score.max(1)[1] == pid).float().mean()

            loss_meter.update(loss.item(), imgs.shape[0])
            acc_meter.update(acc, 1)
            loss_history.append(loss.item())

            torch.cuda.synchronize()
            if (Epoch_n+1) % 400 == 0:
                print("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Lr: {:.2e}".format(
                    epoch, Epoch_n+1, len(heatmap_train_loader),
                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        print("Epoch {} finished in {:.1f}s".format(epoch, time.time() - start_time))

        if epoch % args.eval_interval == 0:
            cmc, mAP = test(model, q_val_set, g_val_set)
            print('CMC: %.4f, mAP: %.4f' % (cmc, mAP))
            with open(os.path.join(eval_dir, 'matrix_best.txt'), 'a') as f:
                f.write('Epoch {}: CMC = {:.4f}, mAP = {:.4f}\n'.format(epoch, cmc, mAP))
            if cmc_rank1 < cmc:
                cmc_rank1 = cmc
                torch.save(model.state_dict(), os.path.join(weight_dir, args.dataset_name + 'best_CMC.pth'))
            if best_map < mAP:
                best_map = mAP
                torch.save(model.state_dict(), os.path.join(weight_dir, args.dataset_name + 'best_mAP.pth'))

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Iterations"); plt.ylabel("Loss")
    plt.title("Training Loss"); plt.legend(); plt.grid()
    plt.savefig(os.path.join(loss_dir, "loss_plot.png")); plt.close()

    print("\n" + "="*60)
    print("FINAL RESULTS (seed={})".format(args.seed))
    print("  Best Rank-1: {:.4f}".format(cmc_rank1))
    print("  Best mAP:    {:.4f}".format(best_map))
    print("="*60)
