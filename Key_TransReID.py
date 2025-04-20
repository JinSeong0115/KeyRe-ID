import os
import argparse
import logging
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.cuda import amp
import torch.distributed as dist
from torch_ema import ExponentialMovingAverage
from Dataloader import dataloader
from heatmap_loader import heatmap_dataloader
from Key_Trans_model import Key_Trans
from Loss_fun import make_loss
from utility import AverageMeter, optimizer, scheduler

def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=21):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    num_valid_q = 0.

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def test(model, queryloader, galleryloader, pool='avg', use_gpu=True):
    model.eval()    
    with torch.no_grad():
        # ----- Query ----- #
        qf, q_pids, q_camids = [], [], []
        for imgs, heatmaps, pids, camids in queryloader:
            if use_gpu:
                imgs = imgs.cuda()
                heatmaps = heatmaps.cuda()

            b, s, c, h, w = imgs.size()
            features = model(imgs, heatmaps, pids, cam_label=camids)

            features = features.view(b, -1)
            features = torch.mean(features, dim=0)  # (b, feat_dim) → (feat_dim, )
            features = features.cpu()
            
            qf.append(features)
            q_pids.append(pids)
            q_camids.extend(camids)

        qf = torch.stack(qf)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        # ----- Gallery ----- #
        gf, g_pids, g_camids = [], [], []
        for imgs, heatmaps, pids, camids in galleryloader:
            if use_gpu:
                imgs = imgs.cuda()
                heatmaps = heatmaps.cuda()

            b, s, c, h, w = imgs.size()
            features = model(imgs, heatmaps, pids, cam_label=camids)
            
            features = features.view(b, -1)
            features = torch.mean(features, dim=0)  # (b, feat_dim) → (feat_dim, )
            features = features.cpu() 

            gf.append(features)
            g_pids.append(pids)
            g_camids.extend(camids)
    gf = torch.stack(gf)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    # ----- Distance matrix ----- #
    print("Computing distance matrix")
    m, n = qf.size(0), gf.size(0)
    distmat = (
        torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) +
        torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)

    distmat = distmat.numpy()
    gf = gf.numpy()
    qf = qf.numpy()

    print("Original Computing CMC and mAP")
    cmc, mean_ap = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    
    print("Results ---------- ")
    print("mAP: {:.1%} ".format(mean_ap))
    print("CMC curve r1:", cmc[0])
    
    return cmc[0], mean_ap


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Key-TransReID")
    parser.add_argument("--Dataset_name", default="Mars", help="The name of the DataSet", type=str)
    parser.add_argument('--ViT_path', default="/home/user/kim_js/ReID/KeyTransReID/weights/jx_vit_base_p16_224-80ecf9dd.pth", type=str, required=True, help='Path to the pre-trained Vision Transformer model')
    args = parser.parse_args()
    
    pretrainpath = str(args.ViT_path)
    Dataset_name = args.Dataset_name

    # ---- Set Seeds ----
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    # ---- Data & Model ----
    heatmap_train_loader, _, num_classes, camera_num, _, q_val_set, g_val_set = heatmap_dataloader(Dataset_name)

    model = Key_Trans(num_classes=num_classes, camera_num=camera_num, pretrainpath=pretrainpath)
    print("🚀 Running load_param")
    model.load_param(pretrainpath)
    
    loss_fun, center_criterion = make_loss(num_classes=num_classes)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=0.5)
    
    optimizer = optimizer(model)
    scheduler = scheduler(optimizer)
    scaler = amp.GradScaler()

    # ---- Train Setup ----
    device = "cuda"
    epochs = 120
    model = model.to(device)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    cmc_rank1 = 0
    map = 0
    loss_history = []
    loss_log_path = "/home/user/kim_js/ReID/KeyTransReID/loss/loss_log_best.txt"
    loss_graph_path = "/home/user/kim_js/ReID/KeyTransReID/loss/loss_plot_best.png"

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        scheduler.step(epoch)
        model.train()

        for Epoch_n, (imgs, heatmaps, pid, target_cam, labels2) in enumerate(heatmap_train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            
            imgs = imgs.to(device)
            heatmaps = heatmaps.to(device)
            pid = pid.to(device)
            target_cam = target_cam.to(device)
            labels2 = labels2.to(device)

            with amp.autocast(enabled=True):
                target_cam = target_cam.view(-1)
                score, feat, a_vals = model(imgs, heatmaps, pid, cam_label=target_cam)
                attn_noise = a_vals * labels2
                attn_loss = attn_noise.sum(dim=1).mean()
                loss_id, center = loss_fun(score, feat, pid, target_cam)
                loss = loss_id + 0.0005 * center + attn_loss

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

            # ---- Logging Loss ----
            loss_history.append(loss.item())
            if (Epoch_n+1) % 200 == 0:
                with open(loss_log_path, "a") as f:
                    f.write(f"Epoch {epoch}, Iteration {Epoch_n+1}, Loss: {loss.item():.6f}, Acc: {acc_meter.avg:.3f}\n")

            torch.cuda.synchronize()
            if (Epoch_n+1) % 400 == 0:
                print("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}".format(epoch, (Epoch_n+1), len(heatmap_train_loader), loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        # ---- Evaluation every 10 epochs ----
        if (epoch >= 20) and (epoch % 5) == 0:
            model.eval()
            cmc, mAP = test(model, q_val_set, g_val_set)
            print('CMC: %.4f, mAP : %.4f' % (cmc, mAP))

            save_dir = '/home/user/kim_js/ReID/KeyTransReID/evaluate'
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'matrix_best.txt')
            with open(save_path, 'a') as f:
                f.write(f'Epoch {epoch}: CMC = {cmc:.4f}, mAP = {mAP:.4f}\n')
            
            if cmc_rank1 < cmc:
                cmc_rank1 = cmc
                save_path = os.path.join(
                    '/home/user/kim_js/ReID/KeyTransReID/weights',
                    Dataset_name + 'best_CMC.pth'
                )
                torch.save(model.state_dict(), save_path)
            if map < mAP:
                map = mAP
                save_path = os.path.join(
                    '/home/user/kim_js/ReID/KeyTransReID/weights',
                    Dataset_name + 'best_mAP.pth'
                )
                torch.save(model.state_dict(), save_path)

    # ---- Plot & Save Loss Curve ----
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid()
    plt.savefig(loss_graph_path)
    plt.show()

    print(f"Loss logs have been saved: {loss_log_path}")
    print(f"The Loss graph has been saved: {loss_graph_path}")
