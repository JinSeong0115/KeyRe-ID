import os
import argparse
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.cuda import amp
from torch_ema import ExponentialMovingAverage
from heatmap_loader import heatmap_dataloader
from Key_Trans_model import KeyTransReID 
from Loss_fun import make_loss
from utility import AverageMeter, optimizer, scheduler
from vit_ID import PatchEmbed_overlap

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

def test(model, queryloader, galleryloader, img_patch_embed, heatmap_patch_embed, pool='avg', use_gpu=True):
    model.eval()
    with torch.no_grad():
        # ----- Query ----- #
        qf, q_pids, q_camids = [], [], []
        for imgs, heatmaps, pids, camids in queryloader:
            if use_gpu:
                imgs = imgs.cuda()
                heatmaps = heatmaps.cuda()

            b, s, c, h, w = imgs.size()
            BT = b * s
            images_reshaped = imgs.view(BT, c, h, w)
            img_tokens = img_patch_embed(images_reshaped)
            heatmaps_reshaped = heatmaps.view(BT, heatmaps.size(2), h, w)
            heatmap_tokens = heatmap_patch_embed(heatmaps_reshaped)
            
            fusion_tokens = model.fusion_module(img_tokens, heatmap_tokens)
            
            features = model(fusion_tokens, heatmaps, cam_label=camids)
            features = features.view(b, -1)  # [B, feature_dim]
            
            if pool == 'avg':
                features = torch.mean(features, dim=0)  # [feature_dim]
            else:
                features, _ = torch.max(features, dim=0)  # [feature_dim]
                
            qf.append(features)
            q_pids.append(pids)
            q_camids.extend(camids)

        qf = torch.stack(qf).cuda()
        q_pids = np.asarray(q_pids, dtype=np.int64)
        q_camids = np.asarray(q_camids, dtype=np.int64)
        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        # ----- Gallery ----- #
        gf, g_pids, g_camids = [], [], []
        for imgs, heatmaps, pids, camids in galleryloader:
            if use_gpu:
                imgs = imgs.cuda()
                heatmaps = heatmaps.cuda()

            b, s, c, h, w = imgs.size()
            BT = b * s
            images_reshaped = imgs.view(BT, c, h, w)
            img_tokens = img_patch_embed(images_reshaped)
            heatmaps_reshaped = heatmaps.view(BT, heatmaps.size(2), h, w)
            heatmap_tokens = heatmap_patch_embed(heatmaps_reshaped)
            
            fusion_tokens = model.fusion_module(img_tokens, heatmap_tokens)
            
            features = model(fusion_tokens, heatmaps, cam_label=camids)
            features = features.view(b, -1)  # [B, feature_dim]
            
            if pool == 'avg':
                features = torch.mean(features, dim=0)  # [feature_dim]
            else:
                features, _ = torch.max(features, dim=0)  # [feature_dim]
                
            gf.append(features)
            g_pids.append(pids)
            g_camids.extend(camids)

        # 모든 갤러리 특징 텐서를 연결
        gf = torch.stack(gf).cuda()
        g_pids = np.asarray(g_pids, dtype=np.int64)
        g_camids = np.asarray(g_camids, dtype=np.int64)
        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

        # ----- Distance matrix calculation ----- #
        distmat = torch.cdist(qf, gf, p=2).cpu().numpy()  # euclidean distance
        
        # ----- CMC, mAP calculation ----- #
        print("Original Computing CMC and mAP")
        cmc, mean_ap = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
        
        print("Results ---------- ")
        print("mAP: {:.1%} ".format(mean_ap))
        print("CMC curve r1:", cmc[0])
        
        return cmc[0], mean_ap

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Key Trans-ReID Training")
    parser.add_argument("--Dataset_name", default="Mars", help="The name of the DataSet", type=str)
    parser.add_argument('--ViT_path', default="/home/user/kim_js/ReID/VidTansReID/jx_vit_base_p16_224-80ecf9dd.pth", type=str, required=True, help='Path to the pre-trained Vision Transformer model')
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
    
    # Data & Model
    heatmap_train_loader, num_query, num_classes, camera_num, num_train, q_val_set, g_val_set = heatmap_dataloader(Dataset_name)
    
    print(f"✅ 현재 데이터셋의 camera_num: {camera_num}")
    
    model = KeyTransReID(num_classes=num_classes, camera_num=camera_num, pretrainpath=pretrainpath)
    print("🚀 Running load_param")
    model.load_param(pretrainpath)
    
    loss_fun, center_criterion = make_loss(num_classes=num_classes)
    # optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=0.5)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=0.0001)

    optimizer = optimizer(model)
    scheduler = scheduler(optimizer)
    scaler = amp.GradScaler()
      
    # ---- Train Setup ----
    epochs = 120
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()    
    cmc_rank1 = 0
    loss_history = []
    loss_log_path = "/home/user/kim_js/ReID/KeyTransReID/loss/loss_log.txt"
    
    # Patch Embedding Modules
    img_patch_embed = PatchEmbed_overlap(img_size=(256, 128), patch_size=(16, 16), stride_size=16, in_chans=3, embed_dim=768)
    heatmap_patch_embed = PatchEmbed_overlap(img_size=(256, 128), patch_size=(16, 16), stride_size=16, in_chans=6, embed_dim=768)
    img_patch_embed = img_patch_embed.to(device)
    heatmap_patch_embed = heatmap_patch_embed.to(device)
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        scheduler.step(epoch)
        model.train()  
        
        for Epoch_n, (imgs, heatmaps, pid, camids, erasing_labels) in enumerate(heatmap_train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()

            imgs = imgs.to(device)
            heatmaps = heatmaps.to(device)
            pid = pid.to(device)
            camids = camids.to(device)
            erasing_labels = erasing_labels.to(device)

            b, s, c, h, w = imgs.size()
            BT = b * s
            images_reshaped = imgs.view(BT, c, h, w)
            img_tokens = img_patch_embed(images_reshaped)
            heatmaps_reshaped = heatmaps.view(BT, heatmaps.size(2), h, w)
            heatmap_tokens = heatmap_patch_embed(heatmaps_reshaped)
            cam_labels = camids.clone().detach().view(BT).to(device)
            
            fusion_tokens = model.fusion_module(img_tokens, heatmap_tokens)

            with amp.autocast(enabled=True):
                score, feat, a_vals = model(fusion_tokens, heatmaps, cam_label=cam_labels)
                attn_noise = a_vals * erasing_labels
                attn_loss = attn_noise.sum(dim=1).mean()
                loss_id, center = loss_fun(score, feat, pid)
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

            loss_history.append(loss.item())
            if (Epoch_n + 1) % 100 == 0:
                with open(loss_log_path, "a") as f:
                    f.write(f"Epoch {epoch}, Iteration {Epoch_n + 1}, Loss: {loss.item():.6f}, Acc: {acc_meter.avg:.3f}\n")
            torch.cuda.synchronize()
            if (Epoch_n + 1) % 200 == 0:
                print(f"Epoch[{epoch}] Iteration[{Epoch_n + 1}/{len(heatmap_train_loader)}] Loss: {loss_meter.avg:.3f}, Acc: {acc_meter.avg:.3f}, Base Lr: {scheduler._get_lr(epoch)[0]:.2e}")

        if (epoch + 1) % 10 == 0:
            model.eval()
            cmc, mAP = test(model, q_val_set, g_val_set, img_patch_embed, heatmap_patch_embed)
            print(f'CMC: {cmc:.4f}, mAP: {mAP:.4f}')
            
            save_dir = '/home/user/kim_js/ReID/KeyTransReID/evaluate'
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'matrix.txt')
            with open(save_path, 'a') as f:
                f.write(f'Epoch {epoch + 1}: CMC = {cmc:.4f}, mAP = {mAP:.4f}\n')
                
            if cmc_rank1 < cmc:
                cmc_rank1 = cmc
                save_path = os.path.join('/home/user/kim_js/ReID/KeyTransReID/weights', f'{Dataset_name}_Main_Model.pth')
                torch.save(model.state_dict(), save_path)

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid()
    plt.savefig("/home/user/kim_js/ReID/KeyTransReID/loss_plot.png")
    plt.show()

    print(f"✅ Loss 로그가 저장되었습니다: {loss_log_path}")
    print("✅ Loss 그래프가 저장되었습니다: /home/user/kim_js/ReID/KeyTransReID/loss_plot.png")