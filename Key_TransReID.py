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

# heatmap_loader.py에서 이미지와 heatmap 데이터를 모두 load하는 함수를 사용
from heatmap_loader import heatmap_dataloader, Heatmap_Dataset, Heatmap_collate_fn, custom_collate_fn
from Key_Trans_model import KeyTransReID 
from vit_ID import PatchEmbed_overlap
from Loss_fun import make_loss
from utility import AverageMeter, optimizer, scheduler

def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=21, device="cuda"):
    # distmat은 이미 NumPy 배열로 들어옴
    q_pids = q_pids  # 이미 NumPy 배열
    g_pids = g_pids  # 이미 NumPy 배열
    q_camids = q_camids  # 이미 NumPy 배열
    g_camids = g_camids  # 이미 NumPy 배열
    
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
        keep = ~remove

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

def test(model, query_loader, gallery_loader, img_patch_embed, heatmap_patch_embed, pool='avg', use_gpu=True, device="cuda"):
    def masked_average_pooling(features, mask):
        """
        features: [B, total_frames, embed_dim]
        mask: [B, total_frames] (유효 프레임은 1, 패딩은 0)
        """
        mask = mask.unsqueeze(-1)  # [B, total_frames, 1]
        features_sum = (features * mask).sum(dim=1)
        valid_counts = mask.sum(dim=1)  # [B, 1]
        pooled = features_sum / (valid_counts + 1e-6)
        return pooled

    model.eval()
    with torch.no_grad():
        # ----- Query ----- #
        qf, q_pids, q_camids = [], [], []
        for imgs, heatmaps, pids, camids, img_paths, masks in query_loader:
            if use_gpu:
                imgs = imgs.cuda()         # [B, num_clips, seq_len, 3, H, W]
                heatmaps = heatmaps.cuda() # [B, num_clips, seq_len, 6, H, W]
                masks = masks.cuda()       # [B, num_clips]

            B, num_clips, seq_len, c, h, w = imgs.size()
            BT = B * num_clips * seq_len  # 총 프레임 수

            images_reshaped = imgs.view(BT, c, h, w)  # [BT, 3, H, W]
            img_tokens = img_patch_embed(images_reshaped)  # [BT, num_patches, embed_dim]

            # heatmaps: [B, num_clips, seq_len, 6, H, W] → [BT, 6, H, W]
            _, _, _, C_h, H, W = heatmaps.size()
            heatmaps_reshaped = heatmaps.view(BT, C_h, H, W)
            heatmap_tokens = heatmap_patch_embed(heatmaps_reshaped)  # [BT, num_patches, embed_dim]

            # 각 샘플의 카메라 ID를 총 프레임 수만큼 반복 (예: [cam, cam, ..., cam])
            cam_labels_list = []
            for cam in camids:
                # camids가 각 샘플당 단일 값이라고 가정
                cam_labels_list.extend([cam] * (num_clips * seq_len))
            cam_labels = torch.tensor(cam_labels_list).to(img_tokens.device)  # 길이: BT

            fusion_tokens = model.fusion_module(img_tokens, heatmap_tokens, cam_labels)
            # fusion_tokens: [BT, 1+num_patches, embed_dim] → 첫 토큰(CLS) 추출
            cls_tokens = fusion_tokens[:, 0, :].view(B, num_clips * seq_len, -1)  # [B, total_frames, embed_dim]

            # masks: [B, num_clips] → 각 클립당 seq_len으로 확장하여 [B, total_frames]
            masks_expanded = masks.unsqueeze(-1).expand(B, masks.size(1), seq_len).contiguous().view(B, -1)

            features = masked_average_pooling(cls_tokens, masks_expanded)  # [B, embed_dim]

            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)

        qf = torch.cat(qf, dim=0)
        q_pids = np.array(q_pids)
        q_camids = np.array(q_camids)
        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        # ----- Gallery ----- #
        gf_list, g_pids, g_camids = [], [], []
        for n, (imgs, heatmaps, pids, camids, img_paths, masks) in enumerate(gallery_loader):
            print(f"Processing gallery sample {n}/{len(gallery_loader)}")
            if use_gpu:
                imgs = imgs.cuda()         # [B, num_clips, seq_len, 3, H, W]
                heatmaps = heatmaps.cuda() # [B, num_clips, seq_len, 6, H, W]
                masks = masks.cuda()       # [B, num_clips]

            B, num_clips, seq_len, c, h, w = imgs.size()
            BT = B * num_clips * seq_len

            images_reshaped = imgs.view(BT, c, h, w)
            img_tokens = img_patch_embed(images_reshaped)

            _, _, _, C_h, H, W = heatmaps.size()
            heatmaps_reshaped = heatmaps.view(BT, C_h, H, W)
            heatmap_tokens = heatmap_patch_embed(heatmaps_reshaped)

            cam_labels_list = []
            for cam in camids:
                cam_labels_list.extend([cam] * (num_clips * seq_len))
            cam_labels = torch.tensor(cam_labels_list).to(img_tokens.device)

            fusion_tokens = model.fusion_module(img_tokens, heatmap_tokens, cam_labels)
            cls_tokens = fusion_tokens[:, 0, :].view(B, num_clips * seq_len, -1)  # [B, total_frames, embed_dim]

            masks_expanded = masks.unsqueeze(-1).expand(B, masks.size(1), seq_len).contiguous().view(B, -1)
            if pool == 'avg':
                features = masked_average_pooling(cls_tokens, masks_expanded)
            else:
                features, _ = torch.max(cls_tokens, dim=1)
            gf_list.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)

        gf = torch.cat(gf_list, dim=0)
        g_pids = np.array(g_pids)
        g_camids = np.array(g_camids)
        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

        torch.cuda.empty_cache()

        # ----- Distance matrix ----- #
        print("Computing distance matrix")
        qf = qf.to(device)
        gf = gf.to(device)
        m, n = qf.size(0), gf.size(0)
        batch_size_dist = 128
        distmat_list = []
        for i in range(0, m, batch_size_dist):
            q_batch = qf[i:i+batch_size_dist]
            d_batch = torch.pow(q_batch, 2).sum(dim=1, keepdim=True) + \
                      torch.pow(gf, 2).sum(dim=1, keepdim=True).t() - \
                      2 * torch.mm(q_batch, gf.t())
            distmat_list.append(d_batch)
        distmat = torch.cat(distmat_list, dim=0)
        distmat = distmat.cpu().numpy()

        print("Original Computing CMC and mAP")
        cmc, mean_ap = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
        print("Results ----------")
        print("mAP: {:.1%}".format(mean_ap))
        print("CMC curve r1:", cmc[0])

    return cmc[0], mean_ap

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Key Trans-ReID Training")
    parser.add_argument("--Dataset_name", default="Mars", help="The name of the DataSet", type=str)
    parser.add_argument('--ViT_path', default="/home/user/kim_js/ReID/VidTansReID/jx_vit_base_p16_224-80ecf9dd.pth", type=str, required=True, help='Path to the pre-trained Vision Transformer model')
    args = parser.parse_args()
    
    pretrainpath = str(args.ViT_path)
    Dataset_name = args.Dataset_name

    heatmap_train_loader, num_query, num_classes, camera_num, num_train, q_val_set, g_val_set = heatmap_dataloader(Dataset_name)
    
    # ---- Set Seeds ----
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    # Query, Gallery DataLoader 생성
    query_loader = torch.utils.data.DataLoader(
        q_val_set, batch_size=8, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn
    )
    gallery_loader = torch.utils.data.DataLoader(
        g_val_set, batch_size=8, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn
    )
    
    # 패치 임베딩 모듈 생성 (embed_dim=768)
    img_patch_embed = PatchEmbed_overlap(img_size=(256, 128), patch_size=(16, 16), stride_size=16, in_chans=3, embed_dim=768)
    # heatmap의 채널 수는 6으로 설정
    heatmap_patch_embed = PatchEmbed_overlap(img_size=(256, 128), patch_size=(16, 16), stride_size=16, in_chans=6, embed_dim=768)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_patch_embed = img_patch_embed.to(device)
    heatmap_patch_embed = heatmap_patch_embed.to(device)

    # 모델 초기화 (KeyTransReID)
    model = KeyTransReID(num_classes=num_classes, camera_num=camera_num, pretrainpath=pretrainpath)
    print("🚀 Running load_param")
    model.load_param(pretrainpath)
    
    # 수정한 loss function을 사용하여 body part loss가 포함된 loss 함수 생성 (Loss_fun.py 수정본 참조)
    loss_fun, center_criterion = make_loss(num_classes=num_classes)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=0.0001)
    
    optimizer = optimizer(model)
    scheduler = scheduler(optimizer)
    scaler = amp.GradScaler()
    
    # ---- Train Setup ----
    epochs = 120
    model = model.to(device)
    
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()    
    
    cmc_rank1 = 0
    loss_history = []
    loss_log_path = "/home/user/kim_js/ReID/KeyTransReID/loss/loss_log.txt"
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        
        scheduler.step(epoch)
        model.train()  
        
        for Epoch_n, (imgs, heatmaps, pid, target_cam, labels2, img_paths) in enumerate(heatmap_train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            
            imgs = imgs.to(device)
            heatmaps = heatmaps.to(device)
            pid = torch.tensor(pid, dtype=torch.long).to(device)
            target_cam = torch.tensor(target_cam, dtype=torch.long).to(device)
            labels2 = labels2.to(device)
            
            b, s, c, h, w = imgs.size()
            BT = b * s
            images_reshaped = imgs.view(BT, c, h, w)
            img_tokens = img_patch_embed(images_reshaped)  # [B*T, 128, 768]
            _, _, C_h, H, W = heatmaps.shape
            heatmaps_reshaped = heatmaps.view(BT, C_h, H, W)
            heatmap_tokens = heatmap_patch_embed(heatmaps_reshaped)  # [B*T, 128, 768]
            cam_labels = target_cam.view(-1)
            
            fusion_tokens = model.fusion_module(img_tokens, heatmap_tokens, cam_labels)
            
            with amp.autocast(enabled=True):
                target_cam = target_cam.view(-1)
                score, feat, a_vals = model(fusion_tokens, cam_label=target_cam)
                
                if isinstance(score, list):
                    score = [s.float() for s in score]
                else:
                    score = score.float()
                
                # attn_loss 계산: a_vals와 labels2의 곱을 배치별로 합산하고 평균내어 계산
                attn_noise = a_vals * labels2
                attn_loss = attn_noise.sum(dim=1).mean()
                
                # 수정한 loss_fun을 호출하여, 추가적으로 part_scores와 part_targets를 전달
                loss_id, center = loss_fun(score, feat, pid, target_cam)
                loss = loss_id + 0.00005 * center + attn_loss

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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
            with open(loss_log_path, "a") as f:
                f.write(f"Epoch {epoch}, Iteration, Loss: {loss.item():.6f}\n")
            if (Epoch_n + 1) % 100 == 0:
                print("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                      .format(epoch, (Epoch_n + 1), len(heatmap_train_loader),
                              loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))
                torch.cuda.synchronize()
        
        # ---- Evaluation every 10 epochs ----
        print("Epoch [{}] Loss: {:.3f}, Acc: {:.3f}, Time: {:.1f}s".format(
            epoch, loss_meter.avg, acc_meter.avg, time.time() - start_time
        ))
        
        torch.cuda.empty_cache()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            cmc, mAP = test(model, query_loader, gallery_loader, img_patch_embed, heatmap_patch_embed)
            if cmc_rank1 < cmc:
                cmc_rank1 = cmc
                save_path = os.path.join(
                    '/home/user/kim_js/ReID/KeyTransReID/weights',
                    Dataset_name + 'Main_Model.pth'
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
    plt.savefig("./loss_plot.png")
    plt.show()
    
    print(f"✅ Loss 로그 저장: {loss_log_path}")
    print("✅ Loss 그래프 저장: ./loss_plot.png")




# def test(model, query_loader, gallery_loader, img_patch_embed, heatmap_patch_embed, pool='avg', use_gpu=True):
#     model.eval()
#     with torch.no_grad():
#         # ----- Query ----- #
#         qf, q_pids, q_camids = [], [], []
#         for imgs, heatmaps, pids, camids, img_paths in query_loader:
#             if use_gpu:
#                 imgs = imgs.cuda()
#                 heatmaps = heatmaps.cuda()
                
#             _, b, t, c, h, w = imgs.size()
#             BT = b * t
            
#             images_reshaped = imgs.view(BT, c, h, w)
#             img_tokens = img_patch_embed(images_reshaped)  # [B*T, 128, 768]
            
#             # heatmaps: [B, T, 6, H, W] → reshape: [BT, 6, H, W]
#             _, b, t, C_h, H, W = heatmaps.shape  # heatmaps: [B, T, 6, H, W]
#             heatmaps_reshaped = heatmaps.view(BT, C_h, H, W)
#             heatmap_tokens = heatmap_patch_embed(heatmaps_reshaped)  # [B*T, 128, 768]
            
#             # cam_ids flatten 처리
#             cam_labels_list = []
#             for cam in camids:
#                 if isinstance(cam, list):
#                     cam_labels_list.extend(cam)
#                 else:
#                     cam_labels_list.append(cam)
#             cam_labels = torch.tensor(cam_labels_list).to(img_tokens.device)
            
#             fusion_tokens = model.fusion_module(img_tokens, heatmap_tokens, cam_labels)
            
#             # Transformer 기반 모델에서는 첫 번째 CLS 토큰을 대표 피처로 사용
#             cls_tokens = fusion_tokens[:, 0, :].view(b, t, -1)  # [B, T, embed_dim]
            
#             # 시퀀스 내 평균 풀링을 통해 각 샘플별 하나의 특징 벡터로 집약
#             features = torch.mean(cls_tokens, dim=1)  # [B, embed_dim]
            
#             qf.append(features)
#             q_pids.append(pids)
#             q_camids.append(camids) 
        
#         qf = torch.cat(qf, dim=0)
#         q_pids = np.asarray(q_pids[0])
#         q_camids = np.asarray(q_camids[0])
#         print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))
        
#         # ----- Gallery ----- #
#         gf_list, g_pids, g_camids = [], [], []
#         for n, (imgs, heatmaps, pids, camids, img_paths) in enumerate(gallery_loader):
#             print(f"실행 중 {n}/{len(gallery_loader)}")
#             if use_gpu:
#                 imgs = imgs.cuda()
#                 heatmaps = heatmaps.cuda()
            
#             _, b, s, c, h, w = imgs.size()
#             BT = b * s

#             images_reshaped = imgs.view(BT, c, h, w)
#             img_tokens = img_patch_embed(images_reshaped)  # [B*T, 128, 768]
            
#             # heatmaps: [B, S, 6, H, W] → reshape: [BT, 6, H, W]
#             _, b, s, C_h, H, W = heatmaps.shape  # heatmaps: [B, S, 6, H, W]
#             heatmaps_reshaped = heatmaps.view(BT, C_h, H, W)
#             heatmap_tokens = heatmap_patch_embed(heatmaps_reshaped)  # [B*T, 128, 768]
            
#             cam_labels_list = []
#             for cam in camids:
#                 if isinstance(cam, list):
#                     cam_labels_list.extend(cam)
#                 else:
#                     cam_labels_list.append(cam)
#             cam_labels = torch.tensor(cam_labels_list).to(img_tokens.device)
            
#             fusion_tokens = model.fusion_module(img_tokens, heatmap_tokens, cam_labels)  # [BT, 1+num_patches, embed_dim]
#             cls_tokens = fusion_tokens[:, 0, :].view(b, s, -1)  # [B, S, embed_dim]
            
#             if pool == 'avg':
#                 features = torch.mean(cls_tokens, dim=1)  # [B, embed_dim]
#             else:
#                 features, _ = torch.max(cls_tokens, dim=1)  # [B, embed_dim]
            
#             gf_list.append(features)
#             g_pids.append(pids)
#             g_camids.append(camids)
        
#         print("continue")
#         gf = torch.cat(gf_list, dim=0)  # [num_gallery, embed_dim]
#         g_pids = np.asarray(g_pids[0])
#         g_camids = np.asarray(g_camids[0])

#         print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

#         # ----- Distance matrix ----- #
#         print("Computing distance matrix")
#         qf = qf.to(device)
#         gf = gf.to(device)
#         m, n = qf.size(0), gf.size(0)
#         batch_size_dist = 128
#         distmat_list = []
#         for i in range(0, m, batch_size_dist):
#             q_batch = qf[i:i+batch_size_dist]  # [batch_size, d]
#             d_batch = torch.pow(q_batch, 2).sum(dim=1, keepdim=True) + \
#                       torch.pow(gf, 2).sum(dim=1, keepdim=True).t() - \
#                       2 * torch.mm(q_batch, gf.t())
#             distmat_list.append(d_batch)
#         distmat = torch.cat(distmat_list, dim=0)
#         distmat = distmat.cpu().numpy()

#         print("Original Computing CMC and mAP")
                
#         # ----- Calculating metrics -----
#         cmc, mean_ap = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
#         print("Results ----------")
#         print("mAP: {:.1%}".format(mean_ap))
#         print("CMC curve r1:", cmc[0])
        
#     return cmc[0], mean_ap

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Key Trans-ReID Training")
#     parser.add_argument("--Dataset_name", default="Mars", help="The name of the DataSet", type=str)
#     parser.add_argument('--ViT_path', default="/home/user/kim_js/ReID/VidTansReID/jx_vit_base_p16_224-80ecf9dd.pth", type=str, required=True, help='Path to the pre-trained Vision Transformer model')
#     args = parser.parse_args()
    
#     pretrainpath = str(args.ViT_path)
#     Dataset_name = args.Dataset_name

#     # 1) Dataloader 준비
#     heatmap_train_loader, num_query, num_classes, camera_num, num_train, q_val_set, g_val_set = heatmap_dataloader(Dataset_name)
    
#     # 2) 데이터셋의 일부만 추출 (예: 처음 32개 샘플만 사용)
#     small_q_val_set = torch.utils.data.Subset(q_val_set, range(32))  # 쿼리 샘플 32개
#     small_g_val_set = torch.utils.data.Subset(g_val_set, range(32))  # 갤러리 샘플 32개
    
#     # 3) Query, Gallery Loader 준비
#     query_loader = torch.utils.data.DataLoader(
#         small_q_val_set, batch_size=8, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn
#     )
#     gallery_loader = torch.utils.data.DataLoader(
#         small_g_val_set, batch_size=8, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn
#     )
    
#     # 4) 모델 준비
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = KeyTransReID(num_classes=num_classes, camera_num=camera_num, pretrainpath=pretrainpath)
#     model.load_param(pretrainpath)
#     model = model.to(device)
    
#     # 5) Patch embedding 모듈 준비
#     img_patch_embed = PatchEmbed_overlap(img_size=(256, 128), patch_size=(16, 16), stride_size=16, in_chans=3, embed_dim=768)
#     heatmap_patch_embed = PatchEmbed_overlap(img_size=(256, 128), patch_size=(16, 16), stride_size=16, in_chans=6, embed_dim=768)
#     img_patch_embed = img_patch_embed.to(device)
#     heatmap_patch_embed = heatmap_patch_embed.to(device)
    
#     # 6) 테스트 함수 호출
#     cmc, mAP = test(model, query_loader, gallery_loader, img_patch_embed, heatmap_patch_embed)
#     print("Test-only run. cmc =", cmc, " mAP =", mAP)
