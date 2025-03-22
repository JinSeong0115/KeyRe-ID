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
# heatmap_loader의 heatmap_dataloader를 사용합니다.
from heatmap_loader import heatmap_dataloader  
from VID_Trans_model import VID_Trans
from Loss_fun import make_loss
from utility import AverageMeter, optimizer, scheduler
from token_fusion import FusionToken
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

def test(model, queryloader, galleryloader, device, camera_num, embed_dim=768, num_patches=128, pool='avg', use_gpu=True, ranks=[1, 5, 10, 20]):
    model.eval()
    # 평가용 패치 임베딩 및 FusionToken 모듈 생성 (이미지 크기 256x128 기준)
    img_patch_embed = PatchEmbed_overlap(img_size=(256, 128), patch_size=(16, 16), stride_size=16, in_chans=3, embed_dim=embed_dim).to(device)
    # 수정: heatmap의 채널 수는 6 (keypoint 그룹화 heatmap)로 처리
    heatmap_patch_embed = PatchEmbed_overlap(img_size=(256, 128), patch_size=(16, 16), stride_size=16, in_chans=6, embed_dim=embed_dim).to(device)
    fusion_module = FusionToken(num_patches=num_patches, embed_dim=embed_dim, cam_num=camera_num).to(device)
    
    qf, q_pids, q_camids = [], [], []
    with torch.no_grad():
        for batch_idx, (img, heatmap, pids, camids, labels, img_paths) in enumerate(queryloader):
            if use_gpu:
                img = img.to(device)
                heatmap = heatmap.to(device)
            b, t, c, h, w = img.size()
            BT = b * t
            # 패치 임베딩
            img_reshaped = img.view(BT, c, h, w)
            heatmap_reshaped = heatmap.view(BT, heatmap.size(2), h, w)
            img_tokens = img_patch_embed(img_reshaped)
            heatmap_tokens = heatmap_patch_embed(heatmap_reshaped)
            # cam 정보 flatten
            cam_labels_list = []
            for cam in camids:
                if isinstance(cam, (list, tuple)):
                    cam_labels_list.extend(cam)
                else:
                    cam_labels_list.append(cam)
            cam_labels = torch.tensor(cam_labels_list).to(device)
            # FusionToken으로 융합
            fusion_tokens = fusion_module(img_tokens, heatmap_tokens, cam_labels)
            num_tokens = fusion_tokens.shape[1]
            fused_inputs = fusion_tokens.view(b, t, num_tokens, embed_dim)
            # 모델에 fused 입력 전달
            features = model(fused_inputs, pids, cam_label=cam_labels)
            features = features.view(b, -1)
            features = torch.mean(features, dim=0)
            features = features.cpu()
            qf.append(features)
            q_pids.append(pids)
            for cam in camids:
                if isinstance(cam, (list, tuple)):
                    q_camids.extend(cam)
                else:
                    q_camids.append(cam)
        qf = torch.stack(qf)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))
        
        gf, g_pids, g_camids = [], [], []
        for batch_idx, (img, heatmap, pids, camids, labels, img_paths) in enumerate(galleryloader):
            if use_gpu:
                img = img.to(device)
                heatmap = heatmap.to(device)
            b, t, c, h, w = img.size()
            BT = b * t
            img_reshaped = img.view(BT, c, h, w)
            heatmap_reshaped = heatmap.view(BT, heatmap.size(2), h, w)
            img_tokens = img_patch_embed(img_reshaped)
            heatmap_tokens = heatmap_patch_embed(heatmap_reshaped)
            cam_labels_list = []
            for cam in camids:
                if isinstance(cam, (list, tuple)):
                    cam_labels_list.extend(cam)
                else:
                    cam_labels_list.append(cam)
            cam_labels = torch.tensor(cam_labels_list).to(device)
            fusion_tokens = fusion_module(img_tokens, heatmap_tokens, cam_labels)
            num_tokens = fusion_tokens.shape[1]
            fused_inputs = fusion_tokens.view(b, t, num_tokens, embed_dim)
            features = model(fused_inputs, pids, cam_label=cam_labels)
            features = features.view(b, -1)
            if pool == 'avg':
                features = torch.mean(features, dim=0)
            else:
                features, _ = torch.max(features, dim=0)
            features = features.cpu()
            gf.append(features)
            g_pids.append(pids)
            for cam in camids:
                if isinstance(cam, (list, tuple)):
                    g_camids.extend(cam)
                else:
                    g_camids.append(cam)
        gf = torch.stack(gf)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
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
    parser = argparse.ArgumentParser(description="VID-Trans-ReID")
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

    # heatmap_dataloader를 사용하여 이미지와 heatmap을 함께 로드
    train_loader, num_query, num_classes, camera_num, num_train, q_val_set, g_val_set = heatmap_dataloader(Dataset_name)
    print(f"✅ 현재 데이터셋의 camera_num: {camera_num}")

    model = VID_Trans(num_classes=num_classes, camera_num=camera_num, pretrainpath=pretrainpath)
    print("🚀 강제로 load_param 실행!")
    model.load_param(pretrainpath)

    loss_fun, center_criterion = make_loss(num_classes=num_classes)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=0.5)
    optim = optimizer(model)
    sched = scheduler(optim)
    scaler = amp.GradScaler()

    device = "cuda"
    epochs = 120
    model = model.to(device)

    # 학습에 사용할 패치 임베딩 및 FusionToken 모듈 (입력 이미지 크기 256x128, embed_dim=768, num_patches=128)
    embed_dim = 768
    num_patches = 128
    img_patch_embed = PatchEmbed_overlap(img_size=(256, 128), patch_size=(16, 16), stride_size=16, in_chans=3, embed_dim=embed_dim).to(device)
    # 수정: heatmap의 채널은 6로 처리 (Custom heatmap transform에 의해 정규화되어 -1~1 범위)
    heatmap_patch_embed = PatchEmbed_overlap(img_size=(256, 128), patch_size=(16, 16), stride_size=16, in_chans=6, embed_dim=embed_dim).to(device)
    fusion_module = FusionToken(num_patches=num_patches, embed_dim=embed_dim, cam_num=camera_num).to(device)

    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    cmc_rank1 = 0
    loss_history = []
    loss_log_path = "/home/user/kim_js/ReID/KeyTransReID/loss_log.txt"

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()

        sched.step(epoch)
        model.train()

        for Epoch_n, (img, heatmap, pid, target_cam, labels2, img_paths) in enumerate(train_loader):
            optim.zero_grad()
            optimizer_center.zero_grad()

            img = img.to(device)
            heatmap = heatmap.to(device)
            # pid가 tuple인 경우 tensor로 변환합니다.
            if not isinstance(pid, torch.Tensor):
                pid = torch.tensor(pid, dtype=torch.int64).to(device)
            else:
                pid = pid.to(device)
            # target_cam은 각 시퀀스별 리스트이므로 flatten 처리
            cam_labels_list = []
            for cam in target_cam:
                if isinstance(cam, (list, tuple)):
                    cam_labels_list.extend(cam)
                else:
                    cam_labels_list.append(cam)
            cam_labels = torch.tensor(cam_labels_list).to(device)
            labels2 = labels2.to(device)

            b, t, c, h, w = img.size()
            BT = b * t
            # 패치 임베딩 수행
            img_reshaped = img.view(BT, c, h, w)
            heatmap_reshaped = heatmap.view(BT, heatmap.size(2), h, w)
            img_tokens = img_patch_embed(img_reshaped)
            heatmap_tokens = heatmap_patch_embed(heatmap_reshaped)
            # FusionToken으로 토큰 융합
            fusion_tokens = fusion_module(img_tokens, heatmap_tokens, cam_labels)
            num_tokens = fusion_tokens.shape[1]
            fused_inputs = fusion_tokens.view(b, t, num_tokens, embed_dim)

            with amp.autocast(enabled=True):
                # fused_inputs를 모델의 입력으로 전달
                score, feat, a_vals = model(fused_inputs, pid, cam_label=cam_labels)
                attn_noise = a_vals * labels2
                attn_loss = attn_noise.sum(dim=1).mean()
                loss_id, center = loss_fun(score, feat, pid, target_cam=cam_labels)
                loss = loss_id + 0.0005 * center + attn_loss

            scaler.scale(loss).backward()
            scaler.step(optim)
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
            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            loss_history.append(loss.item())
            with open(loss_log_path, "a") as f:
                f.write(f"Epoch {epoch}, Iteration {Epoch_n + 1}, Loss: {loss.item():.6f}\n")

            torch.cuda.synchronize()
            if (Epoch_n + 1) % 50 == 0:
                print("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                      .format(epoch, (Epoch_n + 1), len(train_loader),
                              loss_meter.avg, acc_meter.avg, sched._get_lr(epoch)[0]))

        # ---- Evaluation every 10 epochs ----
        if (epoch + 1) % 10 == 0:
            model.eval()
            cmc, mAP = test(model, q_val_set, g_val_set, device, camera_num, embed_dim=embed_dim, num_patches=num_patches)
            print('CMC: %.4f, mAP : %.4f' % (cmc, mAP))
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
    plt.savefig("/home/user/kim_js/ReID/KeyTransReID/loss_plot.png")
    plt.show()

    print(f"✅ Loss 로그가 저장되었습니다: {loss_log_path}")
    print("✅ Loss 그래프가 저장되었습니다: /home/user/kim_js/ReID/KeyTransReID/loss_plot.png")
