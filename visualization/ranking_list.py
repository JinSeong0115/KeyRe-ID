import argparse
import random
import numpy as np
import torch
import os
from tqdm import tqdm
from heatmap_loader import heatmap_dataloader
from KeyRe_ID_model import KeyRe_ID
from torchreid.utils import visualize_ranked_results

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=21):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print(f"Note: gallery size ({num_g}) < max_rank, set max_rank = {num_g}")

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc, all_AP = [], []
    num_valid_q = 0

    for q_idx in range(num_q):
        order = indices[q_idx]
        remove = (g_pids[order] == q_pids[q_idx]) & (g_camids[order] == q_camids[q_idx])
        keep = np.invert(remove)
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1

        num_rel = orig_cmc.sum()
        tmp = orig_cmc.cumsum()
        tmp = np.asarray([x/(i+1) for i, x in enumerate(tmp)]) * orig_cmc
        AP = tmp.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    all_cmc = np.asarray(all_cmc).sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    return all_cmc, mAP

def test(model, queryloader, galleryloader, use_gpu=True,
         visrank=False, visrank_topk=10, save_dir=None):
    model.eval()
    with torch.no_grad():
        # Query features
        print("Extracting query features:")
        qf, q_pids, q_camids, q_img_paths = [], [], [], []
        for imgs, heatmaps, pids, camids, img_paths in tqdm(queryloader, desc="Query batches", unit="batch"):
            if use_gpu:
                imgs = imgs.to(device)
                heatmaps = heatmaps.to(device)
            b, s, c, h, w = imgs.size()
            feats = model(imgs, heatmaps, None, cam_label=camids)
            feats = feats.view(b, -1).mean(dim=0)  # GPU 텐서 유지
            qf.append(feats)
            q_pids.append(pids.cpu().numpy() if torch.is_tensor(pids) else pids)
            q_camids.extend(camids.cpu().numpy() if torch.is_tensor(camids) else camids)
            q_img_paths.append(img_paths[0])
        # 스택 후 GPU로 이동
        qf = torch.stack(qf).to(device)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        print("Query features done.\n")

        # Gallery features
        print("Extracting gallery features:")
        gf, g_pids, g_camids, g_img_paths = [], [], [], []
        for imgs, heatmaps, pids, camids, img_paths in tqdm(galleryloader, desc="Gallery batches", unit="batch"):
            if use_gpu:
                imgs = imgs.to(device)
                heatmaps = heatmaps.to(device)
            feats = model(imgs, heatmaps, None, cam_label=camids)
            feats = feats.view(imgs.size(0), -1).mean(dim=0)  # GPU 텐서 유지
            gf.append(feats)
            g_pids.append(pids.cpu().numpy() if torch.is_tensor(pids) else pids)
            g_camids.extend(camids.cpu().numpy() if torch.is_tensor(camids) else camids)
            g_img_paths.append(img_paths[0])
        # 스택 후 GPU로 이동
        gf = torch.stack(gf).to(device)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        print("Gallery features done.\n")

        # Distance matrix (GPU에서 계산)
        print("Computing distance matrix...")
        m, n = qf.size(0), gf.size(0)
        distmat = (
            torch.pow(qf, 2).sum(1, keepdim=True).expand(m, n) +
            torch.pow(gf, 2).sum(1, keepdim=True).expand(n, m).t()
        )
        distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
        distmat = distmat.cpu().numpy()  # NumPy 변환 전 CPU로 이동
        print("Distance matrix computed.\n")

        # Evaluation
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
        print(f"mAP: {mAP:.2%}, Rank-1: {cmc[0]:.2%}\n")

        # Prepare for visualization
        query = [(q_img_paths[i], int(q_pids[i]), int(q_camids[i])) for i in range(len(q_pids))]
        gallery = [(g_img_paths[i], int(g_pids[i]), int(g_camids[i])) for i in range(len(g_pids))]

        if visrank and save_dir:
            print("Visualizing ranked results...")
            os.makedirs(save_dir, exist_ok=True)
            visualize_ranked_results(
                distmat=distmat,
                dataset=(query, gallery),
                data_type='image',
                width=128, height=256,
                save_dir=save_dir,
                topk=visrank_topk
            )
            print(f"Visualization saved to {save_dir}\n")

    return cmc, mAP

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="KeyRe_ID 시각화 스크립트")
    parser.add_argument("--Dataset_name", type=str, default="Mars",
                        help="데이터셋 이름")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="파인튜닝 완료된 체크포인트 (.pth)")
    parser.add_argument("--visrank", action="store_true",
                        help="랭킹 결과 시각화 활성화")
    parser.add_argument("--visrank_topk", type=int, default=10,
                        help="시각화할 Top-K 개수")
    parser.add_argument("--save_dir", type=str, default="./visualization",
                        help="시각화 결과 저장 경로")
    args = parser.parse_args()

    # 시드 고정
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # 데이터 로더
    print(f"Loading dataset '{args.Dataset_name}'...")
    _, _, num_classes, camera_num, _, q_val_set, g_val_set = heatmap_dataloader(args.Dataset_name)
    print("Dataset loaded.\n")

    # 모델 생성 (pretrainpath=None 으로 ImageNet 가중치 스킵)
    print("Initializing model...")
    model = KeyRe_ID(num_classes=num_classes, camera_num=camera_num, pretrainpath=None)
    model = model.to(device)

    # 체크포인트 로드
    print(f"Loading checkpoint '{args.checkpoint}'...")
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    print("Checkpoint loaded.\n")

    # 테스트 및 시각화
    print("Starting test & visualization...\n")
    cmc, mAP = test(
        model,
        q_val_set,
        g_val_set,
        use_gpu=(device.type=="cuda"),
        visrank=args.visrank,
        visrank_topk=args.visrank_topk,
        save_dir=args.save_dir
    )
    print(f"Finished → Rank-1: {cmc[0]:.4f}, mAP: {mAP:.4f}")
