import argparse
import os
import numpy as np
import torch

from heatmap_loader import heatmap_dataloader
from KeyRe_ID_model import KeyRe_ID


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=21):
    """Compute CMC and mAP metrics."""
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print(f"Note: number of gallery samples is quite small, got {num_g}")
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc, all_AP = [], []
    num_valid_q = 0.0
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
        num_valid_q += 1.0

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    return all_cmc, mAP


def test(model, queryloader, galleryloader, pool='avg', use_gpu=True):
    """Extract features and evaluate CMC/mAP."""
    model.eval()

    # Query feature extraction
    qf, q_pids, q_camids = [], [], []
    with torch.no_grad():
        for imgs, heatmaps, pids, camids, _ in queryloader:
            if use_gpu:
                imgs = imgs.cuda(non_blocking=True)
                heatmaps = heatmaps.cuda(non_blocking=True)
            b, s, c, h, w = imgs.size()
            feats = model(imgs, heatmaps, pids, cam_label=camids)
            feats = feats.view(b, -1)
            feats = feats.mean(dim=0) if pool == 'avg' else feats.max(dim=0)[0]
            qf.append(feats.cpu())
            q_pids.append(pids)
            q_camids.extend(camids)

    qf = torch.stack(qf, dim=0)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    print(f"Extracted query features: {qf.size(0)} x {qf.size(1)}")

    # Gallery feature extraction
    gf, g_pids, g_camids = [], [], []
    with torch.no_grad():
        for imgs, heatmaps, pids, camids, _ in galleryloader:
            if use_gpu:
                imgs = imgs.cuda(non_blocking=True)
                heatmaps = heatmaps.cuda(non_blocking=True)
            b, s, c, h, w = imgs.size()
            feats = model(imgs, heatmaps, pids, cam_label=camids)
            feats = feats.view(b, -1)
            feats = feats.mean(dim=0) if pool == 'avg' else feats.max(dim=0)[0]
            gf.append(feats.cpu())
            g_pids.append(pids)
            g_camids.extend(camids)

    gf = torch.stack(gf, dim=0)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    print(f"Extracted gallery features: {gf.size(0)} x {gf.size(1)}")

    # Compute distance matrix
    m, n = qf.size(0), gf.size(0)
    distmat = (
        qf.pow(2).sum(dim=1, keepdim=True).expand(m, n)
        + gf.pow(2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    distmat = distmat.numpy()

    # Evaluate
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    print("Results ----------")
    print(f"mAP: {mAP:.1%}")
    print("CMC curve r1:", cmc[0])
    return cmc[0], mAP


def main():
    parser = argparse.ArgumentParser(description="KeyRe-ID Test")
    parser.add_argument("--dataset_name", default="MARS", type=str)
    parser.add_argument("--dataset_root", default="./data", type=str,
                        help="Path to dataset root directory")
    parser.add_argument("--ViT_path", required=True, type=str,
                        help="Path to pretrained ViT weights")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--eval_pool", default="avg", choices=["avg", "max"])
    args = parser.parse_args()

    use_gpu = (args.device == "cuda") and torch.cuda.is_available()

    # Validate paths
    if not os.path.exists(args.ViT_path):
        raise ValueError(f"Invalid ViT_path: {args.ViT_path}")
    if not os.path.exists(args.dataset_root):
        raise ValueError(f"Invalid dataset_root: {args.dataset_root}")

    # Data loading
    _, _, num_classes, camera_num, _, q_val_set, g_val_set = \
        heatmap_dataloader(args.dataset_name, args.dataset_root)

    # Model setup
    model = KeyRe_ID(num_classes=num_classes, camera_num=camera_num, pretrainpath=args.ViT_path)
    print("🚀 Running load_param")
    model.load_param(args.ViT_path)

    if use_gpu:
        model = model.cuda()
    model.eval()

    # Run evaluation
    cmc, mAP = test(model, q_val_set, g_val_set, pool=args.eval_pool, use_gpu=use_gpu)
    print("CMC: %.4f, mAP: %.4f" % (cmc, mAP))


if __name__ == "__main__":
    main()
