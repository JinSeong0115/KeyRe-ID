import numpy as np
import torch


def evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=21):
    """Compute CMC and mAP metrics from a distance matrix."""
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print(f"Note: number of gallery samples is quite small, got {num_g}")

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_ap = []
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
        tmp_cmc = np.asarray([x / (i + 1.0) for i, x in enumerate(tmp_cmc)])
        all_ap.append((tmp_cmc * orig_cmc).sum() / num_rel)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    return all_cmc.sum(0) / num_valid_q, np.mean(all_ap)


def _pool_features(features, batch_size, pool):
    features = features.view(batch_size, -1)
    if pool == "avg":
        return features.mean(dim=0)
    if pool == "max":
        return features.max(dim=0)[0]
    raise ValueError(f"Unsupported pool mode: {pool}")


def extract_features(model, dataloader, pool="avg", use_gpu=True):
    """Extract one pooled feature vector per video track."""
    features = []
    pids = []
    camids = []

    model.eval()
    with torch.no_grad():
        for imgs, heatmaps, pid, camid, _ in dataloader:
            if use_gpu:
                imgs = imgs.cuda(non_blocking=True)
                heatmaps = heatmaps.cuda(non_blocking=True)

            batch_size = imgs.size(0)
            output = model(imgs, heatmaps, pid, cam_label=camid)
            features.append(_pool_features(output, batch_size, pool).cpu())
            pids.append(pid)
            camids.extend(camid)

    return torch.stack(features, dim=0), np.asarray(pids), np.asarray(camids)


def compute_distance_matrix(query_features, gallery_features):
    """Compute squared Euclidean distances between query and gallery features."""
    m, n = query_features.size(0), gallery_features.size(0)
    distmat = (
        query_features.pow(2).sum(dim=1, keepdim=True).expand(m, n)
        + gallery_features.pow(2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    distmat.addmm_(query_features, gallery_features.t(), beta=1, alpha=-2)
    return distmat.numpy()


def test(model, queryloader, galleryloader, pool="avg", use_gpu=True):
    """Extract features, compute distances, and report rank-1 CMC/mAP."""
    qf, q_pids, q_camids = extract_features(model, queryloader, pool=pool, use_gpu=use_gpu)
    print(f"Extracted query features: {qf.size(0)} x {qf.size(1)}")

    gf, g_pids, g_camids = extract_features(model, galleryloader, pool=pool, use_gpu=use_gpu)
    print(f"Extracted gallery features: {gf.size(0)} x {gf.size(1)}")

    print("Computing distance matrix")
    distmat = compute_distance_matrix(qf, gf)

    print("Computing CMC and mAP")
    cmc, mean_ap = evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print(f"mAP: {mean_ap:.1%}")
    print("CMC curve r1:", cmc[0])
    print("CMC curve r5:", cmc[4])

    return cmc[0], cmc[4], mean_ap


# Backward-compatible alias for older imports.
evaluate = evaluate_rank
