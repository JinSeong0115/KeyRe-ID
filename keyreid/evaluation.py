"""Tracklet descriptors and retrieval metrics (CMC, mAP).

A tracklet is cut into consecutive non-overlapping 4-frame clips and the clip embeddings are
averaged (Section 3.1). The retrieval descriptor (Eq. 17 of the paper) is

    d = [ sqrt(w) * e_g / ||e_g||_2  ||  sqrt(1 - w) * e_p / ||e_p||_2 ],   w = 0.5,

with e_g the global embedding (768-d) and e_p the average of the K part embeddings (3,072-d),
giving a 3,840-dimensional descriptor ranked by Euclidean distance.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

GLOBAL_DIM = 768
PART_DIM = 3072


def _identity(x):
    return x


def extract_features(model, dataset, device='cuda', num_workers=4):
    """One averaged clip embedding per tracklet: [N, 768 + K*3072]."""
    loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=num_workers, collate_fn=_identity)
    features, pids, camids = [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, pid, camid, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            output = model(imgs, cam_label=camid)
            features.append(output.view(imgs.size(0), -1).mean(dim=0).cpu())
            pids.append(pid)
            camids.extend(camid)
    return torch.stack(features, dim=0), np.asarray(pids), np.asarray(camids)


def _l2n(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def build_descriptor(features, num_parts, w=0.5):
    """[N, 768 + K*3072] tracklet embeddings -> [N, 3840] retrieval descriptors."""
    x = features.numpy() if torch.is_tensor(features) else features
    g = x[:, :GLOBAL_DIM]
    parts = [x[:, GLOBAL_DIM + k * PART_DIM: GLOBAL_DIM + (k + 1) * PART_DIM] for k in range(num_parts)]
    p = np.mean(parts, 0)
    wg, wp = np.float32(np.sqrt(w)), np.float32(np.sqrt(1.0 - w))
    return np.concatenate([wg * _l2n(g), wp * _l2n(p)], 1)


def compute_distance_matrix(query_features, gallery_features):
    """Squared Euclidean distances."""
    m, n = query_features.size(0), gallery_features.size(0)
    distmat = (query_features.pow(2).sum(dim=1, keepdim=True).expand(m, n)
               + gallery_features.pow(2).sum(dim=1, keepdim=True).expand(n, m).t())
    distmat.addmm_(query_features, gallery_features.t(), beta=1, alpha=-2)
    return distmat.numpy()


def evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=21):
    """CMC curve and mAP; gallery items sharing identity and camera with the query are skipped."""
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc, all_ap = [], []
    num_valid_q = 0.0
    for q_idx in range(num_q):
        q_pid, q_camid = q_pids[q_idx], q_camids[q_idx]
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
    assert num_valid_q > 0, 'no query identity appears in the gallery'
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    return all_cmc.sum(0) / num_valid_q, np.mean(all_ap)


def evaluate(model, query_set, gallery_set, num_parts, w=0.5, device='cuda', num_workers=4):
    """Returns Rank-1, Rank-5 and mAP (fractions) with the 3,840-d descriptor."""
    qf, q_pids, q_camids = extract_features(model, query_set, device, num_workers)
    gf, g_pids, g_camids = extract_features(model, gallery_set, device, num_workers)
    qd = build_descriptor(qf, num_parts, w)
    gd = build_descriptor(gf, num_parts, w)
    distmat = compute_distance_matrix(torch.from_numpy(np.ascontiguousarray(qd)),
                                      torch.from_numpy(np.ascontiguousarray(gd)))
    cmc, mAP = evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids)
    return float(cmc[0]), float(cmc[4]), float(mAP)
