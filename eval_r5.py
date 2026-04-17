"""Re-evaluate saved best models to extract R-1, R-5, R-10, mAP."""
import os, sys, argparse, numpy as np, torch
REPO_ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, REPO_ROOT)
from heatmap_loader import heatmap_dataloader
from KeyRe_ID_model import KeyRe_ID
from evaluation import extract_features, compute_distance_matrix, evaluate_rank

def test_full_cmc(model, q_set, g_set, pool="avg", use_gpu=True):
    qf, q_pids, q_camids = extract_features(model, q_set, pool=pool, use_gpu=use_gpu)
    gf, g_pids, g_camids = extract_features(model, g_set, pool=pool, use_gpu=use_gpu)
    distmat = compute_distance_matrix(qf, gf)
    cmc, mAP = evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids)
    return cmc, mAP

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--weight_paths", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    
    dataset_root = os.path.dirname(args.dataset_root)
    ds_name = os.path.basename(args.dataset_root)
    _, _, num_classes, cam_num, _, q_set, g_set = heatmap_dataloader(ds_name, dataset_root)
    
    results = []
    for wpath, seed in zip(args.weight_paths, args.seeds):
        print(f"\n{'='*50}")
        print(f"Evaluating seed={seed}: {wpath}")
        model = KeyRe_ID(num_classes=num_classes, camera_num=cam_num, pretrainpath=None)
        state = torch.load(wpath, map_location='cpu')
        model.load_state_dict(state, strict=False)
        model.cuda().eval()
        cmc, mAP = test_full_cmc(model, q_set, g_set)
        r1, r5, r10 = cmc[0], cmc[4], cmc[9]
        print(f"  Rank-1:  {r1:.4f} ({r1*100:.2f}%)")
        print(f"  Rank-5:  {r5:.4f} ({r5*100:.2f}%)")
        print(f"  Rank-10: {r10:.4f} ({r10*100:.2f}%)")
        print(f"  mAP:     {mAP:.4f} ({mAP*100:.2f}%)")
        results.append((seed, r1, r5, r10, mAP))
        del model; torch.cuda.empty_cache()
    
    print(f"\n{'='*50}")
    print("SUMMARY (mean +/- std)")
    for name, idx in [("Rank-1", 1), ("Rank-5", 2), ("Rank-10", 3), ("mAP", 4)]:
        vals = [r[idx] for r in results]
        print(f"  {name}:  {np.mean(vals)*100:.2f} +/- {np.std(vals)*100:.2f}%")

if __name__ == "__main__":
    main()
