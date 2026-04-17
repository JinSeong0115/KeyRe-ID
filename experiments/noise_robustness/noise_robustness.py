"""Noise Robustness v4 - use heatmap_dataloader but immediately discard train loader"""
import os, sys, gc, argparse, numpy as np, torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from heatmap_loader import heatmap_dataloader
from keyreid import KeyReID
from evaluation import extract_features, compute_distance_matrix, evaluate_rank

class NoisyWrapper:
    def __init__(self, loader, sigma):
        self.loader = loader
        self.sigma = sigma
    def __iter__(self):
        for batch in self.loader:
            imgs, heatmaps, pids, camids, paths = batch
            if self.sigma > 0:
                heatmaps = heatmaps + torch.randn_like(heatmaps) * self.sigma
            yield imgs, heatmaps, pids, camids, paths
    def __len__(self):
        return len(self.loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--dataset_root', default='./data')
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print('Loading dataset...')
    train_loader, _, num_classes, cam_num, _, q_set, g_set = heatmap_dataloader('MARS', args.dataset_root)
    # Immediately discard train loader to free memory
    del train_loader
    gc.collect()
    print(f'Eval loaders ready: {num_classes} classes, {cam_num} cameras')

    print('Loading model...')
    model = KeyReID(num_classes=num_classes, camera_num=cam_num, pretrainpath=None)
    state = torch.load(args.model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state, strict=True)
    model.cuda().eval()
    print('Model loaded.')

    sigmas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    results = []

    for sigma in sigmas:
        print(f'\n=== sigma={sigma} ===')
        q_noisy = NoisyWrapper(q_set, sigma)
        g_noisy = NoisyWrapper(g_set, sigma)
        qf, qp, qc = extract_features(model, q_noisy)
        gf, gp, g_camids = extract_features(model, g_noisy)
        distmat = compute_distance_matrix(qf, gf)
        cmc, mAP = evaluate_rank(distmat, qp, gp, qc, g_camids)
        r1, r5 = cmc[0], cmc[4]
        print(f'  Rank-1: {r1*100:.2f}%, Rank-5: {r5*100:.2f}%, mAP: {mAP*100:.2f}%')
        results.append((sigma, r1, r5, mAP))

    with open(os.path.join(args.output_dir, 'noise_robustness.txt'), 'w') as f:
        f.write('sigma\tRank-1\tRank-5\tmAP\n')
        for sigma, r1, r5, mAP in results:
            f.write(f'{sigma}\t{r1:.4f}\t{r5:.4f}\t{mAP:.4f}\n')

    print('\n=== SUMMARY ===')
    for sigma, r1, r5, mAP in results:
        print(f'sigma={sigma:.1f}: R1={r1*100:.2f}% R5={r5*100:.2f}% mAP={mAP*100:.2f}%')

if __name__ == '__main__':
    main()
