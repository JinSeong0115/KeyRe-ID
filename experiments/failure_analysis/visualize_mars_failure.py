"""MARS Failure Case Visualization"""
import os, sys, argparse, numpy as np, torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from KeyRe_ID_model import KeyRe_ID
from heatmap_loader import heatmap_dataloader
from evaluation import extract_features, compute_distance_matrix, evaluate_rank


def get_representative_frame(tracklet_paths):
    mid = len(tracklet_paths) // 2
    return tracklet_paths[mid]


def load_and_resize(img_path, size=(128, 256)):
    img = Image.open(img_path).convert('RGB')
    return img.resize(size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--dataset_root', default='./data')
    parser.add_argument('--num_vis', type=int, default=5)
    parser.add_argument('--num_success', type=int, default=3)
    parser.add_argument('--top_k', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    _, _, num_classes, cam_num, _, q_set, g_set = heatmap_dataloader('MARS', args.dataset_root)

    model = KeyRe_ID(num_classes=num_classes, camera_num=cam_num, pretrainpath=None)
    state = torch.load(args.model_path, map_location='cpu', weights_only=False)
    keys_skip = [k for k in state if 'Cam' in k or 'classifier' in k]
    if keys_skip:
        for k in keys_skip: del state[k]
        model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(state, strict=True)
    model.cuda().eval()
    print('Model loaded.')

    qf, q_pids, q_camids = extract_features(model, q_set)
    gf, g_pids, g_camids = extract_features(model, g_set)
    print(f'Features: query {qf.shape}, gallery {gf.shape}')

    distmat = compute_distance_matrix(qf, gf)
    cmc, mAP = evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids)
    print(f'Rank-1: {cmc[0]:.4f}, Rank-5: {cmc[4]:.4f}, mAP: {mAP:.4f}')

    distmat = distmat.numpy() if not isinstance(distmat, np.ndarray) else distmat
    indices = np.argsort(distmat, axis=1)

    query_data = q_set.dataset.dataset if hasattr(q_set.dataset, 'dataset') else q_set.dataset
    gallery_data = g_set.dataset.dataset if hasattr(g_set.dataset, 'dataset') else g_set.dataset
    q_paths = [d[0] for d in query_data]
    g_paths = [d[0] for d in gallery_data]

    failures, successes = [], []
    for q_idx in range(len(q_pids)):
        q_pid, q_camid = q_pids[q_idx], q_camids[q_idx]
        order = indices[q_idx]
        keep = ~((g_pids[order] == q_pid) & (g_camids[order] == q_camid))
        filtered = order[keep]
        top_pids = g_pids[filtered[:args.top_k]]
        top_correct = (top_pids == q_pid)

        if g_pids[filtered[0]] != q_pid:
            cr = np.where(g_pids[filtered] == q_pid)[0]
            first_rank = cr[0] + 1 if len(cr) > 0 else -1
            failures.append((q_idx, q_pid, first_rank, filtered[:args.top_k], top_correct))
        else:
            successes.append((q_idx, q_pid, filtered[:args.top_k], top_correct))

    failures.sort(key=lambda x: x[2] if x[2] > 0 else 9999, reverse=True)
    print(f'Failures: {len(failures)}/{len(q_pids)}, Rank-1 acc: {len(successes)/len(q_pids)*100:.2f}%')

    n_fail = min(args.num_vis, len(failures))
    n_succ = min(args.num_success, len(successes))
    n_rows = n_fail + n_succ

    fig, axes = plt.subplots(n_rows, args.top_k + 1, figsize=(2 * (args.top_k + 1), 2.5 * n_rows))
    if n_rows == 1: axes = [axes]

    row = 0
    for i, (q_idx, q_pid, first_rank, top_gallery, top_correct) in enumerate(failures[:n_fail]):
        q_tracklet = q_paths[q_idx]
        q_frame = get_representative_frame(q_tracklet) if isinstance(q_tracklet, (list, tuple)) else q_tracklet
        try:
            axes[row][0].imshow(load_and_resize(q_frame))
        except:
            axes[row][0].text(0.5, 0.5, f'ID:{q_pid}', ha='center', va='center')
        axes[row][0].set_title(f'Query\n(ID {q_pid})', fontsize=7, fontweight='bold')
        axes[row][0].axis('off')

        for j, g_idx in enumerate(top_gallery[:args.top_k]):
            g_tracklet = g_paths[g_idx]
            g_frame = get_representative_frame(g_tracklet) if isinstance(g_tracklet, (list, tuple)) else g_tracklet
            try:
                axes[row][j+1].imshow(load_and_resize(g_frame))
            except:
                axes[row][j+1].text(0.5, 0.5, f'ID:{g_pids[g_idx]}', ha='center', va='center')
            color = 'green' if top_correct[j] else 'red'
            for spine in axes[row][j+1].spines.values():
                spine.set_edgecolor(color); spine.set_linewidth(3); spine.set_visible(True)
            axes[row][j+1].set_title(f'R-{j+1}', fontsize=6, color=color)
            axes[row][j+1].axis('off')
        row += 1

    step = max(1, len(successes) // n_succ)
    for i, (q_idx, q_pid, top_gallery, top_correct) in enumerate(successes[::step][:n_succ]):
        q_tracklet = q_paths[q_idx]
        q_frame = get_representative_frame(q_tracklet) if isinstance(q_tracklet, (list, tuple)) else q_tracklet
        try:
            axes[row][0].imshow(load_and_resize(q_frame))
        except:
            axes[row][0].text(0.5, 0.5, f'ID:{q_pid}', ha='center', va='center')
        axes[row][0].set_title(f'Query\n(ID {q_pid})', fontsize=7, fontweight='bold')
        axes[row][0].axis('off')

        for j, g_idx in enumerate(top_gallery[:args.top_k]):
            g_tracklet = g_paths[g_idx]
            g_frame = get_representative_frame(g_tracklet) if isinstance(g_tracklet, (list, tuple)) else g_tracklet
            try:
                axes[row][j+1].imshow(load_and_resize(g_frame))
            except:
                axes[row][j+1].text(0.5, 0.5, f'ID:{g_pids[g_idx]}', ha='center', va='center')
            color = 'green' if top_correct[j] else 'red'
            for spine in axes[row][j+1].spines.values():
                spine.set_edgecolor(color); spine.set_linewidth(3); spine.set_visible(True)
            axes[row][j+1].set_title(f'R-{j+1}', fontsize=6, color=color)
            axes[row][j+1].axis('off')
        row += 1

    plt.tight_layout()
    save_path = os.path.join(args.output_dir, 'mars_failure_analysis.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')

    with open(os.path.join(args.output_dir, 'mars_failure_summary.txt'), 'w') as f:
        f.write(f'Rank-1: {cmc[0]:.4f}, Rank-5: {cmc[4]:.4f}, mAP: {mAP:.4f}\n')
        f.write(f'Total: {len(q_pids)}, Failures: {len(failures)}\n\n')
        for i, (q_idx, q_pid, first_rank, _, _) in enumerate(failures[:10]):
            f.write(f'Failure #{i+1}: qidx={q_idx}, pid={q_pid}, rank={first_rank}\n')

if __name__ == '__main__':
    main()
