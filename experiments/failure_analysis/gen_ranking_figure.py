"""
Generate Figure 4 (revised): KeyRe-ID ranking list
(a) Success cases (3 rows) - all top-10 correct (green)
(b) Failure cases (2 rows) - rank-1 wrong (red/green mixed)
Query (leftmost) + Rank-1 to Rank-10
"""
import os, sys, argparse, numpy as np, torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from heatmap_loader import heatmap_dataloader
from KeyRe_ID_model import KeyRe_ID
from evaluation import extract_features, compute_distance_matrix, evaluate_rank


def get_mid_frame(paths):
    """Get middle frame path from a tracklet."""
    if isinstance(paths, (list, tuple)):
        return paths[len(paths) // 2]
    return paths


def load_img(path, size=(64, 128)):
    try:
        return Image.open(path).convert('RGB').resize(size)
    except:
        img = Image.new('RGB', size, (128, 128, 128))
        return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--dataset_root', default='./data')
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--n_success', type=int, default=3)
    parser.add_argument('--n_failure', type=int, default=2)
    parser.add_argument('--top_k', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print('Loading dataset...')
    _, _, num_classes, cam_num, _, q_set, g_set = heatmap_dataloader('MARS', args.dataset_root)

    print('Loading model...')
    model = KeyRe_ID(num_classes=num_classes, camera_num=cam_num, pretrainpath=None)
    state = torch.load(args.model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state, strict=True)
    model.cuda().eval()

    print('Extracting features...')
    qf, q_pids, q_camids = extract_features(model, q_set)
    gf, g_pids, g_camids_arr = extract_features(model, g_set)

    distmat = compute_distance_matrix(qf, gf)
    cmc, mAP = evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids_arr)
    print(f'Rank-1: {cmc[0]*100:.2f}%, mAP: {mAP*100:.2f}%')

    distmat_np = distmat.numpy() if not isinstance(distmat, np.ndarray) else distmat
    indices = np.argsort(distmat_np, axis=1)

    # Get image paths
    q_data = q_set.dataset.dataset if hasattr(q_set.dataset, 'dataset') else q_set.dataset
    g_data = g_set.dataset.dataset if hasattr(g_set.dataset, 'dataset') else g_set.dataset
    q_paths = [d[0] for d in q_data]
    g_paths = [d[0] for d in g_data]

    # Classify success / failure
    successes = []
    failures = []
    for qi in range(len(q_pids)):
        qpid, qcam = q_pids[qi], q_camids[qi]
        order = indices[qi]
        keep = ~((g_pids[order] == qpid) & (g_camids_arr[order] == qcam))
        filtered = order[keep]
        top_pids = g_pids[filtered[:args.top_k]]
        top_correct = (top_pids == qpid)

        if g_pids[filtered[0]] == qpid:
            # All top-10 correct? Pick those for cleaner success display
            if top_correct.all():
                successes.append((qi, qpid, filtered[:args.top_k], top_correct))
        else:
            cr = np.where(g_pids[filtered] == qpid)[0]
            first_rank = cr[0] + 1 if len(cr) > 0 else -1
            failures.append((qi, qpid, first_rank, filtered[:args.top_k], top_correct))

    # Sort failures by difficulty (worst first)
    failures.sort(key=lambda x: x[2] if x[2] > 0 else 9999, reverse=True)
    # Pick diverse successes (spread across queries)
    step = max(1, len(successes) // args.n_success)
    sel_success = successes[::step][:args.n_success]
    sel_failure = failures[:args.n_failure]

    print(f'Total success (all top-10 correct): {len(successes)}, Total failure: {len(failures)}')
    print(f'Selected: {len(sel_success)} success + {len(sel_failure)} failure')

    # Generate figure
    n_rows = args.n_success + args.n_failure
    n_cols = args.top_k + 1  # query + top-k
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.3 * n_cols, 1.8 * n_rows))

    # Column headers
    headers = ['Query'] + [f'Rank-{i+1}' for i in range(args.top_k)]
    for j, h in enumerate(headers):
        axes[0][j].set_title(h, fontsize=7, pad=2)

    # Row labels
    for i in range(args.n_success):
        axes[i][0].set_ylabel('', fontsize=1)
    for i in range(args.n_failure):
        axes[args.n_success + i][0].set_ylabel('', fontsize=1)

    # Draw success rows
    for row_idx, (qi, qpid, top_gallery, top_correct) in enumerate(sel_success):
        q_frame = get_mid_frame(q_paths[qi])
        q_img = load_img(q_frame)
        axes[row_idx][0].imshow(q_img)
        axes[row_idx][0].axis('off')
        # Blue border for query
        for spine in axes[row_idx][0].spines.values():
            spine.set_visible(True); spine.set_edgecolor('blue'); spine.set_linewidth(2)

        for j, g_idx in enumerate(top_gallery[:args.top_k]):
            g_frame = get_mid_frame(g_paths[g_idx])
            g_img = load_img(g_frame)
            axes[row_idx][j+1].imshow(g_img)
            axes[row_idx][j+1].axis('off')
            color = 'green' if top_correct[j] else 'red'
            for spine in axes[row_idx][j+1].spines.values():
                spine.set_visible(True); spine.set_edgecolor(color); spine.set_linewidth(2)

    # Draw failure rows
    for fi, (qi, qpid, first_rank, top_gallery, top_correct) in enumerate(sel_failure):
        row_idx = args.n_success + fi
        q_frame = get_mid_frame(q_paths[qi])
        q_img = load_img(q_frame)
        axes[row_idx][0].imshow(q_img)
        axes[row_idx][0].axis('off')
        for spine in axes[row_idx][0].spines.values():
            spine.set_visible(True); spine.set_edgecolor('blue'); spine.set_linewidth(2)

        for j, g_idx in enumerate(top_gallery[:args.top_k]):
            g_frame = get_mid_frame(g_paths[g_idx])
            g_img = load_img(g_frame)
            axes[row_idx][j+1].imshow(g_img)
            axes[row_idx][j+1].axis('off')
            color = 'green' if top_correct[j] else 'red'
            for spine in axes[row_idx][j+1].spines.values():
                spine.set_visible(True); spine.set_edgecolor(color); spine.set_linewidth(2)

    # Add section labels on left side
    fig.text(0.01, 0.5 + 0.5 * args.n_success / n_rows, 'Success\ncases',
             va='center', ha='center', fontsize=9, fontweight='bold', rotation=90,
             color='green')
    fig.text(0.01, 0.5 - 0.5 * args.n_failure / n_rows, 'Failure\ncases',
             va='center', ha='center', fontsize=9, fontweight='bold', rotation=90,
             color='red')

    # Add divider line between success and failure
    line_y = 1.0 - args.n_success / n_rows
    fig.add_artist(plt.Line2D([0.03, 0.97], [line_y, line_y],
                              transform=fig.transFigure, color='gray',
                              linewidth=1, linestyle='--'))

    plt.tight_layout(rect=[0.03, 0, 1, 1])
    plt.savefig(args.output_path, dpi=200, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f'Saved: {args.output_path}')


if __name__ == '__main__':
    main()
