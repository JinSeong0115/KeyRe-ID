"""
Part 1: Extract ranking cases (success + various failure levels)
Saves individual images + metadata JSON.
No figure generation — that's done separately.

Usage:
  python3 -u extract_ranking_cases.py \
    --dataset_name MARS \
    --dataset_root /home/bj_noh/data \
    --model_path /home/bj_noh/data/weights/MARSbest_CMC.pth \
    --output_dir /home/bj_noh/data/experiments/ranking_cases/MARS \
    --num_per_category 10
"""
import os, sys, json, gc, argparse, shutil
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def get_mid_frame(paths):
    if isinstance(paths, (list, tuple)):
        return paths[len(paths) // 2]
    return paths


def copy_frame(src_path, dst_path):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    try:
        shutil.copy2(src_path, dst_path)
    except:
        # Create placeholder if source missing
        Image.new('RGB', (128, 256), (128, 128, 128)).save(dst_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', required=True, choices=['MARS', 'iLIDSVID'])
    parser.add_argument('--dataset_root', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_per_category', type=int, default=10)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--split_id', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Import after path setup
    code_dir = os.environ.get('KEYREID_CODE', '/home/bj_noh/KeyRe-ID')
    sys.path.insert(0, code_dir)
    from heatmap_loader import heatmap_dataloader
    from KeyRe_ID_model import KeyRe_ID
    from evaluation import extract_features, compute_distance_matrix, evaluate_rank

    print(f'=== Extracting ranking cases: {args.dataset_name} ===')

    # Load dataset
    print('Loading dataset...')
    if args.dataset_name == 'iLIDSVID':
        train_loader, _, num_classes, cam_num, _, q_set, g_set = heatmap_dataloader(
            args.dataset_name, args.dataset_root, split_id=args.split_id)
    else:
        train_loader, _, num_classes, cam_num, _, q_set, g_set = heatmap_dataloader(
            args.dataset_name, args.dataset_root)
    del train_loader
    gc.collect()
    print(f'Classes: {num_classes}, Cameras: {cam_num}')

    # Load model
    print('Loading model...')
    model = KeyRe_ID(num_classes=num_classes, camera_num=cam_num, pretrainpath=None)
    state = torch.load(args.model_path, map_location='cpu', weights_only=False)
    # Handle mismatched keys (different num_classes/cameras)
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered, strict=False)
    print(f'Loaded {len(filtered)}/{len(state)} keys')
    model.cuda().eval()

    # Extract features
    print('Extracting features...')
    qf, q_pids, q_camids = extract_features(model, q_set)
    gf, g_pids, g_camids = extract_features(model, g_set)
    print(f'Query: {qf.shape}, Gallery: {gf.shape}')

    # Compute distance
    distmat = compute_distance_matrix(qf, gf)
    cmc, mAP = evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids)
    print(f'Rank-1: {cmc[0]*100:.2f}%, Rank-5: {cmc[4]*100:.2f}%, mAP: {mAP*100:.2f}%')

    distmat_np = distmat.numpy() if hasattr(distmat, 'numpy') else distmat
    indices = np.argsort(distmat_np, axis=1)

    # Get image paths
    q_data = q_set.dataset.dataset if hasattr(q_set.dataset, 'dataset') else q_set.dataset
    g_data = g_set.dataset.dataset if hasattr(g_set.dataset, 'dataset') else g_set.dataset
    q_paths = [d[0] for d in q_data]
    g_paths = [d[0] for d in g_data]

    # Categorize queries by number of failures in top-k
    categories = {}  # key: num_wrong, value: list of (qi, qpid, top_gallery_indices, top_correct)

    for qi in range(len(q_pids)):
        qpid = q_pids[qi]
        qcam = q_camids[qi]
        order = indices[qi]
        keep = ~((g_pids[order] == qpid) & (g_camids[order] == qcam))
        filtered_order = order[keep]
        top_indices = filtered_order[:args.top_k]
        top_pids = g_pids[top_indices]
        top_correct = (top_pids == qpid)
        num_wrong = args.top_k - int(top_correct.sum())

        if num_wrong not in categories:
            categories[num_wrong] = []
        categories[num_wrong].append((qi, int(qpid), top_indices.tolist(), top_correct.tolist()))

    print(f'\nCategories (num_wrong in top-{args.top_k}):')
    for nw in sorted(categories.keys()):
        print(f'  {nw} wrong: {len(categories[nw])} queries')

    # Select samples and save images
    metadata = {
        'dataset': args.dataset_name,
        'rank1': float(cmc[0]),
        'rank5': float(cmc[4]),
        'mAP': float(mAP),
        'top_k': args.top_k,
        'cases': []
    }

    for num_wrong in sorted(categories.keys()):
        cases = categories[num_wrong]
        # Pick diverse samples (spread across queries)
        step = max(1, len(cases) // args.num_per_category)
        selected = cases[::step][:args.num_per_category]

        for idx, (qi, qpid, top_gallery, top_correct) in enumerate(selected):
            case_id = f'wrong{num_wrong}_{idx:02d}'
            case_dir = os.path.join(args.output_dir, 'images', case_id)

            # Save query image
            q_frame = get_mid_frame(q_paths[qi])
            copy_frame(q_frame, os.path.join(case_dir, 'query.jpg'))

            # Save gallery images
            g_pids_list = []
            for j, g_idx in enumerate(top_gallery):
                g_frame = get_mid_frame(g_paths[g_idx])
                copy_frame(g_frame, os.path.join(case_dir, f'rank{j+1:02d}.jpg'))
                g_pids_list.append(int(g_pids[g_idx]))

            metadata['cases'].append({
                'case_id': case_id,
                'num_wrong': num_wrong,
                'query_pid': qpid,
                'gallery_pids': g_pids_list,
                'correct': top_correct
            })

    # Save metadata
    meta_path = os.path.join(args.output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'\nSaved {len(metadata["cases"])} cases to {args.output_dir}')
    print(f'Metadata: {meta_path}')
    print('Done!')


if __name__ == '__main__':
    main()
