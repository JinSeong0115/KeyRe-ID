"""Extract 17 COCO keypoints for every frame with OpenPifPaf (ShuffleNetV2K16).

The keypoints are written as JSON (.pose, 17 x [x, y, confidence]) in a tree that mirrors the
image tree under --data_root. They are used only to render the reference part maps that
supervise the part-support predictor during training; inference needs no keypoints.

Run in a separate environment with openpifpaf==0.13.11 (see README). Extraction runs on CPU.

    python tools/extract_keypoints.py --data_root /path/to/MARS --output_dir /path/to/MARS_keypoints
"""
import argparse
import json
import multiprocessing as mp
import os

import numpy as np
from PIL import Image

_predictor = None
_SKIP_DIRS = {'keypoints', 'heatmap', 'heatmaps'}


def _init_worker(checkpoint, threads):
    global _predictor
    import torch
    torch.set_num_threads(threads)
    import openpifpaf
    _predictor = openpifpaf.Predictor(checkpoint=checkpoint)


def _process_one(task):
    img_path, out_path, threshold = task
    if os.path.exists(out_path):
        return 'skip'
    try:
        image = np.array(Image.open(img_path).convert('RGB'))
        predictions, _, _ = _predictor.numpy_image(image)
        if predictions:
            best = max(predictions, key=lambda p: p.score)
            keypoints = [[float(x), float(y), float(c)] if c > threshold else [0.0, 0.0, 0.0]
                         for x, y, c in best.data]
        else:
            keypoints = [[0.0, 0.0, 0.0]] * 17
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(keypoints, f)
        return 'ok'
    except Exception as exc:  # noqa: BLE001
        return f'fail:{img_path}:{exc}'


def collect_tasks(data_root, output_dir, threshold):
    tasks = []
    for root, dirs, files in os.walk(data_root):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
        for name in sorted(files):
            if not name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            rel = os.path.relpath(root, data_root)
            out_path = os.path.join(output_dir, rel, os.path.splitext(name)[0] + '.pose')
            tasks.append((os.path.join(root, name), out_path, threshold))
    return tasks


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--threads_per_worker', type=int, default=2)
    p.add_argument('--checkpoint', default='shufflenetv2k16')
    p.add_argument('--keypoint_threshold', type=float, default=0.005)
    p.add_argument('--chunksize', type=int, default=32)
    args = p.parse_args()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

    tasks = collect_tasks(args.data_root, args.output_dir, args.keypoint_threshold)
    print(f'{len(tasks)} images', flush=True)
    done = skipped = failed = 0
    with mp.Pool(args.workers, initializer=_init_worker, initargs=(args.checkpoint, args.threads_per_worker)) as pool:
        for result in pool.imap_unordered(_process_one, tasks, chunksize=args.chunksize):
            if result == 'ok':
                done += 1
            elif result == 'skip':
                skipped += 1
            else:
                failed += 1
                if failed <= 5:
                    print(result, flush=True)
            if (done + skipped + failed) % 20000 == 0:
                print(f'{done + skipped + failed}/{len(tasks)}', flush=True)
    print(f'done: new {done}, existing {skipped}, failed {failed}', flush=True)


if __name__ == '__main__':
    main()
