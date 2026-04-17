"""
iLIDS-VID 3-seed training: MARS pre-trained -> 3-stage fine-tune with early stopping.
Split 0 only, 3 seeds.
Usage: python3 train_ilids_3seed.py --seed 1234 --output_dir ...
"""
import os, sys, time, random, argparse
import numpy as np
import torch
from torch.cuda import amp
from torch_ema import ExponentialMovingAverage

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from heatmap_loader import heatmap_dataloader
from keyreid import KeyReID
from losses import make_loss
from utils import AverageMeter, CosineLRScheduler
from evaluation import extract_features, compute_distance_matrix, evaluate_rank

MARS_WEIGHTS = os.path.join(REPO_ROOT, "weights/MARSbest_CMC.pth")
ILIDS_ROOT = "./data"


def load_mars_weights(model, path):
    """Load MARS weights, skipping mismatched keys (classifier, camera embedding)."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    keys_skip = [k for k in state if 'Cam' in k or 'classifier' in k]
    for k in keys_skip:
        del state[k]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'  Loaded MARS weights: {len(state)} keys, {len(missing)} missing, {len(unexpected)} unexpected')


def evaluate_model(model, q_set, g_set):
    """Evaluate and return R-1, R-5, mAP."""
    model.eval()
    qf, qp, qc = extract_features(model, q_set)
    gf, gp, gc = extract_features(model, g_set)
    distmat = compute_distance_matrix(qf, gf)
    cmc, mAP = evaluate_rank(distmat, qp, gp, qc, gc)
    return cmc[0], cmc[4], mAP


def train_stage(model, loader, q_set, g_set, num_classes, lr, max_epochs, patience, stage_name, save_path):
    """Train one stage with early stopping. Returns best R-1, R-5, mAP."""
    loss_fun, center_crit = make_loss(num_classes=num_classes, use_gpu=True)
    center_lr = 0.5 if 'Stage1' in stage_name else 0.1
    opt_center = torch.optim.SGD(center_crit.parameters(), lr=center_lr)

    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        plr = lr * 2 if 'bias' in key else lr
        params.append({'params': [value], 'lr': plr, 'weight_decay': 1e-4})
    optimizer = torch.optim.SGD(params, momentum=0.9)

    sched = CosineLRScheduler(optimizer, t_initial=max_epochs,
                              lr_min=lr * 0.02, warmup_lr_init=lr * 0.1, warmup_t=3,
                              cycle_limit=1, t_in_epochs=True)

    scaler = amp.GradScaler()
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

    best_r1, best_r5, best_mAP = 0, 0, 0
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        sched.step(epoch)
        model.train()

        for it, (imgs, heatmaps, pid, cam, erase) in enumerate(loader):
            optimizer.zero_grad()
            opt_center.zero_grad()

            imgs = imgs.cuda()
            heatmaps = heatmaps.cuda()
            pid = pid.cuda()
            cam_l = cam.cuda().view(-1)
            erase = erase.cuda()

            with amp.autocast(enabled=True):
                score, feat, a_vals = model(imgs, heatmaps, pid, cam_label=cam_l)
                loss_id, center = loss_fun(score, feat, pid)
                attn_loss = (a_vals * erase).sum(1).mean()
                loss = loss_id + 0.0005 * center + attn_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            ema.update()

            for p in center_crit.parameters():
                if p.grad is not None:
                    p.grad.data *= (1.0 / 0.0005)
            scaler.step(opt_center)
            scaler.update()

        # Evaluate every epoch
        r1, r5, mAP = evaluate_model(model, q_set, g_set)

        improved = False
        if r1 > best_r1 or (r1 == best_r1 and r5 > best_r5):
            best_r1, best_r5, best_mAP = r1, r5, mAP
            torch.save(model.state_dict(), save_path)
            improved = True
            no_improve = 0
        else:
            no_improve += 1

        mark = ' *' if improved else ''
        if epoch % 5 == 0 or improved or no_improve >= patience:
            print(f'  {stage_name} ep{epoch}: R1={r1:.4f} R5={r5:.4f} mAP={mAP:.4f}{mark} (pat={no_improve}/{patience})')

        if no_improve >= patience:
            print(f'  {stage_name}: Early stop at ep{epoch}')
            break

    print(f'  {stage_name} BEST: R1={best_r1:.4f} R5={best_r5:.4f} mAP={best_mAP:.4f}')
    return best_r1, best_r5, best_mAP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--split_id', type=int, default=0)
    args = parser.parse_args()

    print(f'=== iLIDS-VID seed={args.seed}, split={args.split_id} ===')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    os.makedirs(args.output_dir, exist_ok=True)

    loader, _, num_classes, cam_num, _, q_set, g_set = heatmap_dataloader(
        'iLIDSVID', ILIDS_ROOT, split_id=args.split_id)

    # Stage 1: MARS pretrained -> lr=0.008
    print('[Stage 1] MARS pretrained, lr=0.008, patience=10')
    model = KeyReID(num_classes=num_classes, camera_num=cam_num, pretrainpath=None)
    load_mars_weights(model, MARS_WEIGHTS)
    model.cuda()

    s1_path = os.path.join(args.output_dir, 'stage1_best.pth')
    r1_s1, r5_s1, mAP_s1 = train_stage(
        model, loader, q_set, g_set, num_classes,
        lr=0.008, max_epochs=60, patience=10,
        stage_name='Stage1', save_path=s1_path)
    del model
    torch.cuda.empty_cache()

    # Stage 2: Stage1 best -> lr=0.001
    print('[Stage 2] from Stage1 best, lr=0.001, patience=10')
    model2 = KeyReID(num_classes=num_classes, camera_num=cam_num, pretrainpath=None)
    model2.load_state_dict(torch.load(s1_path, map_location='cpu', weights_only=False), strict=True)
    model2.cuda()

    s2_path = os.path.join(args.output_dir, 'stage2_best.pth')
    r1_s2, r5_s2, mAP_s2 = train_stage(
        model2, loader, q_set, g_set, num_classes,
        lr=0.001, max_epochs=60, patience=10,
        stage_name='Stage2', save_path=s2_path)
    del model2
    torch.cuda.empty_cache()

    # Stage 3: Stage2 best -> lr=0.0003
    print('[Stage 3] from Stage2 best, lr=0.0003, patience=15')
    model3 = KeyReID(num_classes=num_classes, camera_num=cam_num, pretrainpath=None)
    model3.load_state_dict(torch.load(s2_path, map_location='cpu', weights_only=False), strict=True)
    model3.cuda()

    s3_path = os.path.join(args.output_dir, 'stage3_best.pth')
    r1_s3, r5_s3, mAP_s3 = train_stage(
        model3, loader, q_set, g_set, num_classes,
        lr=0.0003, max_epochs=120, patience=15,
        stage_name='Stage3', save_path=s3_path)

    print(f'\nFINAL seed={args.seed}: R1={r1_s3:.4f} R5={r5_s3:.4f} mAP={mAP_s3:.4f}')

    # Save summary
    with open(os.path.join(args.output_dir, 'result.txt'), 'w') as f:
        f.write(f'seed={args.seed}\n')
        f.write(f'Stage1: R1={r1_s1:.4f} R5={r5_s1:.4f} mAP={mAP_s1:.4f}\n')
        f.write(f'Stage2: R1={r1_s2:.4f} R5={r5_s2:.4f} mAP={mAP_s2:.4f}\n')
        f.write(f'Stage3: R1={r1_s3:.4f} R5={r5_s3:.4f} mAP={mAP_s3:.4f}\n')
        f.write(f'FINAL: R1={r1_s3:.4f} R5={r5_s3:.4f} mAP={mAP_s3:.4f}\n')


if __name__ == '__main__':
    main()
