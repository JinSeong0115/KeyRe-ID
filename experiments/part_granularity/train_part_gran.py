"""Part Granularity Ablation on MARS for 72-server"""
import os, sys, time, random, argparse
import numpy as np, torch
from torch.cuda import amp
from torch_ema import ExponentialMovingAverage

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from heatmap_loader import heatmap_dataloader
from model_parts import KeyReIDParts
from losses import make_loss
from utils import optimizer as build_optimizer, scheduler as build_scheduler
from evaluation import extract_features, compute_distance_matrix, evaluate_rank

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_parts', type=int, required=True, choices=[3, 4, 6])
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    DATASET_ROOT = './data'
    VIT_PATH = 'os.path.join(REPO_ROOT, "weights")/jx_vit_base_p16_224-80ecf9dd.pth'

    print(f'=== Part Granularity: {args.num_parts} parts, {args.epochs} epochs ===')
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed); random.seed(args.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = True

    loader, _, nc, cn, _, q_set, g_set = heatmap_dataloader('MARS', DATASET_ROOT)
    model = KeyReIDParts(num_classes=nc, camera_num=cn, pretrainpath=VIT_PATH, num_parts=args.num_parts)
    model.load_param(VIT_PATH); model.cuda()

    loss_fun, cc = make_loss(num_classes=nc, use_gpu=True)
    oc = torch.optim.SGD(cc.parameters(), lr=0.5)
    opt = build_optimizer(model); sched = build_scheduler(opt)
    scaler = amp.GradScaler(); ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

    os.makedirs(os.path.join(args.output_dir, 'evaluate'), exist_ok=True)
    br1, bm = 0, 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time(); sched.step(epoch); model.train()
        for it, (imgs, hm, pid, cam, er) in enumerate(loader):
            opt.zero_grad(); oc.zero_grad()
            imgs, hm = imgs.cuda(), hm.cuda()
            pid, cl, er = pid.cuda(), cam.cuda().view(-1), er.cuda()
            with amp.autocast(enabled=True):
                sc, ft, av = model(imgs, hm, pid, cam_label=cl)
                li, ct = loss_fun(sc, ft, pid)
                al = (av * er).sum(1).mean()
                loss = li + 0.0005 * ct + al
            scaler.scale(loss).backward()
            scaler.step(opt); ema.update()
            for p in cc.parameters():
                if p.grad is not None: p.grad.data *= (1./0.0005)
            scaler.step(oc); scaler.update()

        if epoch % args.eval_interval == 0:
            model.eval()
            qf, qp, qc = extract_features(model, q_set)
            gf, gp, gc = extract_features(model, g_set)
            dm = compute_distance_matrix(qf, gf)
            cmc, mAP = evaluate_rank(dm, qp, gp, qc, gc)
            r1, r5 = cmc[0], cmc[4]
            el = time.time() - t0
            print(f'Epoch {epoch}: R1={r1:.4f}, R5={r5:.4f}, mAP={mAP:.4f} ({el:.1f}s)')
            with open(os.path.join(args.output_dir, 'evaluate', 'matrix_best.txt'), 'a') as f:
                f.write(f'Epoch {epoch}: CMC={r1:.4f}, R-5={r5:.4f}, mAP={mAP:.4f}\n')
            if br1 < r1: br1 = r1
            if bm < mAP: bm = mAP

    print(f'FINAL [{args.num_parts}parts]: R1={br1:.4f}, mAP={bm:.4f}')

if __name__ == '__main__':
    main()
