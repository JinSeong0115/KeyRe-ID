#!/bin/bash
# Auto-detect repo root from script location (override with REPO_ROOT env var)
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

source ${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}
conda activate keyreid

cd ${REPO_ROOT}

# Step 1: Generate PRID heatmaps
echo "[$(date)] Step 1: Generating PRID-2011 heatmaps..."
python3 gen_heatmap_prid.py
echo "[$(date)] Heatmap generation done"

# Step 2: Train PRID with MARS pretrained weights (3 seeds)
cd ${REPO_ROOT}

PRID_ROOT="${DATA_ROOT:-./data}/PRID-2011"
MARS_BEST="${WEIGHTS_DIR:-./weights}/MARSbest_CMC.pth"
OUTPUT_BASE="${OUTPUT_ROOT:-./outputs}/PRID"

for SEED in 1234 5678 9012; do
    OUTPUT_DIR="${OUTPUT_BASE}/seed${SEED}"
    mkdir -p ${OUTPUT_DIR}
    echo "[$(date)] PRID training seed=${SEED}..."
    
    python3 -c "
import os, sys, time, random
import numpy as np
import torch
from torch.cuda import amp
from torch_ema import ExponentialMovingAverage

sys.path.insert(0, '.')
from heatmap_loader import heatmap_dataloader
from KeyRe_ID_model import KeyRe_ID
from Loss_fun import make_loss
from utility import AverageMeter, optimizer as build_optimizer, scheduler as build_scheduler

# Seed
seed = ${SEED}
torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
np.random.seed(seed); random.seed(seed)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = True

# Data
loader, _, num_classes, cam_num, _, q_set, g_set = heatmap_dataloader('PRID', '${PRID_ROOT}')
model = KeyRe_ID(num_classes=num_classes, camera_num=cam_num, pretrainpath='${MARS_BEST}')
model.load_param('${MARS_BEST}')
model.cuda()

loss_fun, center_crit = make_loss(num_classes=num_classes)
opt_center = torch.optim.SGD(center_crit.parameters(), lr=0.5)
opt = build_optimizer(model)
sched = build_scheduler(opt)
scaler = amp.GradScaler()
ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

best_r1, best_map = 0, 0
eval_dir = '${OUTPUT_DIR}/evaluate'
weight_dir = '${OUTPUT_DIR}/weights'
os.makedirs(eval_dir, exist_ok=True)
os.makedirs(weight_dir, exist_ok=True)

# Import test function
from train import test

for epoch in range(1, 121):
    t0 = time.time()
    sched.step(epoch); model.train()
    for it, (imgs, heatmaps, pid, cam, erase) in enumerate(loader):
        opt.zero_grad(); opt_center.zero_grad()
        imgs, heatmaps, pid, cam, erase = imgs.cuda(), heatmaps.cuda(), pid.cuda(), cam.cuda().view(-1), erase.cuda()
        with amp.autocast(enabled=True):
            score, feat, a_vals = model(imgs, heatmaps, pid, cam_label=cam)
            loss_id, center = loss_fun(score, feat, pid)
            loss = loss_id + 0.0005*center + (a_vals*erase).sum(1).mean()
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); ema.update()
        for p in center_crit.parameters():
            if p.grad is not None: p.grad.data *= (1./0.0005)
        scaler.step(opt_center); scaler.update()
    print('Epoch {} done in {:.1f}s'.format(epoch, time.time()-t0))
    
    if epoch % 10 == 0:
        cmc, mAP = test(model, q_set, g_set)
        print('CMC: {:.4f}, mAP: {:.4f}'.format(cmc, mAP))
        with open(os.path.join(eval_dir, 'matrix_best.txt'), 'a') as f:
            f.write('Epoch {}: CMC = {:.4f}, mAP = {:.4f}\n'.format(epoch, cmc, mAP))
        if best_r1 < cmc:
            best_r1 = cmc
            torch.save(model.state_dict(), os.path.join(weight_dir, 'PRIDbest_CMC.pth'))
        if best_map < mAP:
            best_map = mAP

print()
print('='*60)
print('FINAL RESULTS (seed=${SEED})')
print('  Best Rank-1: {:.4f}'.format(best_r1))
print('  Best mAP:    {:.4f}'.format(best_map))
print('='*60)
" 2>&1 | tee ${OUTPUT_DIR}/train_log.txt

    echo "[$(date)] PRID seed=${SEED} done"
done

echo ""
echo "============================================="
echo "PRID-2011 Summary"
echo "============================================="
for SEED in 1234 5678 9012; do
    echo "--- Seed ${SEED} ---"
    grep "FINAL" "${OUTPUT_BASE}/seed${SEED}/train_log.txt" -A 3 2>/dev/null
done
