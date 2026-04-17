#!/bin/bash
# Ablation study for iLIDS-VID and PRID-2011
# Uses MARS pretrained weights, fine-tune with different ablation configs

# Auto-detect repo root from script location (override with REPO_ROOT env var)
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

source ${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}
conda activate keyreid
cd ${REPO_ROOT}

MARS_BEST="${WEIGHTS_DIR:-./weights}/MARSbest_CMC.pth"
ILIDS_ROOT="${DATA_ROOT:-./data}/iLIDSVID"
PRID_ROOT="${DATA_ROOT:-./data}/PRID-2011"
OUTPUT_BASE="${OUTPUT_ROOT:-./outputs}/ablation"

# Ablation configs: name, use_global, use_local, use_tcss, use_kps
CONFIGS=(
    "wo_tcss,True,True,False,True"
    "wo_kps,True,True,True,False"
    "global_only,True,False,False,False"
    "local_only,False,True,True,True"
    "full,True,True,True,True"
)

DATASETS=(
    "iLIDSVID,${ILIDS_ROOT},300,2"
    "PRID,${PRID_ROOT},89,2"
)

log() {
    echo "[$(date '+%H:%M:%S')] $1"
}

for CONFIG in "${CONFIGS[@]}"; do
    IFS=',' read -r CFG_NAME USE_GLOBAL USE_LOCAL USE_TCSS USE_KPS <<< "$CONFIG"
    
    for DATASET in "${DATASETS[@]}"; do
        IFS=',' read -r DS_NAME DS_ROOT NUM_CLS CAM_NUM <<< "$DATASET"
        
        OUT_DIR="${OUTPUT_BASE}/${CFG_NAME}/${DS_NAME}"
        mkdir -p "${OUT_DIR}/evaluate" "${OUT_DIR}/weights"
        
        log "=== ${CFG_NAME} on ${DS_NAME} ==="
        
        python3 -c "
import os, sys, time, random, numpy as np, torch
from torch.cuda import amp
from torch_ema import ExponentialMovingAverage

sys.path.insert(0, '.')
from heatmap_loader import heatmap_dataloader
from KeyRe_ID_model_ablation import KeyRe_ID_Ablation
from Loss_fun_ablation import make_loss_ablation
from utility import AverageMeter, optimizer as build_optimizer, scheduler as build_scheduler

# Seed
torch.manual_seed(1234); torch.cuda.manual_seed_all(1234)
np.random.seed(1234); random.seed(1234)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = True

# Config
use_global = ${USE_GLOBAL}
use_local = ${USE_LOCAL}
use_tcss = ${USE_TCSS}
use_kps = ${USE_KPS}

# Data
loader, _, num_classes, cam_num, _, q_set, g_set = heatmap_dataloader('${DS_NAME}', '${DS_ROOT}')

# Model
model = KeyRe_ID_Ablation(
    num_classes=num_classes, camera_num=cam_num,
    pretrainpath='${MARS_BEST}',
    use_global=use_global, use_local=use_local,
    use_tcss=use_tcss, use_kps=use_kps
)
model.load_param('${MARS_BEST}')
model.cuda()

# Loss
feat_dim = 768 if use_global else 3072
loss_fun, center_crit = make_loss_ablation(
    num_classes=num_classes, use_global=use_global, use_local=use_local
)
opt_center = torch.optim.SGD(center_crit.parameters(), lr=0.5)
opt = build_optimizer(model)
sched = build_scheduler(opt)
scaler = amp.GradScaler()
ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

# Import test
from train import test

best_r1, best_map = 0, 0
eval_dir = '${OUT_DIR}/evaluate'
weight_dir = '${OUT_DIR}/weights'

for epoch in range(1, 121):
    t0 = time.time()
    sched.step(epoch); model.train()
    for it, (imgs, heatmaps, pid, cam, erase) in enumerate(loader):
        opt.zero_grad(); opt_center.zero_grad()
        imgs, heatmaps = imgs.cuda(), heatmaps.cuda()
        pid, cam_l = pid.cuda(), cam.cuda().view(-1)
        erase = erase.cuda()
        with amp.autocast(enabled=True):
            score, feat, a_vals = model(imgs, heatmaps, pid, cam_label=cam_l)
            loss_id, center = loss_fun(score, feat, pid)
            attn_loss = (a_vals * erase).sum(1).mean() if use_global else torch.tensor(0.0, device='cuda')
            loss = loss_id + 0.0005 * center + attn_loss
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); ema.update()
        for p in center_crit.parameters():
            if p.grad is not None: p.grad.data *= (1./0.0005)
        scaler.step(opt_center); scaler.update()
    
    if epoch % 20 == 0:
        print('Epoch {} done in {:.1f}s'.format(epoch, time.time()-t0))
    
    if epoch % 10 == 0:
        cmc, mAP = test(model, q_set, g_set)
        print('Epoch {}: Rank-1={:.4f}, mAP={:.4f}'.format(epoch, cmc, mAP))
        with open(os.path.join(eval_dir, 'matrix_best.txt'), 'a') as f:
            f.write('Epoch {}: CMC = {:.4f}, mAP = {:.4f}\n'.format(epoch, cmc, mAP))
        if best_r1 < cmc: best_r1 = cmc
        if best_map < mAP: best_map = mAP

print()
print('FINAL [${CFG_NAME}] on [${DS_NAME}]: Rank-1={:.4f}, mAP={:.4f}'.format(best_r1, best_map))
" 2>&1 | tee "${OUT_DIR}/train_log.txt"
        
        log "Done: ${CFG_NAME} on ${DS_NAME}"
    done
done

# Summary
log ""
log "============================================="
log "ABLATION STUDY SUMMARY"
log "============================================="
log ""
log "iLIDS-VID:"
printf "%-15s %10s %10s\n" "Config" "Rank-1" "Rank-5"
for CONFIG in "${CONFIGS[@]}"; do
    IFS=',' read -r CFG_NAME _ _ _ _ <<< "$CONFIG"
    RESULT=$(grep "FINAL" "${OUTPUT_BASE}/${CFG_NAME}/iLIDSVID/train_log.txt" 2>/dev/null)
    log "  ${CFG_NAME}: ${RESULT}"
done

log ""
log "PRID-2011:"
for CONFIG in "${CONFIGS[@]}"; do
    IFS=',' read -r CFG_NAME _ _ _ _ <<< "$CONFIG"
    RESULT=$(grep "FINAL" "${OUTPUT_BASE}/${CFG_NAME}/PRID/train_log.txt" 2>/dev/null)
    log "  ${CFG_NAME}: ${RESULT}"
done
