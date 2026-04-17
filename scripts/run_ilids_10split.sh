#!/bin/bash
# iLIDS-VID 10-split v3: MARS pre-trained, 3-stage with early stopping
# Stage1: lr=0.008, 60ep, every-epoch eval, early stop patience=10
# Stage2: lr=0.001, 60ep, every-epoch eval, early stop patience=10  
# Stage3: lr=0.0003, 120ep, every-epoch eval, early stop patience=15
# Auto-detect repo root from script location (override with REPO_ROOT env var)
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

source ${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}
conda activate keyreid
cd ${REPO_ROOT}

ILIDS_ROOT="${DATA_ROOT:-./data}"
MARS_WEIGHTS="${WEIGHTS_DIR:-./weights}/Marsbest_CMC.pth"
OUTPUT_BASE="${OUTPUT_ROOT:-./outputs}/iLIDS-10split-v3"
SEED=1234
SUMMARY_FILE="${OUTPUT_BASE}/summary.txt"

mkdir -p ${OUTPUT_BASE}
> ${SUMMARY_FILE}

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a ${SUMMARY_FILE}; }

log "iLIDS-VID 10-split v3: 3-stage + early stopping"

for SPLIT in $(seq 0 9); do
    SPLITDIR="${OUTPUT_BASE}/split${SPLIT}"
    mkdir -p ${SPLITDIR}
    
    log "========== Split ${SPLIT}/9 =========="
    
    python3 -c "
import os, sys, time, random, numpy as np, torch
from torch.cuda import amp
from torch_ema import ExponentialMovingAverage
sys.path.insert(0, '.')
from heatmap_loader import heatmap_dataloader
from KeyRe_ID_model import KeyRe_ID
from Loss_fun import make_loss
from utility import AverageMeter, CosineLRScheduler
from evaluation import extract_features, compute_distance_matrix, evaluate_rank

seed = ${SEED}
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
np.random.seed(seed); random.seed(seed)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = True

loader, _, num_classes, cam_num, _, q_set, g_set = heatmap_dataloader('iLIDSVID', '${ILIDS_ROOT}', split_id=${SPLIT})

def load_weights(model, path):
    state = torch.load(path, map_location='cpu')
    keys_skip = [k for k in state if 'Cam' in k or 'classifier' in k]
    for k in keys_skip: del state[k]
    model.load_state_dict(state, strict=False)

def evaluate_model(model, q_set, g_set):
    model.eval()
    qf, qp, qc = extract_features(model, q_set)
    gf, gp, gc = extract_features(model, g_set)
    distmat = compute_distance_matrix(qf, gf)
    cmc, mAP = evaluate_rank(distmat, qp, gp, qc, gc)
    return cmc[0], cmc[4], mAP

def train_stage(model, loader, q_set, g_set, num_classes, lr, epochs, patience, stage_name, save_dir):
    loss_fun, center_crit = make_loss(num_classes=num_classes, use_gpu=True)
    center_lr = 0.1 if 'Stage2' in stage_name or 'Stage3' in stage_name else 0.5
    opt_center = torch.optim.SGD(center_crit.parameters(), lr=center_lr)
    
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad: continue
        wd = 1e-4
        if 'bias' in key: 
            params.append({'params': [value], 'lr': lr*2, 'weight_decay': wd})
        else:
            params.append({'params': [value], 'lr': lr, 'weight_decay': wd})
    optimizer = torch.optim.SGD(params, momentum=0.9)
    
    sched = CosineLRScheduler(optimizer, t_initial=epochs,
        lr_min=lr*0.02, warmup_lr_init=lr*0.1, warmup_t=3,
        cycle_limit=1, t_in_epochs=True)
    
    scaler = amp.GradScaler()
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    
    best_r1, best_r5, best_mAP = 0, 0, 0
    no_improve = 0
    os.makedirs(os.path.join(save_dir, 'evaluate'), exist_ok=True)
    
    for epoch in range(1, epochs+1):
        sched.step(epoch); model.train()
        for it, (imgs, heatmaps, pid, cam, erase) in enumerate(loader):
            optimizer.zero_grad(); opt_center.zero_grad()
            imgs, heatmaps = imgs.cuda(), heatmaps.cuda()
            pid, cam_l, erase = pid.cuda(), cam.cuda().view(-1), erase.cuda()
            with amp.autocast(enabled=True):
                score, feat, a_vals = model(imgs, heatmaps, pid, cam_label=cam_l)
                loss_id, center = loss_fun(score, feat, pid)
                attn_loss = (a_vals * erase).sum(1).mean()
                loss = loss_id + 0.0005 * center + attn_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer); ema.update()
            for p in center_crit.parameters():
                if p.grad is not None: p.grad.data *= (1./0.0005)
            scaler.step(opt_center); scaler.update()
        
        # Eval every epoch
        r1, r5, mAP = evaluate_model(model, q_set, g_set)
        with open(os.path.join(save_dir, 'evaluate', 'matrix_best.txt'), 'a') as f:
            f.write(f'{stage_name} Epoch {epoch}: R1={r1:.4f}, R5={r5:.4f}, mAP={mAP:.4f}\n')
        
        improved = False
        if r1 > best_r1:
            best_r1, best_r5, best_mAP = r1, r5, mAP
            torch.save(model.state_dict(), os.path.join(save_dir, f'{stage_name}_best.pth'))
            improved = True
            no_improve = 0
        elif r1 == best_r1 and r5 > best_r5:
            best_r5 = r5
            torch.save(model.state_dict(), os.path.join(save_dir, f'{stage_name}_best.pth'))
            improved = True
            no_improve = 0
        else:
            no_improve += 1
        
        mark = '*' if improved else ''
        if epoch % 5 == 0 or improved:
            print(f'  {stage_name} ep{epoch}: R1={r1:.4f} R5={r5:.4f} mAP={mAP:.4f} {mark} (patience={no_improve}/{patience})')
        
        if no_improve >= patience:
            print(f'  {stage_name}: Early stopping at epoch {epoch} (no improve for {patience} epochs)')
            break
    
    print(f'  {stage_name} BEST: R1={best_r1:.4f}, R5={best_r5:.4f}, mAP={best_mAP:.4f}')
    return best_r1, best_r5, best_mAP

splitdir = '${SPLITDIR}'

# Stage 1: MARS pretrained -> iLIDS (lr=0.008)
model = KeyRe_ID(num_classes=num_classes, camera_num=cam_num, pretrainpath=None)
load_weights(model, '${MARS_WEIGHTS}')
model.cuda()
print(f'[Split ${SPLIT}] Stage1: MARS pretrained, lr=0.008')
r1_s1, r5_s1, mAP_s1 = train_stage(model, loader, q_set, g_set, num_classes,
    lr=0.008, epochs=60, patience=10, stage_name='Stage1', save_dir=splitdir)

# Stage 2: Stage1 best -> fine-tune (lr=0.001)
best_path = os.path.join(splitdir, 'Stage1_best.pth')
model2 = KeyRe_ID(num_classes=num_classes, camera_num=cam_num, pretrainpath=None)
model2.load_state_dict(torch.load(best_path, map_location='cpu'), strict=False)
model2.cuda()
print(f'[Split ${SPLIT}] Stage2: from Stage1 best, lr=0.001')
r1_s2, r5_s2, mAP_s2 = train_stage(model2, loader, q_set, g_set, num_classes,
    lr=0.001, epochs=60, patience=10, stage_name='Stage2', save_dir=splitdir)

# Stage 3: Stage2 best -> final fine-tune (lr=0.0003)
best_path2 = os.path.join(splitdir, 'Stage2_best.pth')
model3 = KeyRe_ID(num_classes=num_classes, camera_num=cam_num, pretrainpath=None)
model3.load_state_dict(torch.load(best_path2, map_location='cpu'), strict=False)
model3.cuda()
print(f'[Split ${SPLIT}] Stage3: from Stage2 best, lr=0.0003')
r1_s3, r5_s3, mAP_s3 = train_stage(model3, loader, q_set, g_set, num_classes,
    lr=0.0003, epochs=120, patience=15, stage_name='Stage3', save_dir=splitdir)

print(f'FINAL split${SPLIT}: R1={r1_s3:.4f}, R5={r5_s3:.4f}, mAP={mAP_s3:.4f}')
" 2>&1 | tee ${SPLITDIR}/train_log.txt

    FINAL=$(grep "FINAL" ${SPLITDIR}/train_log.txt | tail -1)
    log "Split ${SPLIT}: ${FINAL}"
done

log "========== ALL SPLITS DONE =========="

python3 -c "
import re, numpy as np
r1s, r5s = [], []
with open('${SUMMARY_FILE}') as f:
    for line in f:
        m = re.search(r'R1=(\d+\.\d+), R5=(\d+\.\d+)', line)
        if m:
            r1s.append(float(m.group(1)))
            r5s.append(float(m.group(2)))
if r1s:
    print(f'10-split v3 (3-stage + early stopping):')
    print(f'  Rank-1: {np.mean(r1s)*100:.2f} +/- {np.std(r1s)*100:.2f}%')
    print(f'  Rank-5: {np.mean(r5s)*100:.2f} +/- {np.std(r5s)*100:.2f}%')
    print(f'  Per-split R1: {[round(v*100,1) for v in r1s]}')
" 2>&1 | tee -a ${SUMMARY_FILE}

log "Done!"
