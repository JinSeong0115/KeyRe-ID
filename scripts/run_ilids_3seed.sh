#!/bin/bash
# Auto-detect repo root from script location (override with REPO_ROOT env var)
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

source ${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}
conda activate keyreid

SCRIPT="${REPO_ROOT}/train_ilids_3seed.py"
BASE="${OUTPUT_ROOT:-./outputs}/iLIDS-3seed"

log() { echo "[$(date '+%H:%M:%S')] $1"; }

log "=== iLIDS-VID 3-seed (split 0, 3-stage, early stopping) ==="

for SEED in 1234 5678 9012; do
    OUTDIR="${BASE}/seed${SEED}"
    log "--- seed=${SEED} ---"
    python3 ${SCRIPT} --seed ${SEED} --output_dir ${OUTDIR} 2>&1 | tee ${OUTDIR}/train_log.txt
    log "seed=${SEED} done"
done

log "=== SUMMARY ==="
python3 -c "
import numpy as np, re
r1s, r5s, maps = [], [], []
for seed in [1234, 5678, 9012]:
    path = '${BASE}/seed' + str(seed) + '/result.txt'
    try:
        with open(path) as f:
            for line in f:
                if line.startswith('FINAL'):
                    m = re.search(r'R1=(\d+\.\d+) R5=(\d+\.\d+) mAP=(\d+\.\d+)', line)
                    if m:
                        r1s.append(float(m.group(1)))
                        r5s.append(float(m.group(2)))
                        maps.append(float(m.group(3)))
    except: pass

if r1s:
    print(f'iLIDS-VID 3-seed Results:')
    print(f'  Rank-1: {np.mean(r1s)*100:.2f} +/- {np.std(r1s)*100:.2f}%')
    print(f'  Rank-5: {np.mean(r5s)*100:.2f} +/- {np.std(r5s)*100:.2f}%')
    print(f'  mAP:    {np.mean(maps)*100:.2f} +/- {np.std(maps)*100:.2f}%')
    print(f'  Per-seed: {[round(v*100,2) for v in r1s]}')
else:
    print('No results found')
"

log "=== DONE ==="
