#!/bin/bash
# Auto-detect repo root from script location (override with REPO_ROOT env var)
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

source ${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}
conda activate keyreid
cd ${REPO_ROOT}

MARS_ROOT="${DATA_ROOT:-./data}"
VIT_PATH="${WEIGHTS_DIR:-./weights}/jx_vit_base_p16_224-80ecf9dd.pth"
OUTPUT_DIR="${OUTPUT_ROOT:-./outputs}/MARS/seed5678"
mkdir -p ${OUTPUT_DIR}

echo "[$(date)] Starting MARS seed=5678"
python3 train.py \
    --dataset_name MARS \
    --dataset_root ${MARS_ROOT} \
    --ViT_path ${VIT_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --epochs 120 \
    --eval_interval 10 \
    --seed 5678 \
    2>&1 | tee ${OUTPUT_DIR}/train_log.txt

echo "[$(date)] MARS seed=5678 done"
