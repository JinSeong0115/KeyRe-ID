#!/bin/bash
# Duke multi-seed: 1차(seed=1234)는 이미 완료/진행중이므로 5678, 9012만 실행

# Auto-detect repo root from script location (override with REPO_ROOT env var)
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-keyreid}

cd ${REPO_ROOT}

DATASET_ROOT="${DATA_ROOT:-./data}/DukeMTMC-VideoReID"
VIT_PATH="${WEIGHTS_DIR:-./weights}/jx_vit_base_p16_224-80ecf9dd.pth"
OUTPUT_BASE="${OUTPUT_ROOT:-./outputs}/experiments/Duke"

export XDG_CACHE_HOME="${OUTPUT_ROOT:-./outputs}/cache"
export TMPDIR="${OUTPUT_ROOT:-./outputs}/tmp"

for SEED in 5678 9012; do
    OUTPUT_DIR="${OUTPUT_BASE}/seed${SEED}"
    mkdir -p ${OUTPUT_DIR}
    echo "============================================="
    echo "[Duke] Starting training: seed=${SEED}"
    echo "============================================="

    python3 train_multiseed.py \
        --dataset_name DukeMTMC-VideoReID \
        --dataset_root ${DATASET_ROOT} \
        --ViT_path ${VIT_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --seed ${SEED} \
        --epochs 120 \
        --eval_interval 10 \
        2>&1 | tee ${OUTPUT_DIR}/train_log.txt

    echo "[Duke] Finished seed=${SEED}"
done

echo ""
echo "============================================="
echo "Duke Multi-Seed Summary"
echo "============================================="
# seed=1234 결과 (기존 실험)
echo "--- Seed 1234 (original run) ---"
tail -5 ${OUTPUT_ROOT:-./outputs}/duke_train.log 2>/dev/null
cat ${REPO_ROOT}/evaluate/matrix_best.txt 2>/dev/null | tail -3
echo ""
for SEED in 5678 9012; do
    echo "--- Seed ${SEED} ---"
    grep "FINAL" "${OUTPUT_BASE}/seed${SEED}/train_log.txt" -A 3 2>/dev/null
done
echo "All Duke experiments completed!"
