#!/usr/bin/env bash
# Auto-detect repo root from script location (override with REPO_ROOT env var)
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

set -euo pipefail

PROJECT_DIR="${REPO_ROOT}"
DATASET_DIR="${DATA_ROOT:-./data}/DukeMTMC-VideoReID"
ENV_PY="${ENV_PY:-python}"
VIT_PATH="${WEIGHTS_DIR:-./weights}/jx_vit_base_p16_224-80ecf9dd.pth"
TOTAL_IMAGES=927268

KEYPOINT_DIR="${DATASET_DIR}/keypoints"
HEATMAP_DIR="${DATASET_DIR}/heatmap"
PIPELINE_LOG="${OUTPUT_ROOT:-./outputs}/duke_pipeline.log"
HEATMAP_LOG="${OUTPUT_ROOT:-./outputs}/duke_heatmap_generation.log"
TRAIN_LOG="${OUTPUT_ROOT:-./outputs}/duke_train.log"

export XDG_CACHE_HOME="${OUTPUT_ROOT:-./outputs}/cache"
export TMPDIR="${OUTPUT_ROOT:-./outputs}/tmp"

cd "${PROJECT_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

count_files() {
  local root="$1"
  local pattern="$2"
  if [[ -d "${root}" ]]; then
    find "${root}" -type f -name "${pattern}" | wc -l
  else
    echo 0
  fi
}

log "Duke pipeline watcher started." | tee -a "${PIPELINE_LOG}"
log "Waiting for keypoint extraction to finish. Target images: ${TOTAL_IMAGES}" | tee -a "${PIPELINE_LOG}"

while true; do
  pose_count="$(count_files "${KEYPOINT_DIR}" '*.pose')"
  log "Keypoint progress: ${pose_count}/${TOTAL_IMAGES}" | tee -a "${PIPELINE_LOG}"

  if [[ "${pose_count}" -ge "${TOTAL_IMAGES}" ]]; then
    break
  fi

  if ! tmux has-session -t duke_keypoints 2>/dev/null; then
    log "ERROR: duke_keypoints tmux session ended before all pose files were generated." | tee -a "${PIPELINE_LOG}"
    exit 1
  fi

  sleep 300
done

log "Keypoint extraction complete. Starting heatmap generation." | tee -a "${PIPELINE_LOG}"

"${ENV_PY}" keypoint/keypoint_to_mask.py \
  --dataset_path "${DATASET_DIR}" \
  --output_dir "${HEATMAP_DIR}" \
  --skip_existing \
  --log_every 1000 \
  2>&1 | tee "${HEATMAP_LOG}"

heatmap_count="$(count_files "${HEATMAP_DIR}" '*.npy')"
log "Heatmap generation finished. Heatmaps: ${heatmap_count}/${TOTAL_IMAGES}" | tee -a "${PIPELINE_LOG}"

if [[ "${heatmap_count}" -lt "${TOTAL_IMAGES}" ]]; then
  log "ERROR: Heatmap count is lower than total image count. Training will not start." | tee -a "${PIPELINE_LOG}"
  exit 1
fi

log "Starting DukeMTMC-VideoReID training." | tee -a "${PIPELINE_LOG}"

"${ENV_PY}" train.py \
  --dataset_name DukeMTMC-VideoReID \
  --dataset_root "${DATASET_DIR}" \
  --ViT_path "${VIT_PATH}" \
  2>&1 | tee "${TRAIN_LOG}"

log "Training command finished." | tee -a "${PIPELINE_LOG}"
