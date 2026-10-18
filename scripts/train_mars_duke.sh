#!/bin/bash
# Train on MARS and DukeMTMC-VideoReID with the three seeds.
#   bash scripts/train_mars_duke.sh /path/to/MARS /path/to/MARS_keypoints /path/to/DukeMTMC-VideoReID /path/to/Duke_keypoints
set -e
MARS=$1; MARS_KP=$2; DUKE=$3; DUKE_KP=$4
for s in 1234 5678 9012; do
  python tools/train.py --dataset MARS --data_root $MARS --keypoint_root $MARS_KP \
      --output_dir outputs/mars_seed$s --seed $s
  python tools/train.py --dataset DukeMTMC-VideoReID --data_root $DUKE --keypoint_root $DUKE_KP \
      --output_dir outputs/duke_seed$s --seed $s
done
