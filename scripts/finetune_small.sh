#!/bin/bash
# Fine-tune on iLIDS-VID and PRID-2011 from the MARS models trained by scripts/train_mars_duke.sh.
#   bash scripts/finetune_small.sh /path/to/iLIDS-VID /path/to/iLIDS_keypoints /path/to/PRID-2011 /path/to/PRID_keypoints
set -e
ILIDS=$1; ILIDS_KP=$2; PRID=$3; PRID_KP=$4
for s in 1234 5678 9012; do
  python tools/finetune.py --dataset iLIDS-VID --data_root $ILIDS --keypoint_root $ILIDS_KP \
      --init outputs/mars_seed$s/best_rank1.pth --output_dir outputs/ilids_seed$s --seed $s
  python tools/finetune.py --dataset PRID-2011 --data_root $PRID --keypoint_root $PRID_KP \
      --init outputs/mars_seed$s/best_rank1.pth --output_dir outputs/prid_seed$s --seed $s
done
