#!/bin/bash
# Evaluate all released checkpoints.
#   bash scripts/eval_all.sh /path/to/MARS /path/to/DukeMTMC-VideoReID /path/to/iLIDS-VID /path/to/PRID-2011
set -e
MARS=$1; DUKE=$2; ILIDS=$3; PRID=$4
for s in 1234 5678 9012; do
  python tools/test.py --dataset MARS               --data_root $MARS  --weights weights/MARS/keyreid_mars_seed$s.pth
  python tools/test.py --dataset DukeMTMC-VideoReID --data_root $DUKE  --weights weights/DukeMTMC-VideoReID/keyreid_duke_seed$s.pth
  python tools/test.py --dataset iLIDS-VID          --data_root $ILIDS --weights weights/iLIDS-VID/keyreid_ilids_seed$s.pth
  python tools/test.py --dataset PRID-2011          --data_root $PRID  --weights weights/PRID-2011/keyreid_prid_seed$s.pth
done
