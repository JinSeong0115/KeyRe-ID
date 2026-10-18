from .mars import MARS
from .ilids_vid import iLIDSVID
from .prid_2011 import PRID
from .dukemtmc_videoreid import DukeMTMCVideoReID

FACTORY = {
    'MARS': MARS,
    'DukeMTMC-VideoReID': DukeMTMCVideoReID,
    'iLIDS-VID': iLIDSVID,
    'PRID-2011': PRID,
}
SPLIT_DATASETS = {'iLIDS-VID', 'PRID-2011'}


def build_dataset(name, root, split_id=0):
    if name not in FACTORY:
        raise KeyError(f'unknown dataset {name}; choose from {sorted(FACTORY)}')
    if name in SPLIT_DATASETS:
        return FACTORY[name](root=root, split_id=split_id)
    return FACTORY[name](root=root)
