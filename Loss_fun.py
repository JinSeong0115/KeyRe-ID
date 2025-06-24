import torch.nn.functional as F
from loss.softmax_loss import CrossEntropyLabelSmooth
from loss.triplet_loss import TripletLoss
from loss.center_loss import CenterLoss
import torch

def make_loss(num_classes):   
    feat_dim = 768
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # gloabal branch
    center_criterion2 = CenterLoss(num_classes=num_classes, feat_dim=3072, use_gpu=True)  # local branch
    
    triplet = TripletLoss()
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)

    def loss_func(score, feat, pid):
        num_classes = score[0].size(1) if isinstance(score, list) else score.size(1) 
        pid = pid.clamp(0, num_classes - 1) 

        if isinstance(score, list):
                ID_LOSS = [xent(scor, pid) for scor in score[1:]]
                ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                ID_LOSS = 0.25*ID_LOSS + 0.75*xent(score[0], pid)
        else:
                ID_LOSS = xent(score, pid)

        if isinstance(feat, list):
                TRI_LOSS = [triplet(feats, pid)[0] for feats in feat[1:]]
                TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                TRI_LOSS = 0.25*TRI_LOSS + 0.75*triplet(feat[0], pid)[0]

                center = center_criterion(feat[0], pid)
                centr2 = [center_criterion2(feats, pid) for feats in feat[1:]]
                centr2 = sum(centr2) / len(centr2)
                center = 0.25*centr2 + 0.75*center     
        else:
                TRI_LOSS = triplet(feat, pid)[0]

        return ID_LOSS + TRI_LOSS, center

    return loss_func, center_criterion

