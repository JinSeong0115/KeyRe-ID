from loss.softmax_loss import CrossEntropyLabelSmooth
from loss.triplet_loss import TripletLoss
from loss.center_loss import CenterLoss


def make_loss_ablation(num_classes, feat_dim=768, local_dim=3072, use_global=True, use_local=True):
    """Loss function that adapts to ablation config."""
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)
    center_criterion2 = CenterLoss(num_classes=num_classes, feat_dim=local_dim, use_gpu=True)
    triplet = TripletLoss()
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)

    def loss_func(score, feat, pid):
        num_cls = score[0].size(1) if isinstance(score, list) else score.size(1)
        pid = pid.clamp(0, num_cls - 1)

        if use_global and use_local:
            # Original loss: global + local
            ID_LOSS_local = sum(xent(s, pid) for s in score[1:]) / len(score[1:])
            ID_LOSS = 0.25 * ID_LOSS_local + 0.75 * xent(score[0], pid)

            TRI_LOSS_local = sum(triplet(f, pid)[0] for f in feat[1:]) / len(feat[1:])
            TRI_LOSS = 0.25 * TRI_LOSS_local + 0.75 * triplet(feat[0], pid)[0]

            center = 0.75 * center_criterion(feat[0], pid) + \
                     0.25 * sum(center_criterion2(f, pid) for f in feat[1:]) / len(feat[1:])
        elif use_global and not use_local:
            # Global only
            ID_LOSS = xent(score[0], pid)
            TRI_LOSS = triplet(feat[0], pid)[0]
            center = center_criterion(feat[0], pid)
        elif not use_global and use_local:
            # Local only
            ID_LOSS = sum(xent(s, pid) for s in score) / len(score)
            TRI_LOSS = sum(triplet(f, pid)[0] for f in feat) / len(feat)
            center = sum(center_criterion2(f, pid) for f in feat) / len(feat)
        else:
            raise ValueError("At least one of global/local must be True")

        return ID_LOSS + TRI_LOSS, center

    return loss_func, center_criterion
