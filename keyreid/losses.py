"""Identity, triplet and center losses of the global and local branches (Section 3.5).

    L_global = L_ID + L_triplet + beta * L_center                                  (Eq. 14)
    L_local  = 1/K * sum_k ( L_ID_k + L_triplet_k + beta * L_center_k )            (Eq. 16)
    L_tot    = alpha * L_global + (1 - alpha) * L_local + gamma * L_part + L_attn  (Eq. 13)

make_loss returns the identity + triplet part and the center part of
alpha * L_global + (1 - alpha) * L_local separately, so that beta can be applied to the center
terms; L_part (Eq. 10) and L_attn (Eq. 15) are added in the training scripts. The identity loss
uses label smoothing, the triplet loss batch-hard mining with a soft margin.
"""
import torch
import torch.nn as nn


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy with label smoothing (Szegedy et al., CVPR 2016)."""

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        return (- targets * log_probs).mean(0).sum()


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    return dist.clamp(min=1e-12).sqrt()


def hard_example_mining(dist_mat, labels):
    N = dist_mat.size(0)
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
    dist_ap, _ = torch.max(dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    if dist_mat[is_neg].numel() == 0:
        dist_an = torch.ones_like(dist_ap) * 1e5
    else:
        dist_an, _ = torch.min(dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    return dist_ap.squeeze(1), dist_an.squeeze(1)


class TripletLoss(object):
    """Batch-hard triplet loss with a soft margin (Hermans et al., 2017)."""

    def __init__(self):
        self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, feat, labels):
        dist_ap, dist_an = hard_example_mining(euclidean_dist(feat, feat), labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        return self.ranking_loss(dist_an - dist_ap, y), dist_ap, dist_an


class CenterLoss(nn.Module):
    """Center loss (Wen et al., ECCV 2016)."""

    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        device = torch.device('cuda' if use_gpu else 'cpu')
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim, device=device))

    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
            torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat = distmat.addmm(x, self.centers.t(), beta=1, alpha=-2)
        classes = torch.arange(self.num_classes).long().to(x.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        return distmat[mask].clamp(min=1e-12, max=1e+12).mean()


def make_loss(num_classes, alpha=0.875, global_feat_dim=768, part_feat_dim=3072, use_gpu=True):
    """Returns loss_func(scores, feats, pids) -> (identity + triplet terms, center terms) and the global center module.

    The identity loss is computed from the logits y, the triplet and center losses from f_global and f_k.
    """
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=global_feat_dim, use_gpu=use_gpu)
    center_criterion_part = CenterLoss(num_classes=num_classes, feat_dim=part_feat_dim, use_gpu=use_gpu)
    triplet = TripletLoss()
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)

    def loss_func(score, feat, pid):
        pid = pid.clamp(0, score[0].size(1) - 1)
        part_scores, part_feats = score[1:], feat[1:]
        id_loss = sum(xent(s, pid) for s in part_scores) / len(part_scores)
        id_loss = (1 - alpha) * id_loss + alpha * xent(score[0], pid)
        tri_loss = sum(triplet(f, pid)[0] for f in part_feats) / len(part_feats)
        tri_loss = (1 - alpha) * tri_loss + alpha * triplet(feat[0], pid)[0]
        center = center_criterion(feat[0], pid)
        center_part = sum(center_criterion_part(f, pid) for f in part_feats) / len(part_feats)
        center = (1 - alpha) * center_part + alpha * center
        return id_loss + tri_loss, center

    return loss_func, center_criterion
