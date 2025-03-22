from __future__ import absolute_import

import torch
from torch import nn


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        device = torch.device("cuda" if self.use_gpu else "cpu")
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim, device=device))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat = distmat.addmm(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long().to(x.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        # Optimize distance computation
        dist = distmat[mask].clamp(min=1e-12, max=1e+12)  # Numerical stability
        loss = dist.mean()  # Compute mean loss
        return loss


if __name__ == '__main__':
    use_gpu = False
    device = torch.device("cuda" if use_gpu else "cpu")

    center_loss = CenterLoss(use_gpu=use_gpu).to(device)
    features = torch.rand(16, 2048, device=device)
    targets = torch.tensor([0, 1, 2, 3, 2, 3, 1, 4, 5, 3, 2, 1, 0, 0, 5, 4], device=device)

    loss = center_loss(features, targets)
    print(loss)
