"""
Training losses.

Chamfer alone tends to collapse points onto the nearest surface samples, which
scores well on CD and badly on P2M because the points bunch up. The repulsion
term counteracts that by pushing very close neighbours apart, which is why the
two are used together.
"""

import torch
import torch.nn as nn


class RobustChamferLoss(nn.Module):
    """
    Chamfer distance on sqrt-smoothed distances.

    Taking the square root before averaging down-weights the few far-away
    outliers that otherwise dominate a squared-distance loss, so a handful of
    badly-corrupted points cannot drag the whole patch.
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        dist = torch.cdist(pred, target)
        nearest_pred = dist.min(dim=2)[0]
        nearest_target = dist.min(dim=1)[0]
        return (
            torch.sqrt(nearest_pred + self.eps).mean(dim=1)
            + torch.sqrt(nearest_target + self.eps).mean(dim=1)
        ).mean()


class RepulsionLoss(nn.Module):
    """
    Penalty on points sitting closer together than `h`.

    Uses a Gaussian falloff so the gradient vanishes once points are further
    apart than h, i.e. it only acts where points have actually clumped.
    """

    def __init__(self, k=4, h=0.03):
        super().__init__()
        self.k = k
        self.h = h

    def forward(self, points):
        dist = torch.cdist(points, points)
        n = dist.size(1)
        # exclude self-distances without an in-place edit
        dist = dist + torch.eye(n, device=dist.device).unsqueeze(0) * 1e8
        knn = dist.topk(min(self.k, n - 1), dim=2, largest=False)[0]
        return torch.exp(-(knn ** 2) / (self.h ** 2)).sum(dim=-1).mean()


class DenoisingLoss(nn.Module):
    """Chamfer + weighted repulsion, returned with its components for logging."""

    def __init__(self, repulsion_weight=0.05, repulsion_k=4, repulsion_h=0.03):
        super().__init__()
        self.chamfer = RobustChamferLoss()
        self.repulsion = RepulsionLoss(k=repulsion_k, h=repulsion_h)
        self.repulsion_weight = repulsion_weight

    def forward(self, pred, target):
        chamfer = self.chamfer(pred, target)
        repulsion = self.repulsion(pred)
        total = chamfer + self.repulsion_weight * repulsion
        return total, {
            "chamfer": chamfer.item(),
            "repulsion": repulsion.item(),
            "total": total.item(),
        }
