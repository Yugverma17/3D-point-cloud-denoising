"""
Geometry-aware transformer denoiser.

Structure: multi-scale EdgeConv builds local descriptors, a transformer whose
attention is biased by relative 3D offsets mixes them, and an MLP predicts a
displacement per point. The network outputs a displacement rather than an
absolute position so that "do nothing" is the zero vector, which is a much
easier starting point to optimise from than reproducing the input coordinates.

Only the centre point of each patch is used at evaluation time (see
`predict_centre`); the model still predicts for every point in the patch
because the supervision is denser that way, but writing every point back to
the cloud lets overlapping patches overwrite each other.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn_indices(x, k):
    """k nearest neighbours by squared distance. x: (B, C, N) -> (B, N, k)."""
    with torch.no_grad():
        xt = x.transpose(2, 1)                                    # (B, N, C)
        sq = (xt ** 2).sum(dim=2, keepdim=True)                   # (B, N, 1)
        # -(a-b)^2 = -a^2 + 2ab - b^2, maximised by topk
        neg_sq_dist = -sq + 2 * torch.matmul(xt, xt.transpose(2, 1)) - sq.transpose(2, 1)
        return neg_sq_dist.topk(k=k, dim=-1)[1]


class EdgeConv(nn.Module):
    """DGCNN edge convolution over a k-NN graph of the input coordinates."""

    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        B, C, N = x.shape
        k = min(self.k, N)
        idx = knn_indices(x, k)

        flat_idx = (idx + torch.arange(B, device=x.device).view(-1, 1, 1) * N).view(-1)
        xt = x.transpose(2, 1).contiguous()
        neighbours = xt.reshape(B * N, C)[flat_idx].view(B, N, k, C)
        centre = xt.view(B, N, 1, C).expand(-1, -1, k, -1)

        edge = torch.cat([neighbours - centre, centre], dim=-1).permute(0, 3, 1, 2)
        return self.conv(edge).max(dim=-1)[0]


class RelativePositionAttention(nn.Module):
    """
    Multi-head attention with a learned bias from relative 3D offsets.

    Plain attention is permutation-invariant and knows nothing about where
    points sit. Feeding the offset between every pair through a small MLP and
    adding the result to the logits makes attention geometry-aware, which is
    what a denoiser needs: whether a neighbour should influence a point depends
    on which direction and how far away it is.
    """

    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.rel_bias = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, num_heads))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, positions):
        B, N, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, N, self.num_heads, self.d_k).transpose(1, 2) for t in (q, k, v))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        rel = positions.unsqueeze(2) - positions.unsqueeze(1)          # (B, N, N, 3)
        scores = scores + self.rel_bias(rel).permute(0, 3, 1, 2)

        attn = self.dropout(F.softmax(scores, dim=-1))
        ctx = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, -1)
        return self.out(ctx)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.attn = RelativePositionAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, positions):
        x = x + self.attn(self.norm1(x), positions)
        return x + self.ffn(self.norm2(x))


class Denoiser(nn.Module):
    def __init__(
        self,
        d_model=256,
        num_heads=8,
        num_layers=6,
        k_coarse=20,
        k_fine=10,
        residual_scale=0.1,
        dropout=0.0,
    ):
        super().__init__()
        self.edge_coarse = EdgeConv(3, 64, k=k_coarse)
        self.edge_fine = EdgeConv(3, 128, k=k_fine)
        self.input_proj = nn.Linear(64 + 128, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, dropout) for _ in range(num_layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 3)
        )
        self.residual_scale = residual_scale

        # Start as close to the identity as possible: with a zeroed output layer
        # the network initially predicts no displacement, so early training
        # cannot make the cloud worse than its input.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def displacement(self, points):
        """Predicted per-point offset, in the same units as `points`."""
        x = points.transpose(1, 2).contiguous()
        feats = torch.cat(
            [self.edge_coarse(x).transpose(1, 2), self.edge_fine(x).transpose(1, 2)], dim=-1
        )
        h = self.input_proj(feats)
        for block in self.blocks:
            h = block(h, points)
        return self.residual_scale * self.head(h)

    def forward(self, points, iters=1):
        """points: (B, N, 3) in the patch-local frame. Returns denoised points."""
        assert points.dim() == 3 and points.size(-1) == 3, "expected (B, N, 3)"
        for _ in range(iters):
            points = points + self.displacement(points)
        return points

    @torch.no_grad()
    def predict_centre(self, patches, centre_index=0, iters=1):
        """
        Denoised position of one designated point per patch.

        This is what evaluation uses. Each point of the full cloud is the
        centre of exactly one patch, so every point gets exactly one prediction
        and overlapping patches cannot overwrite each other.
        """
        return self.forward(patches, iters=iters)[:, centre_index, :]
