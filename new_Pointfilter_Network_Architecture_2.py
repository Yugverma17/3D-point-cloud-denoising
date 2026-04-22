# TDNetDenoiser_improved.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

# --------------------------
# EdgeConv (DGCNN-style) - supports arbitrary k
# --------------------------
class EdgeConvolutionLayer(nn.Module):
    """Local feature extraction with dynamic graph (EdgeConv)."""
    def __init__(self, in_channels: int, out_channels: int, k: int = 20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    @staticmethod
    def knn(x: torch.Tensor, k: int) -> torch.Tensor:
        # x: [B, C, N]
        # returns idx: [B, N, k]
        with torch.no_grad():
            B, C, N = x.shape
            x_t = x.transpose(2, 1).contiguous()  # [B,N,C]
            inner = -2 * torch.matmul(x_t, x_t.transpose(2, 1))  # [B,N,N]
            xx = (x_t ** 2).sum(dim=2, keepdim=True)  # [B,N,1]
            pairwise = -xx - inner - xx.transpose(2, 1)  # negative squared distance
            _, idx = pairwise.topk(k=k, dim=-1)  # [B,N,k]
        return idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, N]
        returns: [B, out_channels, N]
        """
        B, C, N = x.shape
        idx = EdgeConvolutionLayer.knn(x, self.k)  # [B, N, k]

        idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
        idx = (idx + idx_base).view(-1)  # (B*N*k,)

        x_t = x.transpose(2, 1).contiguous()  # [B, N, C]
        neighbors = x_t.view(B * N, -1)[idx, :].view(B, N, self.k, C)  # [B,N,k,C]
        center = x_t.view(B, N, 1, C).repeat(1, 1, self.k, 1)  # [B,N,k,C]

        edge_feat = torch.cat([neighbors - center, center], dim=-1)  # [B,N,k,2C]
        edge_feat = edge_feat.permute(0, 3, 1, 2).contiguous()  # [B,2C,N,k]

        out = self.conv(edge_feat).max(dim=-1)[0]  # [B, out_channels, N]
        return out

# --------------------------
# Multi-head attention with full relative 3D vector positional encoding
# --------------------------
class MultiHeadAttention(nn.Module):
    """Geometry-aware multi-head self-attention using relative 3D vector encoding."""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # linear proj
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # MLP to convert relative vector (3) -> per-head scalar bias
        # input: (B,N,N,3) -> output: (B,N,N,num_heads)
        self.rel_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, num_heads)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        query/key/value: [B, N, d_model]
        positions: [B, N, 3] raw coordinates (should be normalized per patch ideally)
        returns: [B, N, d_model]
        """
        B, N, _ = query.shape

        Q = self.w_q(query).view(B, N, self.num_heads, self.d_k).permute(0, 2, 1, 3)  # [B,H,N,d_k]
        K = self.w_k(key).view(B, N, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        V = self.w_v(value).view(B, N, self.num_heads, self.d_k).permute(0, 2, 1, 3)

        # content-based scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B,H,N,N]

        # relative vectors
        rel = positions.unsqueeze(2) - positions.unsqueeze(1)  # [B,N,N,3]
        # project relative vectors to per-head scalar bias
        # output: [B,N,N,num_heads]
        pos_bias = self.rel_mlp(rel)  
        # permute to [B,H,N,N]
        pos_bias = pos_bias.permute(0, 3, 1, 2)

        # add to attention scores
        scores = scores + pos_bias

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)  # [B,H,N,d_k]
        context = context.permute(0, 2, 1, 3).contiguous().view(B, N, self.d_model)  # [B,N,d_model]

        out = self.w_o(context)
        return out

# --------------------------
# Transformer layer w/ residuals and LayerNorm
# --------------------------
class GeometryAwareLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # Pre-norm style: norm before attention/ffn
        x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x), pos)
        x = x + self.ffn(self.norm2(x))
        return x

class GeometryAwareTransformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([GeometryAwareLayer(d_model, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, pos)
        return x

# --------------------------
# Robust Chamfer Loss
# --------------------------
class RobustChamferLoss(nn.Module):
    """Robust Chamfer using sqrt smoothing to reduce outlier sensitivity."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred: [B, N, 3], target: [B, M, 3]
        returns scalar
        """
        # pairwise L2 distances
        dists = torch.cdist(pred, target, p=2)  # [B,N,M]
        d1 = torch.min(dists, dim=2)[0]  # [B,N]
        d2 = torch.min(dists, dim=1)[0]  # [B,M]
        loss = (torch.sqrt(d1 + self.eps).mean(dim=1) + torch.sqrt(d2 + self.eps).mean(dim=1)).mean()
        return loss

# --------------------------
# Repulsion Loss (For Uniform Distribution / Perfect P2M)
# --------------------------
class RepulsionLoss(nn.Module):
    """Penalizes points that are too close, effectively decreasing P2M distance."""
    def __init__(self, k=4, h=0.03):
        super().__init__()
        self.k = k
        self.h = h

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        # Pairwise distance matrix
        dist = torch.cdist(pred, pred, p=2)
        # Avoid self-distance using an out-of-place operation
        identity = torch.eye(dist.size(1), device=dist.device).unsqueeze(0)
        dist = dist + identity * 1e8
        # Get k nearest neighbors
        knn_dist, _ = dist.topk(self.k, dim=2, largest=False)
        # Apply exponential penalty
        penalty = torch.exp(-(knn_dist**2) / (self.h**2))
        return penalty.sum(dim=-1).mean()

# --------------------------
# TDNetDenoiser improved
# --------------------------
class TDNetDenoiser(nn.Module):
    """
    Improved TDNetDenoiser:
     - Multi-scale EdgeConv: k1 (coarse) and k2 (fine)
     - Fuse multi-scale features and project to transformer d_model
     - Geometry-aware transformer with full relative vector positional encoding
     - Predict displacement and apply residual scaling
     - Iterative refinement mode
    """
    def __init__(self, d_model: int = 256, num_heads: int = 8, num_layers: int = 6,
                 k1: int = 20, k2: int = 10, residual_scale: float = 0.1, dropout: float = 0.0):
        super().__init__()
        # Multi-scale local extractors: both operate on raw XYZ but different k -> multi-scale neighborhood
        self.edge_conv1 = EdgeConvolutionLayer(3, 64, k=k1)   # coarse / larger neighborhood
        self.edge_conv2 = EdgeConvolutionLayer(3, 128, k=k2)  # finer / smaller neighborhood

        # Fuse channels: 64 + 128 -> project to d_model
        self.input_proj = nn.Linear(64 + 128, d_model)

        # transformer
        self.transformer = GeometryAwareTransformer(d_model, num_heads, num_layers, dropout=dropout)

        # output MLP predicting displacement
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 3)
        )

        self.residual_scale = residual_scale

    def forward(self, noisy_points: torch.Tensor, iters: int = 1) -> torch.Tensor:
        """
        noisy_points: [B, N, 3]
        iters: Number of iterative refinement steps. (Defaults to 1 to prevent OOM)
        returns: denoised points [B, N, 3]
        """
        assert noisy_points.dim() == 3 and noisy_points.size(-1) == 3, "Input must be [B,N,3]"
        
        B, N, _ = noisy_points.shape
        denoised = noisy_points

        for _ in range(iters):
            # EdgeConv expects [B, C, N]
            x = denoised.transpose(1, 2).contiguous()  # [B,3,N]

            # Multi-scale features (both applied to raw xyz with different k)
            f1 = self.edge_conv1(x)  # [B,64,N]
            f2 = self.edge_conv2(x)  # [B,128,N]

            # transpose to [B, N, C] and fuse
            f1t = f1.transpose(1, 2).contiguous()  # [B,N,64]
            f2t = f2.transpose(1, 2).contiguous()  # [B,N,128]
            fused = torch.cat([f1t, f2t], dim=-1)  # [B,N,192]

            # project to transformer dimension
            feats = self.input_proj(fused)  # [B,N,d_model]

            # transformer with positions (use current coordinates)
            out = self.transformer(feats, denoised)  # [B,N,d_model]

            # displacement and residual scaling
            disp = self.output_proj(out)  # [B,N,3]
            denoised = denoised + self.residual_scale * disp

        return denoised

