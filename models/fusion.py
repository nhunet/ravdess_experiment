"""Fusion models: Concat-MLP, AttGate 3-branch, Cross-Attention, and baselines."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class ConcatMLP(nn.Module):
    """F1: Concat(FT-emb, MFCC, Prosody) → MLP → 8 classes."""

    def __init__(self, ft_dim: int = 256, mfcc_dim: int = 240,
                 prosody_dim: int = 34, n_classes: int = 8):
        super().__init__()
        total = ft_dim + mfcc_dim + prosody_dim  # 530
        self.mlp = nn.Sequential(
            nn.Linear(total, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, ft_emb: torch.Tensor, mfcc: torch.Tensor,
                prosody: torch.Tensor) -> torch.Tensor:
        x = torch.cat([ft_emb, mfcc, prosody], dim=-1)
        return self.mlp(x)


class AttGateFusion(nn.Module):
    """
    F2: 3-branch attention gate fusion.
    Gate-forced variant adds entropy regularization.
    """

    def __init__(self, ft_dim: int = 256, mfcc_dim: int = 240,
                 prosody_dim: int = 34, embed_dim: int = 128,
                 n_classes: int = 8, gate_forced: bool = False,
                 entropy_lambda: float = config.GATE_ENTROPY_LAMBDA):
        super().__init__()
        self.gate_forced = gate_forced
        self.entropy_lambda = entropy_lambda

        self.branch_ft = nn.Sequential(
            nn.Linear(ft_dim, embed_dim), nn.BatchNorm1d(embed_dim),
            nn.ReLU(), nn.Dropout(0.2),
        )
        self.branch_mfcc = nn.Sequential(
            nn.Linear(mfcc_dim, embed_dim), nn.BatchNorm1d(embed_dim),
            nn.ReLU(), nn.Dropout(0.2),
        )
        self.branch_prosody = nn.Sequential(
            nn.Linear(prosody_dim, embed_dim), nn.BatchNorm1d(embed_dim),
            nn.ReLU(), nn.Dropout(0.2),
        )

        self.gate = nn.Linear(embed_dim * 3, 3)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, ft_emb: torch.Tensor, mfcc: torch.Tensor,
                prosody: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b1 = self.branch_ft(ft_emb)      # (B, embed_dim)
        b2 = self.branch_mfcc(mfcc)
        b3 = self.branch_prosody(prosody)

        gate_input = torch.cat([b1, b2, b3], dim=-1)
        g = F.softmax(self.gate(gate_input), dim=-1)  # (B, 3)

        fused = g[:, 0:1] * b1 + g[:, 1:2] * b2 + g[:, 2:3] * b3
        logits = self.classifier(fused)

        return logits, g

    def entropy_loss(self, gate_weights: torch.Tensor) -> torch.Tensor:
        """H(g) = -Σ g_m · log(g_m)"""
        eps = 1e-8
        entropy = -(gate_weights * (gate_weights + eps).log()).sum(dim=-1).mean()
        return entropy


class CrossAttentionFusion(nn.Module):
    """F3: Cross-attention — FT embedding query attends over MFCC and Prosody tokens."""

    def __init__(self, ft_dim: int = 256, mfcc_dim: int = 240,
                 prosody_dim: int = 34, n_heads: int = 4,
                 n_classes: int = 8, proj_dim: int = 128):
        super().__init__()
        self.proj_ft = nn.Linear(ft_dim, proj_dim)
        self.proj_mfcc = nn.Linear(mfcc_dim, proj_dim)
        self.proj_prosody = nn.Linear(prosody_dim, proj_dim)
        self.cross_attn = nn.MultiheadAttention(proj_dim, n_heads, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, ft_emb: torch.Tensor, mfcc: torch.Tensor,
                prosody: torch.Tensor) -> torch.Tensor:
        q = self.proj_ft(ft_emb).unsqueeze(1)        # (B, 1, proj_dim)
        k1 = self.proj_mfcc(mfcc).unsqueeze(1)       # (B, 1, proj_dim)
        k2 = self.proj_prosody(prosody).unsqueeze(1)  # (B, 1, proj_dim)
        kv = torch.cat([k1, k2], dim=1)              # (B, 2, proj_dim)
        attn_out, _ = self.cross_attn(q, kv, kv)     # (B, 1, proj_dim)
        return self.classifier(attn_out.squeeze(1))


class EmbeddingOnlyBaseline(nn.Module):
    """B0: FT-embedding only baseline with matched architecture."""

    def __init__(self, ft_dim: int = 256, embed_dim: int = 128,
                 n_classes: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ft_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, n_classes),
        )

    def forward(self, ft_emb: torch.Tensor) -> torch.Tensor:
        return self.net(ft_emb)


class SingleFeatureBaseline(nn.Module):
    """B1/B2: Single feature branch baseline."""

    def __init__(self, input_dim: int, embed_dim: int = 128,
                 n_classes: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
