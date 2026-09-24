"""xHuBERT: weighted layer-sum (SLA) + attention pooling + classification head."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import HubertModel, HubertConfig

import config
from utils import fix_pos_conv_weight_norm


class AttentionPooling(nn.Module):
    """2-layer MLP attention pooling over time dimension."""

    def __init__(self, hidden_dim: int = 768, attn_dim: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        scores = self.attn(x)  # (B, T, 1)
        weights = F.softmax(scores, dim=1)  # (B, T, 1)
        return (weights * x).sum(dim=1)  # (B, D)


class xHuBERT(nn.Module):
    """
    xHuBERT-full: HuBERT backbone + SLA + Attention Pooling + head.

    Supports ablation via use_sla and use_attention_pool flags:
    - use_sla=False, use_attention_pool=False → HuBERT-vanilla (fine-tuned)
    - use_sla=True,  use_attention_pool=False → +SLA
    - use_sla=False, use_attention_pool=True  → +AP
    - use_sla=True,  use_attention_pool=True  → xHuBERT-full
    """

    def __init__(self, n_classes: int = config.NUM_EMOTIONS,
                 embed_dim: int = config.EMBEDDING_DIM,
                 dropout: float = config.DROPOUT,
                 use_sla: bool = True,
                 use_attention_pool: bool = True):
        super().__init__()
        self.use_sla = use_sla
        self.use_attention_pool = use_attention_pool

        cfg = HubertConfig.from_pretrained(config.PRETRAINED_MODEL)
        cfg.output_hidden_states = True
        self.hubert = HubertModel.from_pretrained(config.PRETRAINED_MODEL, config=cfg)
        fix_pos_conv_weight_norm(self.hubert)

        # Freeze CNN encoder only
        if config.FREEZE_CNN:
            for param in self.hubert.feature_extractor.parameters():
                param.requires_grad = False
            for param in self.hubert.feature_projection.parameters():
                param.requires_grad = False

        # SLA: 13 learnable layer weights (CNN output + 12 transformer layers)
        if self.use_sla:
            self.layer_weights = nn.Parameter(torch.zeros(13))

        # Attention pooling
        if self.use_attention_pool:
            self.pool = AttentionPooling(768, 128)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(768, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, n_classes),
        )

    def forward(self, input_values: torch.Tensor,
                return_embedding: bool = False):
        outputs = self.hubert(input_values)

        if self.use_sla:
            hidden_states = outputs.hidden_states  # tuple of 13 tensors (B, T, 768)
            stacked = torch.stack(hidden_states, dim=0)  # (13, B, T, 768)
            weights = F.softmax(self.layer_weights, dim=0)  # (13,)
            h = (weights[:, None, None, None] * stacked).sum(dim=0)  # (B, T, 768)
        else:
            h = outputs.last_hidden_state  # (B, T, 768)

        if self.use_attention_pool:
            pooled = self.pool(h)  # (B, 768)
        else:
            pooled = h.mean(dim=1)  # (B, 768)

        if return_embedding:
            emb = self.head[0](pooled)  # Linear
            emb = self.head[1](emb)     # ReLU
            logits = self.head[3](self.head[2](emb))  # Dropout + Linear
            return logits, emb

        return self.head(pooled)

    def get_layer_weights(self) -> torch.Tensor:
        if self.use_sla:
            return F.softmax(self.layer_weights, dim=0).detach().cpu()
        return None

    def get_param_groups(self):
        backbone_params = list(self.hubert.parameters())
        head_params = list(self.head.parameters())

        other_params = []
        if self.use_attention_pool:
            other_params.extend(self.pool.parameters())

        groups = [
            {"params": [p for p in backbone_params if p.requires_grad],
             "lr": config.BACKBONE_LR},
            {"params": head_params + other_params, "lr": config.HEAD_LR},
        ]
        if self.use_sla:
            groups.append({
                "params": [self.layer_weights],
                "lr": config.HEAD_LR,
                "weight_decay": 0.0,
            })
        return groups

    def count_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        sla_params = self.layer_weights.numel() if self.use_sla else 0
        pool_params = sum(p.numel() for p in self.pool.parameters()) if self.use_attention_pool else 0
        head_params = sum(p.numel() for p in self.head.parameters())
        return {
            "total": total,
            "trainable": trainable,
            "sla": sla_params,
            "attention_pool": pool_params,
            "head": head_params,
        }
