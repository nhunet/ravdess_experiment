"""HuBERT vanilla: frozen or fine-tuned with last_hidden_state + mean-pool."""

import torch
import torch.nn as nn
from transformers import HubertModel, HubertConfig

import config
from utils import fix_pos_conv_weight_norm


class HuBERTVanilla(nn.Module):
    """
    HuBERT with last_hidden_state + mean-pool + classification head.
    No SLA, no attention pooling.
    Can be frozen (for Exp2) or fine-tuned (for Exp4 ablation).
    """

    def __init__(self, n_classes: int = config.NUM_EMOTIONS,
                 embed_dim: int = config.EMBEDDING_DIM,
                 dropout: float = config.DROPOUT,
                 freeze_backbone: bool = False):
        super().__init__()

        cfg = HubertConfig.from_pretrained(config.PRETRAINED_MODEL)
        cfg.output_hidden_states = False
        self.hubert = HubertModel.from_pretrained(config.PRETRAINED_MODEL, config=cfg)
        fix_pos_conv_weight_norm(self.hubert)

        if config.FREEZE_CNN:
            for param in self.hubert.feature_extractor.parameters():
                param.requires_grad = False
            for param in self.hubert.feature_projection.parameters():
                param.requires_grad = False

        if freeze_backbone:
            for param in self.hubert.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(768, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, n_classes),
        )

    def forward(self, input_values: torch.Tensor,
                return_embedding: bool = False):
        outputs = self.hubert(input_values)
        hidden = outputs.last_hidden_state  # (B, T, 768)
        pooled = hidden.mean(dim=1)  # (B, 768)

        if return_embedding:
            emb = self.head[0](pooled)
            emb = self.head[1](emb)  # ReLU
            logits = self.head[3](self.head[2](emb))  # Dropout + Linear
            return logits, emb

        return self.head(pooled)

    def get_param_groups(self):
        backbone_params = list(self.hubert.parameters())
        head_params = list(self.head.parameters())
        return [
            {"params": [p for p in backbone_params if p.requires_grad],
             "lr": config.BACKBONE_LR},
            {"params": head_params, "lr": config.HEAD_LR},
        ]
