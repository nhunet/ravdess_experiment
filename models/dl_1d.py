"""1D deep learning baselines: CNN1D, LSTM on feature vectors."""

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """1D CNN on feature vector treated as single-channel 1D signal."""

    def __init__(self, input_dim: int, n_classes: int = 8, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim)
        x = x.unsqueeze(1)  # (B, 1, input_dim)
        x = self.features(x).squeeze(-1)  # (B, 128)
        return self.classifier(x)


class LSTMClassifier(nn.Module):
    """LSTM treating feature vector as a sequence (each dim = 1 timestep)."""

    def __init__(self, input_dim: int, n_classes: int = 8,
                 hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim) → (B, input_dim, 1)
        x = x.unsqueeze(-1)
        out, (hn, _) = self.lstm(x)
        # Use last hidden state from both directions
        last = torch.cat([hn[-2], hn[-1]], dim=-1)  # (B, hidden*2)
        return self.classifier(last)
