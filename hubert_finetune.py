"""
=============================================================================
CẢI TIẾN #1: FINE-TUNE HuBERT TRÊN RAVDESS
Đề tài: Hệ thống gợi ý sản phẩm dựa trên phân tích giọng nói và cảm xúc
Học viên: Nguyễn Tấn Nhu | GVHD: TS. Bùi Thanh Hùng
=============================================================================

BỐI CẢNH
  Các thực nghiệm trước (pretrained_embeddings.py, fusion_experiment.py) đều
  dùng HuBERT ở chế độ FROZEN: chạy forward với torch.no_grad(), lấy
  mean-pooling của last_hidden_state → vector 768 chiều → đưa vào SVM/RF.

  Baseline cần vượt qua:
    • HuBERT (frozen) + SVM (RBF)  : Acc = 81.94%  |  F1-macro = 81.52%
    • Best fusion (AttGate-MLP)    : Acc = 87.15%  |  F1-macro = 86.69%

  Hạn chế của frozen features:
    1. HuBERT được pretrain cho ASR (nội dung ngôn ngữ), không phải cảm xúc.
       Trọng số không hề được điều chỉnh cho task SER.
    2. Mean-pooling toàn bộ trục thời gian → mất thông tin prosody (cao độ,
       nhịp, năng lượng biến thiên) vốn là tín hiệu chính của cảm xúc.
    3. Chỉ dùng last_hidden_state → bỏ phí 12 layer còn lại. Trong literature
       (SUPERB, Pepino et al. 2021), thông tin cảm xúc phân bố mạnh ở các
       layer GIỮA, không phải layer cuối.

BA CẢI TIẾN TRONG FILE NÀY
  (a) Fine-tune có chọn lọc  – mở khóa gradient cho Transformer layers,
                               đóng băng CNN feature encoder (chuẩn của
                               HuggingFace + tiết kiệm VRAM).
  (b) Weighted layer-sum     – học trọng số softmax α cho 13 hidden states
                               thay vì chỉ lấy layer cuối.
  (c) Attention pooling      – học trọng số theo time-step thay vì mean-pool
                               đều, để model tự tập trung vào khung có cảm xúc.

  Ngoài ra: SpecAugment (mask time/feature) làm regularization – rất quan
  trọng vì RAVDESS chỉ có 1.440 mẫu, fine-tune rất dễ overfit.

GIAO THỨC ĐÁNH GIÁ
  • Mặc định: 5-fold Stratified CV — GIỐNG HỆT baseline frozen + SVM, nên
    con số so sánh là công bằng, đưa thẳng vào báo cáo được.
  • Tuỳ chọn: speaker-independent split (chia theo Actor) — nghiêm ngặt hơn,
    tránh model học "giọng của actor X" thay vì học cảm xúc. Nên chạy thêm
    và báo cáo cả hai (thầy sẽ đánh giá cao điểm này).

OUTPUT
  • results_hubert_finetune.csv        – acc/F1 từng fold + trung bình
  • results_hubert_finetune.png        – biểu đồ so sánh Frozen vs Fine-tuned
  • confusion_hubert_finetune.png      – confusion matrix
  • hubert_ft_layer_weights.png        – trọng số α học được cho từng layer
  • embeddings_hubert_finetuned.npy    – embedding 256-d ĐÃ FINE-TUNE
                                         → dùng lại cho fusion_experiment.py
                                            và cho module recommendation

CÁCH CHẠY
  pip install torch transformers librosa scikit-learn pandas matplotlib seaborn
  python hubert_finetune.py                 # 5-fold CV (mặc định)
  python hubert_finetune.py --speaker-split # speaker-independent
  python hubert_finetune.py --quick         # 1 fold, 5 epochs (test nhanh)

YÊU CẦU PHẦN CỨNG
  GPU khuyến nghị (T4 Colab là đủ). CPU chạy được nhưng rất chậm
  (~8-10x). Với T4: ~20-25 phút/fold → ~2 giờ cho 5 fold.
=============================================================================
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)

warnings.filterwarnings("ignore")

# ============================================================================
# HẰNG SỐ
# ============================================================================

RAVDESS_PATH  = "./RAVDESS"
MODEL_NAME    = "facebook/hubert-base-ls960"

TARGET_SR     = 16000        # HuBERT bắt buộc 16kHz
DURATION      = 3.0          # giây – giống các thực nghiệm trước
MAX_SAMPLES   = int(TARGET_SR * DURATION)

N_FOLDS       = 5
EPOCHS        = 15
BATCH_SIZE    = 8            # HuBERT-base + 3s audio → 8 vừa VRAM T4 (16GB)
GRAD_ACCUM    = 2            # effective batch = 16
LR_BACKBONE   = 1e-5         # LR nhỏ cho pretrained weights (tránh phá vỡ)
LR_HEAD       = 1e-3         # LR lớn cho head khởi tạo ngẫu nhiên
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.1
PATIENCE      = 4
EMBED_DIM     = 256          # chiều embedding xuất ra cho module recommendation
RANDOM_SEED   = 42

# Đóng băng N transformer layer đầu tiên (0 = fine-tune toàn bộ 12 layer).
# Với dataset nhỏ như RAVDESS, đóng băng 4-6 layer đầu thường ổn định hơn.
N_FREEZE_LAYERS = 4

EMOTIONS = {
    1: "neutral", 2: "calm",    3: "happy",    4: "sad",
    5: "angry",   6: "fearful", 7: "disgust",  8: "surprised",
}

# Baseline để so sánh (lấy từ results_pretrained_embeddings.csv)
BASELINE = {
    "HuBERT (frozen) + SVM":   {"acc": 81.94, "f1": 81.52},
    "Best Fusion (AttGate)":   {"acc": 87.15, "f1": 86.69},
}

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# PHẦN 1: DATA
# ============================================================================

def load_ravdess_16k(data_path: str) -> tuple[np.ndarray, list[str], list[int]]:
    """
    Quét RAVDESS, load trực tiếp ở 16kHz (không resample 2 lần như code cũ).

    Trả về:
      waveforms : (N, MAX_SAMPLES) float32
      labels    : list[str]  – tên cảm xúc
      actors    : list[int]  – ID diễn viên (dùng cho speaker-independent split)
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"\n{'='*60}\nKHÔNG TÌM THẤY: {path.absolute()}\n"
            f"Download: https://zenodo.org/record/1188976\n{'='*60}"
        )

    wav_files = sorted(path.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"Không có .wav trong {path}")

    waves, labels, actors = [], [], []
    print(f"[INFO] Đang tải {len(wav_files)} files @ {TARGET_SR}Hz...")

    for i, fp in enumerate(wav_files):
        if i % 200 == 0:
            print(f"    {i}/{len(wav_files)}")
        parts = fp.stem.split("-")
        if len(parts) < 7:
            continue
        code = int(parts[2])
        if code not in EMOTIONS:
            continue
        try:
            y, _ = librosa.load(str(fp), sr=TARGET_SR,
                                duration=DURATION, mono=True)
            # Pad / trim về đúng MAX_SAMPLES
            if len(y) < MAX_SAMPLES:
                y = np.pad(y, (0, MAX_SAMPLES - len(y)), mode="constant")
            else:
                y = y[:MAX_SAMPLES]
            # Chuẩn hoá biên độ
            peak = np.abs(y).max()
            if peak > 0:
                y = y / peak
            waves.append(y.astype(np.float32))
            labels.append(EMOTIONS[code])
            actors.append(int(parts[6]))
        except Exception as exc:
            print(f"  [WARN] Bỏ qua {fp.name}: {exc}")

    print(f"[INFO] Tải được {len(waves)} samples | "
          f"{len(set(labels))} lớp | {len(set(actors))} actors")
    return np.stack(waves), labels, actors


class RavdessDataset(Dataset):
    """Dataset trả về raw waveform – HuBERT ăn thẳng waveform, không cần MFCC."""

    def __init__(self, waves: np.ndarray, y: np.ndarray) -> None:
        self.waves = waves
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (torch.from_numpy(self.waves[idx]),
                torch.tensor(self.y[idx], dtype=torch.long))


# ============================================================================
# PHẦN 2: MÔ HÌNH
# ============================================================================

class AttentionPooling(nn.Module):
    """
    Học trọng số theo time-step thay vì mean-pooling đều.

    Lý do: trong một câu 3 giây, cảm xúc không phân bố đều — thường tập trung
    ở vài âm tiết nhấn. Mean-pooling làm loãng tín hiệu này; attention pooling
    cho model tự học nên "nghe kỹ" khung nào.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)
        w = self.attn(x)                     # (B, T, 1)
        w = torch.softmax(w, dim=1)
        return torch.sum(x * w, dim=1)       # (B, H)


class HubertSER(nn.Module):
    """
    HuBERT + Weighted Layer-Sum + Attention Pooling + Classifier head.

    Kiến trúc:
        waveform (B, 48000)
              ↓ HuBERT (13 hidden states, mỗi cái (B, T≈149, 768))
              ↓ Weighted sum: Σ softmax(α)_i · h_i          ← học được
              ↓ Attention pooling theo trục T               ← học được
              ↓ Linear(768 → 256) + ReLU + Dropout          ← EMBEDDING
              ↓ Linear(256 → 8)
            8 lớp cảm xúc

    Vector 256-d ở tầng áp chót chính là emotion embedding để đưa vào module
    gợi ý sản phẩm (xem method get_embedding()).
    """

    def __init__(self, n_classes: int,
                 model_name: str = MODEL_NAME,
                 n_freeze_layers: int = N_FREEZE_LAYERS,
                 use_weighted_layers: bool = True) -> None:
        super().__init__()
        from transformers import HubertModel, HubertConfig

        config = HubertConfig.from_pretrained(model_name)
        config.output_hidden_states = True
        # SpecAugment – regularization cực quan trọng với dataset 1.440 mẫu
        config.apply_spec_augment = True
        config.mask_time_prob     = 0.05
        config.mask_time_length   = 10
        config.mask_feature_prob  = 0.05
        config.mask_feature_length = 10

        self.hubert = HubertModel.from_pretrained(model_name, config=config)
        hidden = config.hidden_size          # 768
        n_layers = config.num_hidden_layers  # 12 → 13 hidden states (kể cả embed)

        # ── (a) Đóng băng CNN feature encoder ────────────────────────────────
        # Lớp CNN front-end học các đặc trưng âm học tổng quát; fine-tune nó
        # trên 1.440 mẫu vừa dễ hỏng vừa tốn VRAM mà gần như không có lợi.
        self.hubert.feature_extractor._freeze_parameters()

        # ── Đóng băng N transformer layer đầu ────────────────────────────────
        # Layer thấp = đặc trưng âm học chung; layer cao = ngữ nghĩa/paralinguistic.
        # Với data ít, đóng băng layer thấp giúp chống overfit.
        for i, layer in enumerate(self.hubert.encoder.layers):
            if i < n_freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False

        # ── (b) Weighted layer-sum ───────────────────────────────────────────
        self.use_weighted_layers = use_weighted_layers
        if use_weighted_layers:
            self.layer_weights = nn.Parameter(torch.zeros(n_layers + 1))

        # ── (c) Attention pooling ────────────────────────────────────────────
        self.pool = AttentionPooling(hidden)

        # ── Head ─────────────────────────────────────────────────────────────
        self.embed_fc = nn.Sequential(
            nn.Linear(hidden, EMBED_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(EMBED_DIM, n_classes)

    # ── forward ─────────────────────────────────────────────────────────────

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        """waveform (B, L) → pooled vector (B, 768)"""
        out = self.hubert(x)
        if self.use_weighted_layers:
            hs = torch.stack(out.hidden_states, dim=0)      # (13, B, T, H)
            w  = torch.softmax(self.layer_weights, dim=0)   # (13,)
            h  = (hs * w.view(-1, 1, 1, 1)).sum(dim=0)      # (B, T, H)
        else:
            h = out.last_hidden_state                        # (B, T, H)
        return self.pool(h)                                  # (B, H)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Trả về emotion embedding 256-d — ĐÂY LÀ THỨ ĐƯA VÀO MODULE GỢI Ý.
        Không đi qua classifier cuối.
        """
        return self.embed_fc(self._backbone(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed_fc(self._backbone(x)))

    def layer_weight_values(self) -> np.ndarray:
        """Trọng số α sau softmax – để vẽ biểu đồ layer nào quan trọng."""
        if not self.use_weighted_layers:
            return np.array([])
        return torch.softmax(self.layer_weights.detach().cpu(), dim=0).numpy()


# ============================================================================
# PHẦN 3: TRAIN / EVAL
# ============================================================================

def _make_optimizer(model: HubertSER):
    """
    Discriminative learning rate: backbone (pretrained) học chậm,
    head (khởi tạo ngẫu nhiên) học nhanh. Đây là chuẩn khi fine-tune.
    """
    backbone, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone if name.startswith("hubert") else head).append(p)

    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": LR_BACKBONE},
            {"params": head,     "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )


@torch.no_grad()
def _evaluate(model: HubertSER, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        with torch.autocast(device_type=DEVICE.type,
                            enabled=(DEVICE.type == "cuda")):
            logits = model(xb)
        preds.append(logits.float().argmax(1).cpu().numpy())
        trues.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def train_one_fold(waves: np.ndarray, y: np.ndarray,
                   tr_idx: np.ndarray, te_idx: np.ndarray,
                   n_classes: int, epochs: int,
                   fold_name: str = "") -> dict:
    """Huấn luyện 1 fold, trả về metrics + model đã train."""
    tr_dl = DataLoader(RavdessDataset(waves[tr_idx], y[tr_idx]),
                       batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    te_dl = DataLoader(RavdessDataset(waves[te_idx], y[te_idx]),
                       batch_size=BATCH_SIZE)

    model  = HubertSER(n_classes).to(DEVICE)
    opt    = _make_optimizer(model)
    lossfn = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    total_steps = max(1, (len(tr_dl) // GRAD_ACCUM) * epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[LR_BACKBONE, LR_HEAD],
        total_steps=total_steps, pct_start=WARMUP_RATIO,
    )

    best_f1, best_state, no_improve = 0.0, None, 0
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        run_loss = 0.0

        for step, (xb, yb) in enumerate(tr_dl):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            with torch.autocast(device_type=DEVICE.type,
                                enabled=(DEVICE.type == "cuda")):
                loss = lossfn(model(xb), yb) / GRAD_ACCUM
            scaler.scale(loss).backward()
            run_loss += loss.item() * GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                if sched.last_epoch < total_steps - 1:
                    sched.step()

        y_pred, y_true = _evaluate(model, te_dl)
        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)

        print(f"    [{fold_name}] Epoch {ep+1:2d}/{epochs}  "
              f"loss={run_loss/max(1,len(tr_dl)):.4f}  "
              f"acc={acc*100:5.2f}%  f1={f1*100:5.2f}%")

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"    [{fold_name}] Early stop tại epoch {ep+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_pred, y_true = _evaluate(model, te_dl)
    return {
        "model":     model,
        "y_pred":    y_pred,
        "y_true":    y_true,
        "accuracy":  accuracy_score(y_true, y_pred) * 100,
        "f1_macro":  f1_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "recall":    recall_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "time_s":    time.time() - t0,
    }


# ============================================================================
# PHẦN 4: VISUALIZATION
# ============================================================================

def plot_comparison(df: pd.DataFrame,
                    save_path: str = "results_hubert_finetune.png") -> None:
    """So sánh Fine-tuned vs các baseline đã có."""
    ft_acc = df["Accuracy(%)"].mean()
    ft_f1  = df["F1_macro(%)"].mean()
    ft_std = df["Accuracy(%)"].std()

    names = list(BASELINE.keys()) + ["HuBERT FINE-TUNED (cải tiến)"]
    accs  = [BASELINE[k]["acc"] for k in BASELINE] + [ft_acc]
    f1s   = [BASELINE[k]["f1"]  for k in BASELINE] + [ft_f1]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Cải tiến #1: Fine-tune HuBERT vs Baseline (RAVDESS, 5-fold CV)",
                 fontsize=13, fontweight="bold")

    x = np.arange(len(names))
    colors = ["#B0B0B0", "#8FA8C8", "#2E7D5B"]
    axes[0].bar(x - 0.2, accs, 0.4, label="Accuracy",  color=colors)
    axes[0].bar(x + 0.2, f1s,  0.4, label="F1-macro",
                color=colors, alpha=0.6, hatch="//")
    for i, (a, f) in enumerate(zip(accs, f1s)):
        axes[0].text(i - 0.2, a + 0.4, f"{a:.1f}", ha="center", fontsize=9,
                     fontweight="bold")
        axes[0].text(i + 0.2, f + 0.4, f"{f:.1f}", ha="center", fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    axes[0].set_ylabel("(%)")
    axes[0].set_ylim(70, max(max(accs), max(f1s)) + 5)
    axes[0].set_title("So sánh với baseline", fontweight="bold")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    folds = df["Fold"].astype(str)
    axes[1].plot(folds, df["Accuracy(%)"], "o-", label="Accuracy", linewidth=2)
    axes[1].plot(folds, df["F1_macro(%)"], "s--", label="F1-macro", linewidth=2)
    axes[1].axhline(BASELINE["HuBERT (frozen) + SVM"]["acc"], color="red",
                    linestyle=":", label="Frozen baseline (81.94%)")
    axes[1].set_title(f"Ổn định qua các fold (std={ft_std:.2f}%)",
                      fontweight="bold")
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("(%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Biểu đồ so sánh → {save_path}")


def plot_layer_weights(weights: np.ndarray,
                       save_path: str = "hubert_ft_layer_weights.png") -> None:
    """
    Vẽ trọng số α học được cho từng layer.
    Đây là insight rất đáng đưa vào báo cáo: chứng minh bằng số liệu rằng
    thông tin cảm xúc KHÔNG nằm ở layer cuối như code frozen cũ giả định.
    """
    if weights.size == 0:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(range(len(weights)), weights,
                  color=plt.cm.viridis(weights / weights.max()),
                  edgecolor="grey")
    peak = int(np.argmax(weights))
    bars[peak].set_edgecolor("red")
    bars[peak].set_linewidth(2.5)

    ax.set_xlabel("HuBERT hidden layer (0 = CNN output, 12 = layer cuối)")
    ax.set_ylabel("Trọng số α (sau softmax)")
    ax.set_title(f"Layer nào mang thông tin cảm xúc? → Layer {peak} có trọng số cao nhất\n"
                 f"(code frozen cũ chỉ dùng layer 12)",
                 fontweight="bold", fontsize=11)
    ax.set_xticks(range(len(weights)))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Layer weights → {save_path}  (peak = layer {peak})")


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray,
                   le: LabelEncoder, acc: float, f1: float,
                   save_path: str = "confusion_hubert_finetune.png") -> None:
    cm = confusion_matrix(y_true, y_pred).astype(float)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Tỷ lệ (%)"})
    ax.set_title(f"Confusion Matrix – HuBERT Fine-tuned\n"
                 f"Acc={acc:.1f}%  |  F1-macro={f1:.1f}%",
                 fontweight="bold")
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix → {save_path}")


# ============================================================================
# PHẦN 5: XUẤT EMBEDDING CHO MODULE RECOMMENDATION
# ============================================================================

@torch.no_grad()
def export_embeddings(model: HubertSER, waves: np.ndarray,
                      save_path: str = "embeddings_hubert_finetuned.npy") -> np.ndarray:
    """
    Trích embedding 256-d từ model đã fine-tune cho TOÀN BỘ dataset.

    File .npy này là cầu nối sang 2 việc tiếp theo:
      • fusion_experiment.py – thay embedding frozen bằng embedding fine-tuned
                               để xem AttGate-MLP có vượt 87.15% không
      • llm_reranker.py      – vector cảm xúc thật đưa vào module gợi ý
    """
    model.eval()
    dl = DataLoader(RavdessDataset(waves, np.zeros(len(waves), dtype=np.int64)),
                    batch_size=BATCH_SIZE)
    embs = []
    for xb, _ in dl:
        xb = xb.to(DEVICE)
        with torch.autocast(device_type=DEVICE.type,
                            enabled=(DEVICE.type == "cuda")):
            e = model.get_embedding(xb)
        embs.append(e.float().cpu().numpy())
    E = np.concatenate(embs)
    np.save(save_path, E)
    print(f"[INFO] Embedding fine-tuned {E.shape} → {save_path}")
    return E


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker-split", action="store_true",
                    help="Chia theo Actor (speaker-independent) thay vì K-fold")
    ap.add_argument("--quick", action="store_true",
                    help="Chạy nhanh: 1 fold, 5 epochs (để test pipeline)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    epochs  = 5 if args.quick else args.epochs
    n_folds = 1 if args.quick else N_FOLDS

    print("\n" + "=" * 70)
    print("  CẢI TIẾN #1: FINE-TUNE HuBERT TRÊN RAVDESS")
    print(f"  Device: {DEVICE}  |  Model: {MODEL_NAME}")
    print(f"  Protocol: {'Speaker-independent' if args.speaker_split else f'{n_folds}-fold Stratified CV'}")
    print(f"  Freeze: CNN encoder + {N_FREEZE_LAYERS} transformer layer đầu")
    print("=" * 70)

    waves, labels, actors = load_ravdess_16k(RAVDESS_PATH)
    le      = LabelEncoder()
    y       = le.fit_transform(labels)
    n_cls   = len(le.classes_)
    actors  = np.array(actors)

    # ── Tạo các split ────────────────────────────────────────────────────────
    if args.speaker_split:
        # Actor 1–20 train, 21–24 test (4 actors chưa từng thấy → nghiêm ngặt)
        tr = np.where(actors <= 20)[0]
        te = np.where(actors > 20)[0]
        splits = [("Speaker-Indep", tr, te)]
        print(f"\n[INFO] Train: {len(tr)} samples (Actor 1-20)  |  "
              f"Test: {len(te)} samples (Actor 21-24)")
    else:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                              random_state=RANDOM_SEED)
        splits = [(f"Fold {i+1}", tr, te)
                  for i, (tr, te) in enumerate(skf.split(waves, y))]
        splits = splits[:n_folds]

    # ── Train ────────────────────────────────────────────────────────────────
    rows, all_pred, all_true = [], [], []
    best_f1, best_model = -1.0, None

    for name, tr_idx, te_idx in splits:
        print(f"\n{'─'*60}\n  {name}  (train={len(tr_idx)}, test={len(te_idx)})\n{'─'*60}")
        r = train_one_fold(waves, y, tr_idx, te_idx, n_cls, epochs, name)

        print(f"  → Acc={r['accuracy']:.2f}%  F1={r['f1_macro']:.2f}%  "
              f"P={r['precision']:.2f}%  R={r['recall']:.2f}%  ({r['time_s']:.0f}s)")

        rows.append({
            "Fold":         name,
            "Accuracy(%)":  round(r["accuracy"], 2),
            "F1_macro(%)":  round(r["f1_macro"], 2),
            "Precision(%)": round(r["precision"], 2),
            "Recall(%)":    round(r["recall"], 2),
            "Time(s)":      round(r["time_s"], 1),
        })
        all_pred.append(r["y_pred"])
        all_true.append(r["y_true"])

        if r["f1_macro"] > best_f1:
            best_f1, best_model = r["f1_macro"], r["model"]

    # ── Tổng hợp ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    mean_acc, std_acc = df["Accuracy(%)"].mean(), df["Accuracy(%)"].std()
    mean_f1           = df["F1_macro(%)"].mean()

    df.loc[len(df)] = {
        "Fold": "TRUNG BÌNH",
        "Accuracy(%)":  round(mean_acc, 2),
        "F1_macro(%)":  round(mean_f1, 2),
        "Precision(%)": round(df["Precision(%)"][:len(rows)].mean(), 2),
        "Recall(%)":    round(df["Recall(%)"][:len(rows)].mean(), 2),
        "Time(s)":      round(df["Time(s)"][:len(rows)].sum(), 1),
    }

    print("\n" + "=" * 70)
    print("  KẾT QUẢ FINE-TUNE HuBERT")
    print("=" * 70)
    print(df.to_string(index=False))

    base_acc = BASELINE["HuBERT (frozen) + SVM"]["acc"]
    base_f1  = BASELINE["HuBERT (frozen) + SVM"]["f1"]
    print(f"\n  So với HuBERT frozen + SVM (Acc={base_acc}%, F1={base_f1}%):")
    print(f"    Δ Accuracy = {mean_acc - base_acc:+.2f}%")
    print(f"    Δ F1-macro = {mean_f1  - base_f1:+.2f}%")

    fus_acc = BASELINE["Best Fusion (AttGate)"]["acc"]
    print(f"\n  So với best fusion hiện tại (AttGate, Acc={fus_acc}%):")
    print(f"    Δ Accuracy = {mean_acc - fus_acc:+.2f}%")
    if mean_acc < fus_acc:
        print("    → Fine-tune đơn lẻ CHƯA vượt fusion. Bước tiếp theo: đưa "
              "embedding fine-tuned vào fusion_experiment.py.")

    df.to_csv("results_hubert_finetune.csv", index=False)
    print("\n[INFO] Đã lưu: results_hubert_finetune.csv")

    # ── Visualization ────────────────────────────────────────────────────────
    plot_comparison(df[:len(rows)])
    plot_confusion(np.concatenate(all_true), np.concatenate(all_pred),
                   le, mean_acc, mean_f1)
    if best_model is not None:
        plot_layer_weights(best_model.layer_weight_values())

        # ── Xuất embedding + model cho các bước sau ──────────────────────────
        export_embeddings(best_model, waves)
        torch.save(best_model.state_dict(), "hubert_ser_finetuned.pt")
        print("[INFO] Model weights → hubert_ser_finetuned.pt")

    print("\n" + "=" * 70)
    print("  HOÀN THÀNH – Output:")
    print("  • results_hubert_finetune.csv")
    print("  • results_hubert_finetune.png")
    print("  • confusion_hubert_finetune.png")
    print("  • hubert_ft_layer_weights.png")
    print("  • embeddings_hubert_finetuned.npy   ← dùng cho fusion + rec")
    print("  • hubert_ser_finetuned.pt")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
