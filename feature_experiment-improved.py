"""
=============================================================================
THỰC NGHIỆM SO SÁNH ĐẶC TRƯNG ÂM THANH – SPEECH EMOTION RECOGNITION
=============================================================================

Tương thích: Python 3.9 – 3.14
  - ML:  scikit-learn >= 1.3,  numpy >= 1.24,  librosa >= 0.10
  - DL:  PyTorch >= 2.1  (ưu tiên – hỗ trợ Python 3.14 từ v2.5+)
         TensorFlow >= 2.13  (fallback – Python 3.8–3.12 only)

  Script tự phát hiện DL backend theo thứ tự ưu tiên:
    1. PyTorch  → Python 3.9–3.14
    2. TensorFlow → Python 3.8–3.12
    3. Bỏ qua DL (chỉ chạy ML) nếu không có backend nào

Dataset: RAVDESS – https://zenodo.org/record/1188976
  Giải nén vào: ./RAVDESS/
  Tên file: 03-01-{emotion}-{intensity}-{statement}-{repetition}-{actor}.wav
  Emotion codes: 01=neutral 02=calm 03=happy 04=sad
                 05=angry   06=fearful 07=disgust 08=surprised
=============================================================================
"""

from __future__ import annotations      # PEP 563: forward refs, Python 3.7+

import os
import sys
import time
import warnings
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")                    # non-interactive backend – safe everywhere
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── Python version guard ─────────────────────────────────────────────────────
_MIN = (3, 9)
if sys.version_info < _MIN:
    raise RuntimeError(
        f"Yêu cầu Python >= {_MIN[0]}.{_MIN[1]}, "
        f"hiện tại: {sys.version_info.major}.{sys.version_info.minor}"
    )


# ── DL backend detection ─────────────────────────────────────────────────────
def _detect_dl_backend() -> str:
    """Phát hiện thư viện DL khả dụng. Trả về: 'torch' | 'tensorflow' | 'none'."""
    if importlib.util.find_spec("torch") is not None:
        return "torch"
    if importlib.util.find_spec("tensorflow") is not None:
        if sys.version_info >= (3, 13):
            print(
                "[WARN] TensorFlow chưa hỗ trợ Python "
                f"{sys.version_info.major}.{sys.version_info.minor}.\n"
                "       Cài PyTorch để dùng DL:\n"
                "       pip install torch --index-url "
                "https://download.pytorch.org/whl/cpu"
            )
            return "none"
        return "tensorflow"
    return "none"


DL_BACKEND = _detect_dl_backend()

print(f"[INFO] Python {sys.version_info.major}.{sys.version_info.minor}"
      f".{sys.version_info.micro}")
print(f"[INFO] DL backend: "
      f"{DL_BACKEND if DL_BACKEND != 'none' else 'không có – chỉ chạy ML'}")

# ── Hằng số ──────────────────────────────────────────────────────────────────
RAVDESS_PATH: str   = "./RAVDESS"
SR:           int   = 22050
DURATION:     float = 3.0
N_MFCC:       int   = 40
N_MELS:       int   = 128
N_FFT:        int   = 2048
HOP_LENGTH:   int   = 512
RANDOM_SEED:  int   = 42
DL_EPOCHS:    int   = 100
DL_BATCH:     int   = 32

EMOTIONS: dict[int, str] = {
    1: "neutral", 2: "calm",    3: "happy",    4: "sad",
    5: "angry",   6: "fearful", 7: "disgust",  8: "surprised",
}

np.random.seed(RANDOM_SEED)


# ============================================================================
# PHẦN 1: TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU
# ============================================================================

def load_audio(file_path: str,
               sr: int = SR,
               duration: float = DURATION) -> np.ndarray:
    """
    Tải file audio, chuẩn hóa độ dài và biên độ.
      - Pad silence nếu clip ngắn hơn duration.
      - Normalize biên độ về [-1, 1].
    """
    y, _ = librosa.load(file_path, sr=sr, duration=duration, mono=True)
    target = int(sr * duration)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)), mode="constant")
    else:
        y = y[:target]
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak
    return y


def load_ravdess(data_path: str) -> tuple[list, list, list]:
    """Quét RAVDESS, trả về (signals, labels, file_paths)."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"\n{'='*60}\n"
            f"KHÔNG TÌM THẤY: {path.absolute()}\n"
            f"Download: https://zenodo.org/record/1188976\n"
            f"Giải nén vào: {path.absolute()}\n"
            f"{'='*60}"
        )
    wav_files = sorted(path.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"Không có .wav trong {path}")

    signals: list[np.ndarray] = []
    labels:  list[str]        = []
    paths:   list[str]        = []

    print(f"[INFO] Đang tải {len(wav_files)} files từ {path}...")
    for fp in wav_files:
        parts = fp.stem.split("-")
        if len(parts) < 3:
            continue
        code = int(parts[2])
        if code not in EMOTIONS:
            continue
        try:
            signals.append(load_audio(str(fp)))
            labels.append(EMOTIONS[code])
            paths.append(str(fp))
        except Exception as exc:
            print(f"  [WARN] Bỏ qua {fp.name}: {exc}")

    print(f"[INFO] Tải được {len(signals)} samples, {len(set(labels))} lớp.")
    return signals, labels, paths


# ============================================================================
# PHẦN 2: TRÍCH XUẤT ĐẶC TRƯNG
# ============================================================================

def extract_mfcc(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    MFCC + delta + delta2 → 240 chiều.
    Mô phỏng cách tai người cảm nhận âm thanh; phổ biến nhất cho SER.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    return np.concatenate([
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
        np.mean(d1,   axis=1), np.std(d1,   axis=1),
        np.mean(d2,   axis=1), np.std(d2,   axis=1),
    ])   # 240 chiều


def extract_logmel(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Log-Mel Spectrogram → 256 chiều.
    Giữ thông tin thời gian-tần số đầy đủ, tốt cho CNN.
    """
    mel     = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                              n_fft=N_FFT, hop_length=HOP_LENGTH)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return np.concatenate([np.mean(log_mel, axis=1), np.std(log_mel, axis=1)])
    # 256 chiều


def extract_lpcc(y: np.ndarray, sr: int = SR, order: int = 13) -> np.ndarray:
    """
    LPCC (Linear Prediction Cepstral Coefficients) → 28 chiều.
    Mô hình bộ lọc thanh quản, phù hợp phân biệt cách phát âm.
    """
    frame_len = int(0.025 * sr)
    hop_f     = int(0.010 * sr)
    frames    = librosa.util.frame(y, frame_length=frame_len, hop_length=hop_f)

    lpcc_list: list[np.ndarray] = []
    for col in range(frames.shape[1]):
        frame = frames[:, col]
        try:
            lpc = librosa.lpc(y=frame, order=order)
        except Exception:
            lpcc_list.append(np.zeros(order + 1))
            continue
        c = np.zeros(order + 1)
        c[0] = float(np.log(np.abs(lpc[0]) + 1e-8))
        for n in range(1, order + 1):
            c[n] = -float(lpc[n])
            for k in range(1, n):
                c[n] -= (k / n) * c[k] * float(lpc[n - k])
        lpcc_list.append(c)

    mat = np.array(lpcc_list).T        # (order+1, n_frames)
    return np.concatenate([np.mean(mat, axis=1), np.std(mat, axis=1)])
    # 28 chiều


def extract_chroma_contrast_tonnetz(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Chroma + Spectral Contrast + Tonnetz → 50 chiều.
    Nắm bắt thông tin hòa âm và độ tương phản phổ.
    """
    harmonic = librosa.effects.harmonic(y)
    chroma   = librosa.feature.chroma_cqt(y=harmonic, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr,
                                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    tonnetz  = librosa.feature.tonnetz(y=harmonic, sr=sr)
    return np.concatenate([
        np.mean(chroma,   axis=1), np.std(chroma,   axis=1),
        np.mean(contrast, axis=1), np.std(contrast, axis=1),
        np.mean(tonnetz,  axis=1), np.std(tonnetz,  axis=1),
    ])   # 50 chiều


def extract_statistical(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Đặc trưng thống kê cơ bản → 10 chiều.
    ZCR, RMSE, Spectral Centroid, Bandwidth, Rolloff.
    """
    zcr      = librosa.feature.zero_crossing_rate(y)
    rmse     = librosa.feature.rms(y=y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr,
                                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    bwidth   = librosa.feature.spectral_bandwidth(y=y, sr=sr,
                                                   n_fft=N_FFT, hop_length=HOP_LENGTH)
    rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr,
                                                 n_fft=N_FFT, hop_length=HOP_LENGTH)
    return np.concatenate(
        [[float(np.mean(f)), float(np.std(f))]
         for f in [zcr, rmse, centroid, bwidth, rolloff]]
    )   # 10 chiều


FEATURE_EXTRACTORS: dict[str, object] = {
    "MFCC":                    extract_mfcc,
    "Log-Mel Spectrogram":     extract_logmel,
    "LPCC":                    extract_lpcc,
    "Chroma+Contrast+Tonnetz": extract_chroma_contrast_tonnetz,
    "Statistical (ZCR+RMSE)":  extract_statistical,
}

# ── Extractors 2D giữ nguyên trục thời gian (dùng cho CNN2D / CNN1D-seq) ──────
# Với DURATION=3s, SR=22050, HOP=512 → ~130 time-steps

def extract_mfcc_seq(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    MFCC + delta + delta2 dạng sequence → shape (3*N_MFCC, T).
    Giữ nguyên trục thời gian, dùng cho CNN2D / CNN1D theo time-axis.
    Padding/truncation đến TIME_STEPS cố định để batch đều nhau.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    d1   = librosa.feature.delta(mfcc)
    d2   = librosa.feature.delta(mfcc, order=2)
    mat  = np.concatenate([mfcc, d1, d2], axis=0)   # (3*N_MFCC, T)
    return _pad_or_trim_2d(mat)                       # (3*N_MFCC, TIME_STEPS)


def extract_logmel_seq(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Log-Mel Spectrogram dạng 2D → shape (N_MELS, TIME_STEPS).
    Đây là đầu vào chuẩn cho CNN2D trong SER literature.
    """
    mel     = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                              n_fft=N_FFT, hop_length=HOP_LENGTH)
    log_mel = librosa.power_to_db(mel, ref=np.max)   # (N_MELS, T)
    return _pad_or_trim_2d(log_mel)                   # (N_MELS, TIME_STEPS)


# Số time-steps cố định: tính từ DURATION để đồng nhất toàn batch
TIME_STEPS: int = int(np.ceil(SR * DURATION / HOP_LENGTH)) + 1   # ≈ 130


def _pad_or_trim_2d(mat: np.ndarray) -> np.ndarray:
    """Trim hoặc pad trục time (axis=1) về TIME_STEPS."""
    T = mat.shape[1]
    if T >= TIME_STEPS:
        return mat[:, :TIME_STEPS]
    return np.pad(mat, ((0, 0), (0, TIME_STEPS - T)), mode="constant")


# Registry riêng cho các extractor trả về 2D – dùng pipeline DL 2D
FEATURE_EXTRACTORS_2D: dict[str, object] = {
    "MFCC-Seq (CNN2D)":    extract_mfcc_seq,
    "LogMel-Seq (CNN2D)":  extract_logmel_seq,
}


def build_feature_matrix(signals: list[np.ndarray],
                         extractor_fn: object,
                         name: str = "") -> np.ndarray:
    """Trích xuất đặc trưng cho toàn bộ dataset, xử lý lỗi an toàn."""
    X: list[np.ndarray] = []
    errors = 0
    for y in signals:
        try:
            X.append(extractor_fn(y))  # type: ignore[operator]
        except Exception:
            errors += 1
            X.append(X[-1].copy() if X else np.zeros(10))
    if errors:
        print(f"  [WARN] {name}: {errors}/{len(signals)} lỗi trích xuất")
    return np.array(X)


def build_feature_matrix_2d(signals: list[np.ndarray],
                             extractor_fn: object,
                             name: str = "") -> np.ndarray:
    """
    Trích xuất đặc trưng 2D → ndarray shape (N, freq_bins, TIME_STEPS).
    Dùng cho CNN2D pipeline.
    """
    X: list[np.ndarray] = []
    errors = 0
    dummy: np.ndarray | None = None
    for y in signals:
        try:
            feat = extractor_fn(y)   # type: ignore[operator]
            X.append(feat)
            if dummy is None:
                dummy = np.zeros_like(feat)
        except Exception:
            errors += 1
            X.append(dummy if dummy is not None else np.zeros((1, TIME_STEPS)))
    if errors:
        print(f"  [WARN] {name}: {errors}/{len(signals)} lỗi trích xuất")
    return np.array(X)   # (N, freq_bins, TIME_STEPS)


# ============================================================================
# PHẦN 3: MÔ HÌNH PHÂN LOẠI
# ============================================================================

# ── ML ───────────────────────────────────────────────────────────────────────

def get_ml_models() -> dict[str, Pipeline]:
    return {
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="rbf", C=10, gamma="scale",
                          probability=True, random_state=RANDOM_SEED)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=200,
                                              random_state=RANDOM_SEED, n_jobs=-1)),
        ]),
        "KNN (k=5)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    KNeighborsClassifier(n_neighbors=5, metric="euclidean")),
        ]),
    }


def evaluate_ml(
    X: np.ndarray, y_enc: np.ndarray, model: Pipeline, cv: int = 5
) -> dict[str, float]:
    """
    5-fold stratified CV.
    Trả về dict chứa:
      accuracy_mean / accuracy_std
      f1_mean / f1_std          (macro-average)
      precision_mean            (macro-average)
      recall_mean               (macro-average)
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)

    acc_scores: list[float] = []
    f1_scores:  list[float] = []
    pre_scores: list[float] = []
    rec_scores: list[float] = []

    for tr_idx, te_idx in skf.split(X, y_enc):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y_enc[tr_idx], y_enc[te_idx]
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        acc_scores.append(accuracy_score(y_te, y_pred))
        f1_scores.append( f1_score(y_te, y_pred, average="macro", zero_division=0))
        pre_scores.append(precision_score(y_te, y_pred, average="macro", zero_division=0))
        rec_scores.append(recall_score(y_te, y_pred, average="macro", zero_division=0))

    return {
        "accuracy_mean":  float(np.mean(acc_scores)),
        "accuracy_std":   float(np.std(acc_scores)),
        "f1_mean":        float(np.mean(f1_scores)),
        "f1_std":         float(np.std(f1_scores)),
        "precision_mean": float(np.mean(pre_scores)),
        "recall_mean":    float(np.mean(rec_scores)),
    }


# ── PyTorch DL (Python 3.9 – 3.14) ──────────────────────────────────────────

def _build_cnn1d_torch(input_dim: int, n_classes: int) -> object:
    import torch.nn as nn

    class CNN1D(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 64,  3, padding=1), nn.BatchNorm1d(64),
                nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.3),
                nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128),
                nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.3),
                nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256),
                nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Dropout(0.4),
            )
            self.fc = nn.Sequential(
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, n_classes),
            )

        def forward(self, x):          # type: ignore[override]
            # x: (batch, 1, dim) – correct for Conv1d(in_channels=1)
            return self.fc(self.conv(x).squeeze(-1))

    return CNN1D()


def _build_lstm_torch(input_dim: int, n_classes: int) -> object:
    import torch.nn as nn

    class LSTMModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(1, 128, num_layers=2, batch_first=True,
                                dropout=0.3)
            self.fc   = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(64, n_classes),
            )

        def forward(self, x):          # type: ignore[override]
            # x arrives as (batch, 1, dim) from shared DataLoader
            # LSTM needs (batch, seq_len, input_size) → transpose to (batch, dim, 1)
            x = x.transpose(1, 2)          # (batch, dim, 1)
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    return LSTMModel()


_TORCH_BUILDERS: dict[str, object] = {
    "CNN 1D": _build_cnn1d_torch,
    "LSTM":   _build_lstm_torch,
}


# ── CNN2D builder (cho 2D sequence input) ────────────────────────────────────

def _build_cnn2d_torch(freq_bins: int, n_classes: int) -> object:
    """
    CNN2D nhận đầu vào (batch, 1, freq_bins, TIME_STEPS).
    Đây là kiến trúc chuẩn cho SER với spectrogram 2D.
    Trục freq và time đều được khai thác → không mất temporal info.
    """
    import torch.nn as nn

    class CNN2D(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                # Block 1
                nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d((2, 2)), nn.Dropout2d(0.25),
                # Block 2
                nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d((2, 2)), nn.Dropout2d(0.25),
                # Block 3
                nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(128), nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)), nn.Dropout2d(0.3),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(256, n_classes),
            )

        def forward(self, x):   # type: ignore[override]
            # x: (batch, 1, freq_bins, TIME_STEPS)
            return self.classifier(self.features(x))

    return CNN2D()


def _train_torch(X: np.ndarray, y_enc: np.ndarray,
                 builder, n_classes: int) -> tuple[float | None, float | None]:
    """Huấn luyện PyTorch model với early stopping."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        torch.manual_seed(RANDOM_SEED)

        sc    = StandardScaler()
        X_s   = sc.fit_transform(X).astype(np.float32)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_s, y_enc.astype(np.int64),
            test_size=0.2, stratify=y_enc, random_state=RANDOM_SEED,
        )

        def to3d(a: np.ndarray) -> "torch.Tensor":
            return torch.tensor(a).unsqueeze(1)    # (N, 1, dim) – Conv1D: (batch, channels, length)

        tr_dl = DataLoader(
            TensorDataset(to3d(X_tr),  torch.tensor(y_tr)),
            batch_size=DL_BATCH, shuffle=True,
        )
        val_dl = DataLoader(
            TensorDataset(to3d(X_val), torch.tensor(y_val)),
            batch_size=DL_BATCH,
        )

        device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model    = builder(X_tr.shape[1], n_classes).to(device)  # type: ignore[operator]
        opt      = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn  = nn.CrossEntropyLoss()

        best_acc   = 0.0
        no_improve = 0
        patience   = 25
        t0         = time.time()

        for epoch in range(DL_EPOCHS):
            model.train()
            for xb, yb in tr_dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    correct += (model(xb).argmax(1) == yb).sum().item()
                    total   += yb.size(0)

            val_acc = correct / total
            print(f"Epoch {epoch+1}/{DL_EPOCHS}, Val Acc: {val_acc:.4f}")
            if val_acc > best_acc:
                best_acc   = val_acc
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break

        return best_acc, time.time() - t0
    except ImportError:
        return None, None


# ── TensorFlow fallback (Python 3.8 – 3.12) ──────────────────────────────────

def _train_tf(X: np.ndarray, y_enc: np.ndarray,
              model_name: str, n_classes: int) -> tuple[float | None, float | None]:
    """Huấn luyện TensorFlow/Keras model (fallback)."""
    try:
        import tensorflow as tf
        tf.random.set_seed(RANDOM_SEED)

        sc = StandardScaler()
        X_s = sc.fit_transform(X)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_s, y_enc, test_size=0.2,
            stratify=y_enc, random_state=RANDOM_SEED,
        )
        X_tr_3d  = X_tr.reshape(X_tr.shape[0],  -1, 1)
        X_val_3d = X_val.reshape(X_val.shape[0], -1, 1)

        from tensorflow.keras import layers, models as km

        dim = X_tr_3d.shape[1]
        if model_name == "CNN 1D":
            m = km.Sequential([
                layers.Input(shape=(dim, 1)),
                layers.Conv1D(64,  3, activation="relu", padding="same"),
                layers.BatchNormalization(), layers.MaxPooling1D(2), layers.Dropout(0.3),
                layers.Conv1D(128, 3, activation="relu", padding="same"),
                layers.BatchNormalization(), layers.MaxPooling1D(2), layers.Dropout(0.3),
                layers.Conv1D(256, 3, activation="relu", padding="same"),
                layers.GlobalAveragePooling1D(), layers.Dropout(0.4),
                layers.Dense(128, activation="relu"), layers.Dropout(0.3),
                layers.Dense(n_classes, activation="softmax"),
            ])
        else:
            m = km.Sequential([
                layers.Input(shape=(dim, 1)),
                layers.LSTM(128, return_sequences=True), layers.Dropout(0.3),
                layers.LSTM(64), layers.Dropout(0.3),
                layers.Dense(64, activation="relu"),
                layers.Dense(n_classes, activation="softmax"),
            ])

        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=25,
            restore_best_weights=True, verbose=0,
        )
        t0   = time.time()
        hist = m.fit(X_tr_3d, y_tr, validation_data=(X_val_3d, y_val),
                     epochs=DL_EPOCHS, batch_size=DL_BATCH,
                     callbacks=[es], verbose=1)
        return max(hist.history["val_accuracy"]), time.time() - t0
    except ImportError:
        return None, None


def evaluate_dl(X: np.ndarray, y_enc: np.ndarray,
                model_name: str, n_classes: int) -> tuple[float | None, float | None]:
    """Gọi đúng backend DL đang khả dụng."""
    if DL_BACKEND == "torch":
        return _train_torch(X, y_enc, _TORCH_BUILDERS[model_name], n_classes)
    if DL_BACKEND == "tensorflow":
        return _train_tf(X, y_enc, model_name, n_classes)
    return None, None


def _train_torch_2d(
    X3d: np.ndarray, y_enc: np.ndarray, n_classes: int
) -> tuple[float | None, float | None]:
    """
    Huấn luyện CNN2D với đầu vào 2D giữ temporal info.
    X3d: shape (N, freq_bins, TIME_STEPS)
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        torch.manual_seed(RANDOM_SEED)

        # Normalize per sample (mean/std trên từng spectrogram)
        X_norm = X3d.astype(np.float32)
        mu  = X_norm.mean(axis=(1, 2), keepdims=True)
        sig = X_norm.std(axis=(1, 2), keepdims=True) + 1e-8
        X_norm = (X_norm - mu) / sig

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_norm, y_enc.astype(np.int64),
            test_size=0.2, stratify=y_enc, random_state=RANDOM_SEED,
        )

        # Thêm channel dim → (N, 1, freq_bins, TIME_STEPS)
        def to4d(a: np.ndarray) -> "torch.Tensor":
            return torch.tensor(a).unsqueeze(1)

        tr_dl  = DataLoader(TensorDataset(to4d(X_tr),  torch.tensor(y_tr)),
                            batch_size=DL_BATCH, shuffle=True)
        val_dl = DataLoader(TensorDataset(to4d(X_val), torch.tensor(y_val)),
                            batch_size=DL_BATCH)

        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        freq_bins = X_tr.shape[1]
        model   = _build_cnn2d_torch(freq_bins, n_classes).to(device)  # type: ignore[arg-type]
        opt     = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", patience=4, factor=0.5
        )

        best_acc   = 0.0
        no_improve = 0
        patience   = 25
        t0         = time.time()

        for epoch in range(DL_EPOCHS):
            model.train()
            for xb, yb in tr_dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    correct += (model(xb).argmax(1) == yb).sum().item()
                    total   += yb.size(0)
            val_acc = correct / total
            scheduler.step(val_acc)
            print(f"Epoch {epoch+1}/{DL_EPOCHS}, Val Acc: {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc   = val_acc
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break

        return best_acc, time.time() - t0
    except ImportError:
        return None, None


# ============================================================================
# PHẦN 4: VISUALIZATION
# ============================================================================

def plot_feature_comparison(df: pd.DataFrame,
                             save_path: str = "feature_experience/results_feature_comparison.png") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "So sánh hiệu quả các đặc trưng âm thanh – Speech Emotion Recognition\n"
        f"(Dataset: RAVDESS | Python {sys.version_info.major}.{sys.version_info.minor})",
        fontsize=13, fontweight="bold",
    )

    pivot = df.pivot(index="Feature", columns="Model", values="Accuracy(%)")
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, ax=axes[0], cbar_kws={"label": "Accuracy (%)"})
    axes[0].set_title("Heatmap: Accuracy (%) – Đặc trưng × Mô hình", fontweight="bold")
    axes[0].set_xlabel("Mô hình")
    axes[0].set_ylabel("Đặc trưng")
    axes[0].tick_params(axis="x", rotation=30)

    avg    = df.groupby("Feature")["Accuracy(%)"].mean().sort_values()
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(avg)))
    bars   = axes[1].barh(avg.index, avg.values, color=colors, edgecolor="grey")
    for bar, val in zip(bars, avg.values):
        axes[1].text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Accuracy trung bình (%)")
    axes[1].set_title("Xếp hạng đặc trưng (avg qua các mô hình)", fontweight="bold")
    axes[1].set_xlim(0, max(avg.values) * 1.15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Biểu đồ so sánh → {save_path}")


def plot_confusion_matrix_best(
    feature_cache: dict[str, np.ndarray],
    y_enc: np.ndarray,
    le: LabelEncoder,
    best_row: pd.Series,
    save_path: str = "feature_experience/confusion_matrix_best.png",
) -> None:
    """
    Vẽ confusion matrix cho combination Feature+Model tốt nhất.
    best_row: 1 hàng từ results DataFrame (có cột Feature, Model, Accuracy(%)).
    feature_cache: dict feat_name → X đã trích xuất (tránh extract lại).
    """
    feat_name  = str(best_row["Feature"])
    model_name = str(best_row["Model"])
    best_acc   = float(best_row["Accuracy(%)"])

    X = feature_cache[feat_name]

    # Dựng lại model theo tên
    ml_models = get_ml_models()
    if model_name in ml_models:
        clf = ml_models[model_name]
    else:
        # Fallback: SVM mặc định (trường hợp best là DL – không thể refit nhanh)
        print(f"  [WARN] Best model là DL ({model_name}), "
              "dùng SVM làm fallback cho confusion matrix.")
        clf = ml_models["SVM (RBF)"]
        model_name = "SVM (RBF) [fallback]"

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=RANDOM_SEED,
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    acc   = accuracy_score(y_te, y_pred)
    f1    = f1_score(y_te, y_pred, average="macro", zero_division=0)
    prec  = precision_score(y_te, y_pred, average="macro", zero_division=0)
    rec   = recall_score(y_te, y_pred, average="macro", zero_division=0)

    cm_pct = (confusion_matrix(y_te, y_pred).astype(float)
              / confusion_matrix(y_te, y_pred).sum(axis=1, keepdims=True) * 100)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Tỷ lệ (%)"})
    ax.set_title(
        f"Confusion Matrix – Best: {model_name} + {feat_name}\n"
        f"Acc={acc*100:.1f}%  |  F1={f1*100:.1f}%  |  "
        f"Precision={prec*100:.1f}%  |  Recall={rec*100:.1f}%",
        fontweight="bold", fontsize=10,
    )
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix (best: {model_name} + {feat_name}) → {save_path}")


def plot_sample_features(y: np.ndarray, sr: int = SR,
                          save_path: str = "feature_experience/sample_features_visualization.png") -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Trực quan hóa đặc trưng âm thanh (1 sample)",
                 fontsize=13, fontweight="bold")

    t = np.linspace(0, len(y) / sr, len(y))
    axes[0, 0].plot(t, y, color="steelblue", linewidth=0.5)
    axes[0, 0].set_title("Waveform")
    axes[0, 0].set_xlabel("Thời gian (s)")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    img1 = librosa.display.specshow(mfcc, sr=sr, hop_length=HOP_LENGTH,
                                     x_axis="time", ax=axes[0, 1])
    axes[0, 1].set_title("MFCC (40 coefficients)")
    fig.colorbar(img1, ax=axes[0, 1])

    mel     = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    img2    = librosa.display.specshow(log_mel, sr=sr, hop_length=HOP_LENGTH,
                                       x_axis="time", y_axis="mel", ax=axes[1, 0])
    axes[1, 0].set_title("Log-Mel Spectrogram")
    fig.colorbar(img2, ax=axes[1, 0], format="%+2.0f dB")

    chroma = librosa.feature.chroma_cqt(y=librosa.effects.harmonic(y), sr=sr)
    img3   = librosa.display.specshow(chroma, sr=sr, hop_length=HOP_LENGTH,
                                       x_axis="time", y_axis="chroma", ax=axes[1, 1])
    axes[1, 1].set_title("Chroma Features")
    fig.colorbar(img3, ax=axes[1, 1])

    zcr    = librosa.feature.zero_crossing_rate(y)[0]
    rmse   = librosa.feature.rms(y=y)[0]
    ft     = librosa.frames_to_time(np.arange(len(zcr)), sr=sr, hop_length=HOP_LENGTH)
    axes[2, 0].plot(ft, zcr,  label="ZCR",  color="orange")
    axes[2, 0].plot(ft, rmse, label="RMSE", color="green")
    axes[2, 0].set_title("ZCR & RMSE")
    axes[2, 0].set_xlabel("Thời gian (s)")
    axes[2, 0].legend()

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    axes[2, 1].semilogy(ft, centroid, color="purple")
    axes[2, 1].set_title("Spectral Centroid")
    axes[2, 1].set_xlabel("Thời gian (s)")
    axes[2, 1].set_ylabel("Hz")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Feature visualization → {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    print("\n" + "=" * 65)
    print("  THỰC NGHIỆM SO SÁNH ĐẶC TRƯNG ÂM THANH – SER")
    print(f"  Python {sys.version_info.major}.{sys.version_info.minor}"
          f"  |  DL: {DL_BACKEND}")
    print("=" * 65)

    os.makedirs("feature_experience", exist_ok=True)

    signals, labels, _ = load_ravdess(RAVDESS_PATH)
    le    = LabelEncoder()
    y_enc = le.fit_transform(labels)
    n_cls = int(len(le.classes_))

    print("\n[INFO] Phân phối cảm xúc:")
    for emotion, count in zip(*np.unique(labels, return_counts=True)):
        print(f"  {emotion:12s}: {count:4d} samples")

    print("\n[INFO] Đang vẽ feature visualization...")
    plot_sample_features(signals[0])

    results: list[dict] = []

    # Cache X đã trích xuất – dùng lại cho confusion matrix (tránh extract 2 lần)
    feature_cache: dict[str, np.ndarray] = {}

    # ── Vòng lặp 1D (ML + DL trên feature vector) ────────────────────────────
    for feat_name, extractor in FEATURE_EXTRACTORS.items():
        print(f"\n{'─'*55}")
        print(f"  ĐẶC TRƯNG: {feat_name}")
        print(f"{'─'*55}")

        t0 = time.time()
        X  = build_feature_matrix(signals, extractor, feat_name)
        feature_cache[feat_name] = X
        print(f"  Số chiều: {X.shape[1]:4d}  |  Trích xuất: {time.time()-t0:.1f}s")

        # ML – giờ dùng dict metrics (Điểm 5)
        for model_name, model in get_ml_models().items():
            metrics = evaluate_ml(X, y_enc, model)
            print(
                f"  [{model_name:18s}]  "
                f"Acc={metrics['accuracy_mean']*100:5.1f}%±{metrics['accuracy_std']*100:.1f}%  "
                f"F1={metrics['f1_mean']*100:.1f}%  "
                f"P={metrics['precision_mean']*100:.1f}%  "
                f"R={metrics['recall_mean']*100:.1f}%"
            )
            results.append({
                "Feature":        feat_name,
                "Model":          model_name,
                "Accuracy(%)":    round(metrics["accuracy_mean"] * 100, 2),
                "Std(%)":         round(metrics["accuracy_std"]  * 100, 2),
                "F1_macro(%)":    round(metrics["f1_mean"]        * 100, 2),
                "Precision(%)":   round(metrics["precision_mean"] * 100, 2),
                "Recall(%)":      round(metrics["recall_mean"]    * 100, 2),
                "Dim":            X.shape[1],
                "Category":       "ML",
            })

        # DL 1D (CNN1D / LSTM trên feature vector)
        if DL_BACKEND != "none":
            for dl_name in _TORCH_BUILDERS:
                acc, elapsed = evaluate_dl(X, y_enc, dl_name, n_cls)
                if acc is not None:
                    print(f"  [{dl_name:18s}]  Acc={acc*100:5.1f}%  ({elapsed:.0f}s)"
                          "  [NOTE: feature vector – temporal info đã bị gộp]")
                    results.append({
                        "Feature":      feat_name,
                        "Model":        dl_name,
                        "Accuracy(%)":  round(acc * 100, 2),
                        "Std(%)":       0.0,
                        "F1_macro(%)":  0.0,   # DL không tính CV F1 ở đây
                        "Precision(%)"  : 0.0,
                        "Recall(%)":    0.0,
                        "Dim":          X.shape[1],
                        "Category":     "DL-1D",
                    })

    # ── Vòng lặp 2D (CNN2D giữ temporal info – Điểm 2) ───────────────────────
    if DL_BACKEND == "torch":
        print(f"\n{'─'*55}")
        print("  [CÁCH TIẾP CẬN MỚI] CNN2D trên spectrogram 2D (giữ temporal info)")
        print(f"{'─'*55}")

        for feat_name, extractor in FEATURE_EXTRACTORS_2D.items():
            t0   = time.time()
            X3d  = build_feature_matrix_2d(signals, extractor, feat_name)
            # Lưu vào cache dạng 1D flatten để dùng với ML nếu cần
            feature_cache[feat_name] = X3d.reshape(X3d.shape[0], -1)
            freq_bins = X3d.shape[1]
            print(f"\n  {feat_name}: shape={X3d.shape}  |  Trích xuất: {time.time()-t0:.1f}s")

            acc, elapsed = _train_torch_2d(X3d, y_enc, n_cls)
            if acc is not None:
                print(f"  [CNN2D           ]  Acc={acc*100:5.1f}%  ({elapsed:.0f}s)"
                      "  [temporal info được giữ nguyên ✓]")
                results.append({
                    "Feature":      feat_name,
                    "Model":        "CNN2D",
                    "Accuracy(%)":  round(acc * 100, 2),
                    "Std(%)":       0.0,
                    "F1_macro(%)":  0.0,
                    "Precision(%)": 0.0,
                    "Recall(%)":    0.0,
                    "Dim":          freq_bins * TIME_STEPS,
                    "Category":     "DL-2D",
                })

    # ── Tổng hợp kết quả ─────────────────────────────────────────────────────
    df    = pd.DataFrame(results)
    pivot = df.pivot_table(index="Feature", columns="Model",
                           values="Accuracy(%)", aggfunc="mean").round(1)

    print("\n" + "=" * 65)
    print("  KẾT QUẢ TỔNG HỢP – ACCURACY (%)")
    print("=" * 65)
    print(pivot.to_string())

    # In thêm bảng F1 (chỉ cho ML rows có F1 thực)
    ml_df = df[df["F1_macro(%)"] > 0]
    if not ml_df.empty:
        pivot_f1 = ml_df.pivot_table(
            index="Feature", columns="Model", values="F1_macro(%)", aggfunc="mean"
        ).round(1)
        print("\n  KẾT QUẢ TỔNG HỢP – F1-MACRO (%)")
        print(pivot_f1.to_string())

    top5 = (df.nlargest(5, "Accuracy(%)")
              [["Feature", "Model", "Accuracy(%)", "Std(%)", "F1_macro(%)"]]
              .reset_index(drop=True))
    top5.index += 1
    print(f"\n Top 5 kết hợp tốt nhất:")
    print(top5.to_string())

    df.to_csv("results_full.csv", index=False)
    pivot.to_csv("results_pivot.csv")
    print("\n[INFO] Đã lưu: results_full.csv, results_pivot.csv")

    plot_feature_comparison(df)

    # ── Confusion matrix – best model thật sự (Điểm 6) ───────────────────────
    # Chỉ lấy best trong ML rows (có thể refit đơn giản)
    ml_rows = df[df["Category"] == "ML"]
    if not ml_rows.empty:
        best_row = ml_rows.loc[ml_rows["Accuracy(%)"].idxmax()]
        print(
            f"\n[BEST] Feature={best_row['Feature']}  |  Model={best_row['Model']}  |  "
            f"Accuracy={best_row['Accuracy(%)']:.1f}%  |  F1={best_row['F1_macro(%)']:.1f}%"
        )
        plot_confusion_matrix_best(feature_cache, y_enc, le, best_row)
    else:
        print("[WARN] Không có ML results để vẽ confusion matrix.")

    print("\n" + "=" * 65)
    print("  HOÀN THÀNH! Output:")
    print("  • results_full.csv")
    print("  • results_pivot.csv")
    print("  • feature_experience/results_feature_comparison.png")
    print("  • feature_experience/confusion_matrix_best.png")
    print("  • feature_experience/sample_features_visualization.png")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
