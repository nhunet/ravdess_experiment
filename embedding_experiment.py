"""
=============================================================================
THỰC NGHIỆM MỤC 2 – PRETRAINED AUDIO EMBEDDINGS CHO SPEECH EMOTION RECOGNITION
=============================================================================

Mục tiêu:
  So sánh 4 pretrained audio embedding cho bài toán SER trên RAVDESS:
    1. Wav2Vec 2.0  (facebook/wav2vec2-base)      – 768 chiều
    2. HuBERT       (facebook/hubert-base-ls960)   – 768 chiều
    3. Whisper      (openai/whisper-base encoder)  – 512 chiều
    4. MERT         (m-a-p/MERT-v1-95M)           – 768 chiều

  Trên 2 nhóm mô hình phân loại:
    - ML:  SVM (RBF), Random Forest, KNN
    - DL:  MLP Classifier head (PyTorch ưu tiên, TF fallback)

  Và so sánh với kết quả mục 1 (hand-crafted: MFCC, Log-Mel, LPCC, ...)

Tương thích: Python 3.9 – 3.14
  - torch >= 2.1  (DL ưu tiên, hỗ trợ Python 3.14 từ v2.5+)
  - tensorflow >= 2.13  (fallback, Python 3.8–3.12)
  - transformers >= 4.35

Cài đặt:
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
  pip install transformers librosa scikit-learn numpy pandas matplotlib seaborn

Dataset: RAVDESS – https://zenodo.org/record/1188976
  Giải nén vào: ./RAVDESS/
=============================================================================
"""

from __future__ import annotations      # PEP 563 – safe type hints Python 3.9+

import os
import sys
import time
import warnings
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, f1_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

warnings.filterwarnings("ignore")

# ── Python version guard ─────────────────────────────────────────────────────
if sys.version_info < (3, 9):
    raise RuntimeError(f"Cần Python >= 3.9, hiện tại: {sys.version}")

# ── DL backend detection ─────────────────────────────────────────────────────
def _detect_dl() -> str:
    if importlib.util.find_spec("torch") is not None:
        return "torch"
    if importlib.util.find_spec("tensorflow") is not None:
        if sys.version_info >= (3, 13):
            print("[WARN] TensorFlow chưa hỗ trợ Python "
                  f"{sys.version_info.major}.{sys.version_info.minor}. "
                  "Cài PyTorch: pip install torch --index-url "
                  "https://download.pytorch.org/whl/cpu")
            return "none"
        return "tensorflow"
    return "none"

DL_BACKEND = _detect_dl()

print(f"[INFO] Python {sys.version_info.major}.{sys.version_info.minor}"
      f".{sys.version_info.micro}")
print(f"[INFO] DL backend : {DL_BACKEND if DL_BACKEND != 'none' else 'không có – chỉ ML'}")

# ── Hằng số ──────────────────────────────────────────────────────────────────
RAVDESS_PATH : str   = "./RAVDESS"
SR           : int   = 16000          # Wav2Vec2 / HuBERT / Whisper đều cần 16 kHz
DURATION     : float = 3.0
RANDOM_SEED  : int   = 42
DL_EPOCHS    : int   = 50
DL_BATCH     : int   = 32
CACHE_DIR    : str   = "./embedding_cache"    # Lưu embedding đã tính để tránh tính lại

# ── Layer selection cho Wav2Vec2 / HuBERT ────────────────────────────────────
# Nghiên cứu (SUPERB benchmark, Yang et al. 2021) cho thấy:
#   - Layer cuối (None)  → tốt cho ASR (nhận dạng từ)
#   - Layer giữa (6–9)   → tốt hơn cho emotion, speaker ID (paralinguistic)
#   - "weighted_sum"     → học trọng số α cho từng layer (tốt nhất, cần fine-tune)
# Với SER: dùng layer 9 (middle-upper) là lựa chọn tốt theo literature.
# Đặt None để dùng last layer (default), hoặc int 0–11 để chọn layer cụ thể.
LAYER_IDX: int | None = 9     # None = last layer, 9 = layer thứ 10 (0-indexed)

EMOTIONS: dict[int, str] = {
    1: "neutral", 2: "calm",    3: "happy",    4: "sad",
    5: "angry",   6: "fearful", 7: "disgust",  8: "surprised",
}

# Kết quả mục 1 (MFCC+SVM) để so sánh – cập nhật sau khi chạy mục 1
BASELINE_RESULTS: dict[str, float] = {
    "MFCC + SVM":             68.5,   # từ kết quả mục 1 của em
    "Log-Mel + SVM":          0.0,    # cập nhật sau khi chạy
    "LPCC + SVM":             0.0,
}

np.random.seed(RANDOM_SEED)
os.makedirs(CACHE_DIR, exist_ok=True)


# ── Cấu hình từng embedding ───────────────────────────────────────────────────
EMBEDDING_CONFIGS: dict[str, dict] = {
    "Wav2Vec2": {
        "model_id"  : "facebook/wav2vec2-base",
        "model_type": "wav2vec2",
        "dim"       : 768,
        "sr"        : 16000,
        "desc"      : "Self-supervised speech model (Facebook), trained on LibriSpeech 960h",
    },
    "HuBERT": {
        "model_id"  : "facebook/hubert-base-ls960",
        "model_type": "hubert",
        "dim"       : 768,
        "sr"        : 16000,
        "desc"      : "Hidden-Unit BERT for speech (Facebook), masked prediction pretraining",
    },
    "Whisper": {
        "model_id"  : "openai/whisper-base",
        "model_type": "whisper",
        "dim"       : 512,
        "sr"        : 16000,
        "desc"      : "Encoder của Whisper ASR (OpenAI), trained on 680k hours multilingual",
    },
    "MERT": {
        "model_id"  : "m-a-p/MERT-v1-95M",
        "model_type": "mert",
        "dim"       : 768,
        "sr"        : 24000,   # MERT yêu cầu 24 kHz (khác các model speech khác)
        "desc"      : "Music undERstanding Transformer (m-a-p), 95M params, "
                      "trained on music audio với RVQ-VAE + CQT teacher",
    },
}


# ============================================================================
# PHẦN 1: TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU
# ============================================================================

def load_audio(file_path: str,
               sr: int = SR,
               duration: float = DURATION) -> np.ndarray:
    """
    Tải file audio, chuẩn hóa độ dài và biên độ.
    Resample về sr=16000 Hz (yêu cầu của các speech model).
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
# PHẦN 2: TRÍCH XUẤT PRETRAINED EMBEDDING
# ============================================================================

class EmbeddingExtractor:
    """
    Wrapper thống nhất cho 3 pretrained speech model.

    Cách hoạt động:
      1. Load model + processor từ HuggingFace (lần đầu tự download ~400MB)
      2. Chạy forward pass, lấy hidden states của encoder
      3. Mean pooling theo time dimension → vector cố định
      4. Cache embedding ra disk để tránh tính lại

    Kết quả: vector đặc trưng (embedding) biểu diễn nội dung âm thanh theo
    cách model pretrained đã học từ hàng trăm nghìn giờ dữ liệu.
    """

    def __init__(self, name: str, config: dict) -> None:
        self.name       = name
        self.config     = config
        self.model_id   = config["model_id"]
        self.model_type = config["model_type"]
        self.dim        = config["dim"]
        self._model     = None
        self._processor = None

    def _load_model(self) -> None:
        """Load model lên RAM (lazy loading – chỉ load khi cần)."""
        if self._model is not None:
            return

        print(f"  [INFO] Đang load {self.name} ({self.model_id})...")
        print(f"         Lần đầu sẽ download ~350-400MB từ HuggingFace...")

        try:
            if self.model_type == "wav2vec2":
                from transformers import Wav2Vec2Processor, Wav2Vec2Model
                self._processor = Wav2Vec2Processor.from_pretrained(self.model_id)
                self._model     = Wav2Vec2Model.from_pretrained(self.model_id)

            elif self.model_type == "hubert":
                from transformers import HubertModel
                from transformers import Wav2Vec2FeatureExtractor
                self._processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_id)
                self._model     = HubertModel.from_pretrained(self.model_id)

            elif self.model_type == "whisper":
                from transformers import WhisperProcessor, WhisperModel
                self._processor = WhisperProcessor.from_pretrained(self.model_id)
                self._model     = WhisperModel.from_pretrained(self.model_id)

            elif self.model_type == "mert":
                # MERT dùng AutoProcessor + AutoModel từ HuggingFace
                # Lưu ý: MERT yêu cầu sample rate 24000 Hz
                from transformers import AutoProcessor, AutoModel
                self._processor = AutoProcessor.from_pretrained(
                    self.model_id, trust_remote_code=True
                )
                self._model = AutoModel.from_pretrained(
                    self.model_id, trust_remote_code=True
                )

            # Chuyển sang eval mode, tắt gradient (không fine-tune)
            self._model.eval()
            print(f"  [INFO] Load {self.name} thành công.")

        except Exception as exc:
            raise RuntimeError(
                f"Không thể load {self.name}.\n"
                f"  Lỗi: {exc}\n"
                f"  Kiểm tra kết nối mạng và cài đặt:\n"
                f"  pip install transformers torch torchaudio"
            ) from exc

    def _extract_one(self, y: np.ndarray) -> np.ndarray:
        """
        Trích xuất embedding cho 1 file audio.
        Trả về vector 1D sau mean pooling.
        """
        import torch

        with torch.no_grad():
            if self.model_type in ("wav2vec2", "hubert"):
                inputs = self._processor(
                    y, sampling_rate=SR,
                    return_tensors="pt", padding=True
                )
                # output_hidden_states=True → lấy được tất cả layer
                outputs = self._model(**inputs, output_hidden_states=True)

                if LAYER_IDX is None:
                    # Dùng last hidden state (default)
                    hidden = outputs.last_hidden_state   # (1, T, 768)
                else:
                    # Chọn layer cụ thể theo SUPERB benchmark recommendation
                    # hidden_states[0] = embedding layer, [1..12] = transformer layers
                    hidden = outputs.hidden_states[LAYER_IDX + 1]  # (1, T, 768)

                # Mean pooling theo time axis → (768,)
                emb = hidden.squeeze(0).mean(dim=0)

            elif self.model_type == "whisper":
                # Whisper cần mel spectrogram làm input
                inputs = self._processor(
                    y, sampling_rate=SR,
                    return_tensors="pt"
                )
                # Encoder của Whisper xử lý audio feature
                encoder_out = self._model.encoder(
                    inputs["input_features"]
                )
                # last_hidden_state: (1, time_steps, 512)
                emb = encoder_out.last_hidden_state.squeeze(0).mean(dim=0)

            elif self.model_type == "mert":
                # MERT cần sample rate 24000 Hz – resample nếu cần
                mert_sr = self.config["sr"]    # 24000
                if SR != mert_sr:
                    import librosa as _lb
                    y_mert = _lb.resample(y, orig_sr=SR, target_sr=mert_sr)
                else:
                    y_mert = y
                inputs = self._processor(
                    y_mert, sampling_rate=mert_sr,
                    return_tensors="pt"
                )
                outputs = self._model(**inputs, output_hidden_states=True)
                # MERT trả về 13 lớp hidden states (như BERT)
                # Lấy trung bình tất cả các lớp → richer representation
                all_layers = torch.stack(outputs.hidden_states)   # (13, 1, T, 768)
                emb = all_layers.mean(dim=0).squeeze(0).mean(dim=0)  # (768,)

        return emb.cpu().numpy().astype(np.float32)

    def extract_all(self, signals: list[np.ndarray],
                    cache_key: str = "") -> np.ndarray:
        """
        Trích xuất embedding cho toàn bộ dataset.
        Dùng cache nếu đã tính trước để tiết kiệm thời gian.
        """
        cache_file = Path(CACHE_DIR) / f"{cache_key}_{self.name}.npy"

        # Thử load từ cache
        if cache_file.exists():
            print(f"  [CACHE] Load {self.name} embedding từ cache: {cache_file}")
            return np.load(str(cache_file))

        # Chưa có cache → tính mới
        self._load_model()

        embeddings: list[np.ndarray] = []
        t0        = time.time()
        errors    = 0
        n         = len(signals)

        print(f"  [INFO] Đang trích xuất {self.name} embedding cho {n} samples...")

        for i, y in enumerate(signals):
            try:
                emb = self._extract_one(y)
                embeddings.append(emb)
            except Exception as exc:
                errors += 1
                # Fallback: vector zero
                embeddings.append(np.zeros(self.dim, dtype=np.float32))
                if errors <= 3:
                    print(f"    [WARN] Sample {i}: {exc}")

            # Progress mỗi 100 samples
            if (i + 1) % 100 == 0 or (i + 1) == n:
                elapsed = time.time() - t0
                eta     = elapsed / (i + 1) * (n - i - 1)
                print(f"    {i+1}/{n}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s còn lại)")

        X = np.array(embeddings)
        elapsed_total = time.time() - t0
        print(f"  [INFO] {self.name}: {n} embeddings, "
              f"dim={X.shape[1]}, "
              f"thời gian={elapsed_total:.1f}s, "
              f"lỗi={errors}")

        # Lưu cache
        np.save(str(cache_file), X)
        print(f"  [CACHE] Đã lưu → {cache_file}")

        # Giải phóng VRAM / RAM
        self._model     = None
        self._processor = None

        return X


# ============================================================================
# PHẦN 3: CÁC MÔ HÌNH PHÂN LOẠI
# ============================================================================

# ── ML ───────────────────────────────────────────────────────────────────────

def get_ml_models() -> dict[str, Pipeline]:
    """
    Pipeline cho từng ML model:
      StandardScaler  → scale từng feature về mean=0, std=1
      Normalizer(L2)  → normalize toàn bộ vector embedding về unit norm
                        (quan trọng với embedding space – cosine similarity)
      Classifier      → SVM / RF / KNN
    """
    return {
        "SVM (RBF)": Pipeline([
            ("scaler",     StandardScaler()),
            ("normalizer", Normalizer(norm="l2")),
            ("clf",        SVC(kernel="rbf", C=10, gamma="scale",
                               probability=True, random_state=RANDOM_SEED)),
        ]),
        "Random Forest": Pipeline([
            ("scaler",     StandardScaler()),
            ("normalizer", Normalizer(norm="l2")),
            ("clf",        RandomForestClassifier(
                n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1
            )),
        ]),
        "KNN (k=5)": Pipeline([
            ("scaler",     StandardScaler()),
            ("normalizer", Normalizer(norm="l2")),
            ("clf",        KNeighborsClassifier(n_neighbors=5, metric="cosine")),
        ]),
    }


def evaluate_ml(X: np.ndarray, y_enc: np.ndarray,
                model: Pipeline,
                cv: int = 5) -> tuple[float, float, float, float]:
    """
    5-fold Stratified Cross-Validation.

    Trả về: (acc_mean, acc_std, f1_macro_mean, f1_weighted_mean)

    Tại sao cần cả 3 metric?
      - Accuracy       : dễ hiểu, nhưng bị ảnh hưởng bởi imbalance
      - F1-macro       : trung bình F1 từng lớp KHÔNG tính class size
                         → công bằng với cảm xúc hiếm (neutral chỉ 96 samples)
      - F1-weighted    : trung bình F1 CÓ tính class size
                         → phản ánh overall performance thực tế hơn accuracy

    Với RAVDESS: neutral=96, còn lại=192 → imbalance 1:2 → cần F1-macro
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)

    acc      = cross_val_score(model, X, y_enc, cv=skf,
                               scoring="accuracy",    n_jobs=-1)
    f1_mac   = cross_val_score(model, X, y_enc, cv=skf,
                               scoring="f1_macro",    n_jobs=-1)
    f1_wt    = cross_val_score(model, X, y_enc, cv=skf,
                               scoring="f1_weighted", n_jobs=-1)

    return (float(acc.mean()), float(acc.std()),
            float(f1_mac.mean()), float(f1_wt.mean()))


# ── PyTorch MLP head (Python 3.9 – 3.14) ─────────────────────────────────────

def _build_mlp_torch(input_dim: int, n_classes: int) -> object:
    """
    MLP Classifier head – phù hợp nhất để phân loại embedding.
    Không cần conv hay LSTM vì embedding đã là vector ngữ nghĩa cao cấp.

    Architecture:
      Linear(dim→512) → BN → ReLU → Dropout(0.4)
      Linear(512→256) → BN → ReLU → Dropout(0.3)
      Linear(256→128) → ReLU → Dropout(0.2)
      Linear(128→n_classes)
    """
    import torch.nn as nn

    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),

                nn.Linear(512, 256),
                nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),

                nn.Linear(256, 128),
                nn.ReLU(), nn.Dropout(0.2),

                nn.Linear(128, n_classes),
            )

        def forward(self, x):   # type: ignore[override]
            return self.net(x)

    return MLP()


def _train_mlp_torch(X: np.ndarray, y_enc: np.ndarray,
                     n_classes: int) -> tuple[float | None, float | None]:
    """Huấn luyện MLP bằng PyTorch với early stopping."""
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

        tr_dl = DataLoader(
            TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
            batch_size=DL_BATCH, shuffle=True,
        )
        val_dl = DataLoader(
            TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
            batch_size=DL_BATCH,
        )

        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model   = _build_mlp_torch(X_tr.shape[1], n_classes).to(device)
        opt     = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=DL_EPOCHS)
        loss_fn = nn.CrossEntropyLoss()

        best_acc   = 0.0
        no_improve = 0
        patience   = 10
        t0         = time.time()

        for epoch in range(DL_EPOCHS):
            model.train()
            for xb, yb in tr_dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()
            sched.step()

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    correct += (model(xb).argmax(1) == yb).sum().item()
                    total   += yb.size(0)

            val_acc = correct / total
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


# ── TensorFlow MLP fallback ───────────────────────────────────────────────────

def _train_mlp_tf(X: np.ndarray, y_enc: np.ndarray,
                  n_classes: int) -> tuple[float | None, float | None]:
    """MLP bằng TensorFlow/Keras (fallback cho Python < 3.13)."""
    try:
        import tensorflow as tf
        tf.random.set_seed(RANDOM_SEED)

        sc = StandardScaler()
        X_s = sc.fit_transform(X).astype(np.float32)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_s, y_enc, test_size=0.2,
            stratify=y_enc, random_state=RANDOM_SEED,
        )
        dim = X_tr.shape[1]

        from tensorflow.keras import layers, models as km
        m = km.Sequential([
            layers.Input(shape=(dim,)),
            layers.Dense(512), layers.BatchNormalization(),
            layers.ReLU(), layers.Dropout(0.4),
            layers.Dense(256), layers.BatchNormalization(),
            layers.ReLU(), layers.Dropout(0.3),
            layers.Dense(128), layers.ReLU(), layers.Dropout(0.2),
            layers.Dense(n_classes, activation="softmax"),
        ])
        m.compile(optimizer=tf.keras.optimizers.AdamW(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10,
            restore_best_weights=True, verbose=0,
        )
        t0   = time.time()
        hist = m.fit(X_tr, y_tr,
                     validation_data=(X_val, y_val),
                     epochs=DL_EPOCHS, batch_size=DL_BATCH,
                     callbacks=[es], verbose=0)
        return max(hist.history["val_accuracy"]), time.time() - t0
    except ImportError:
        return None, None


def evaluate_dl_mlp(X: np.ndarray, y_enc: np.ndarray,
                    n_classes: int) -> tuple[float | None, float | None]:
    """Gọi đúng backend DL khả dụng."""
    if DL_BACKEND == "torch":
        return _train_mlp_torch(X, y_enc, n_classes)
    if DL_BACKEND == "tensorflow":
        return _train_mlp_tf(X, y_enc, n_classes)
    return None, None


# ============================================================================
# PHẦN 4: VISUALIZATION
# ============================================================================

def plot_embedding_comparison(results_df: pd.DataFrame,
                               baseline_df: pd.DataFrame | None = None,
                               save_path: str = "results_embedding_comparison.png"
                               ) -> None:
    """
    3 biểu đồ:
      (a) Heatmap: Embedding × Model
      (b) Bar chart xếp hạng embedding (avg)
      (c) So sánh với baseline hand-crafted features (nếu có)
    """
    has_baseline = baseline_df is not None and not baseline_df.empty
    ncols = 3 if has_baseline else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 7))
    fig.suptitle(
        "So sánh Pretrained Audio Embeddings – Speech Emotion Recognition\n"
        f"(Dataset: RAVDESS | Python "
        f"{sys.version_info.major}.{sys.version_info.minor})",
        fontsize=13, fontweight="bold",
    )

    # (a) Heatmap
    pivot = results_df.pivot(index="Embedding", columns="Model",
                             values="Accuracy(%)")
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Blues",
                linewidths=0.5, ax=axes[0],
                cbar_kws={"label": "Accuracy (%)"})
    axes[0].set_title("(a) Accuracy (%) – Embedding × Mô hình", fontweight="bold")
    axes[0].set_xlabel("Mô hình")
    axes[0].set_ylabel("Pretrained Embedding")
    axes[0].tick_params(axis="x", rotation=30)

    # (b) Bar chart xếp hạng embedding
    avg    = results_df.groupby("Embedding")["Accuracy(%)"].mean().sort_values()
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(avg)))
    bars   = axes[1].barh(avg.index, avg.values, color=colors, edgecolor="grey")
    for bar, val in zip(bars, avg.values):
        axes[1].text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}%", va="center", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Accuracy trung bình (%)")
    axes[1].set_title("(b) Xếp hạng Embedding (avg qua các mô hình)",
                       fontweight="bold")
    axes[1].set_xlim(0, max(avg.values) * 1.15)

    # (c) So sánh với hand-crafted baseline
    if has_baseline:
        all_results = pd.concat([
            results_df.assign(Group="Pretrained Embedding"),
            baseline_df.assign(Group="Hand-crafted Feature"),
        ], ignore_index=True)
        all_avg = all_results.groupby(["Feature/Embedding", "Group"])[
            "Accuracy(%)"
        ].mean().reset_index()
        all_avg = all_avg.sort_values("Accuracy(%)", ascending=True)

        colors_map = {"Pretrained Embedding": "#2196F3",
                      "Hand-crafted Feature": "#FF9800"}
        bar_colors = [colors_map[g] for g in all_avg["Group"]]
        axes[2].barh(all_avg["Feature/Embedding"], all_avg["Accuracy(%)"],
                     color=bar_colors, edgecolor="grey")
        for i, (_, row) in enumerate(all_avg.iterrows()):
            axes[2].text(row["Accuracy(%)"] + 0.3, i,
                         f"{row['Accuracy(%)']:.1f}%",
                         va="center", fontsize=9, fontweight="bold")

        from matplotlib.patches import Patch
        legend = [Patch(color="#2196F3", label="Pretrained Embedding"),
                  Patch(color="#FF9800", label="Hand-crafted (Mục 1)")]
        axes[2].legend(handles=legend, loc="lower right")
        axes[2].set_xlabel("Accuracy trung bình (%)")
        axes[2].set_title("(c) Embedding vs Hand-crafted Features", fontweight="bold")
        axes[2].set_xlim(0, max(all_avg["Accuracy(%)"]) * 1.15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Biểu đồ so sánh → {save_path}")


def plot_confusion_matrix(X: np.ndarray, y_enc: np.ndarray,
                           le: LabelEncoder,
                           embed_name: str,
                           save_path: str = "") -> None:
    """Confusion matrix cho mô hình SVM + embedding tốt nhất."""
    if not save_path:
        save_path = f"confusion_{embed_name.lower()}_svm.png"

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=RANDOM_SEED,
    )
    clf = Pipeline([("scaler", StandardScaler()),
                    ("clf", SVC(kernel="rbf", C=10, gamma="scale",
                                random_state=RANDOM_SEED))])
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    # In classification report
    print(f"\n  Classification Report – {embed_name} + SVM:")
    print(classification_report(y_te, y_pred, target_names=le.classes_,
                                 digits=3))

    cm     = confusion_matrix(y_te, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Tỷ lệ (%)"})
    ax.set_title(
        f"Confusion Matrix – {embed_name} + SVM\n"
        f"Accuracy = {accuracy_score(y_te, y_pred)*100:.1f}%",
        fontweight="bold",
    )
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [INFO] Confusion matrix → {save_path}")


def plot_embedding_space(embeddings_dict: dict[str, np.ndarray],
                          y_enc: np.ndarray,
                          le: LabelEncoder,
                          save_path: str = "embedding_tsne.png") -> None:
    """
    Visualize không gian embedding bằng t-SNE (2D).
    Giúp thấy trực quan các cụm cảm xúc trong không gian embedding.
    """
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA

        n_embeds = len(embeddings_dict)
        fig, axes = plt.subplots(1, n_embeds, figsize=(7 * n_embeds, 6))
        if n_embeds == 1:
            axes = [axes]

        fig.suptitle("t-SNE Visualization – Không gian Embedding 2D\n"
                     "(mỗi màu = 1 cảm xúc)", fontsize=13, fontweight="bold")

        colors  = plt.cm.Set1(np.linspace(0, 1, len(le.classes_)))
        n_per   = 200    # lấy tối đa 200 samples để t-SNE nhanh hơn

        for ax, (name, X) in zip(axes, embeddings_dict.items()):
            # Lấy subset để t-SNE nhanh
            idx = np.random.choice(len(X), min(n_per * len(le.classes_), len(X)),
                                   replace=False)
            X_sub  = X[idx]
            y_sub  = y_enc[idx]

            # Giảm chiều bằng PCA trước (nếu dim > 50) rồi mới t-SNE
            if X_sub.shape[1] > 50:
                X_sub = PCA(n_components=50,
                            random_state=RANDOM_SEED).fit_transform(X_sub)

            X_2d = TSNE(n_components=2, perplexity=30, random_state=RANDOM_SEED,
                        n_iter=1000).fit_transform(X_sub)

            for cls_idx, cls_name in enumerate(le.classes_):
                mask = y_sub == cls_idx
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                           c=[colors[cls_idx]], label=cls_name,
                           alpha=0.7, s=25, edgecolors="none")

            ax.set_title(f"{name}\n(dim={X.shape[1]})", fontweight="bold")
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.legend(fontsize=7, markerscale=1.5, loc="best")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[INFO] t-SNE visualization → {save_path}")

    except Exception as exc:
        print(f"[WARN] Không thể vẽ t-SNE: {exc}")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    print("\n" + "=" * 65)
    print("  THỰC NGHIỆM MỤC 2 – PRETRAINED AUDIO EMBEDDINGS + SER")
    print(f"  Python {sys.version_info.major}.{sys.version_info.minor}"
          f"  |  DL: {DL_BACKEND}")
    print("=" * 65)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    signals, labels, _ = load_ravdess(RAVDESS_PATH)
    le    = LabelEncoder()
    y_enc = le.fit_transform(labels)
    n_cls = int(len(le.classes_))

    print(f"\n[INFO] Phân phối cảm xúc:")
    for emotion, count in zip(*np.unique(labels, return_counts=True)):
        print(f"  {emotion:12s}: {count:4d} samples")

    # ── 2. Trích xuất embedding + đánh giá ───────────────────────────────────
    results: list[dict]                   = []
    embeddings_dict: dict[str, np.ndarray] = {}
    cache_key = f"ravdess_{len(signals)}"

    for embed_name, config in EMBEDDING_CONFIGS.items():
        print(f"\n{'─'*65}")
        print(f"  EMBEDDING: {embed_name}")
        print(f"  Model    : {config['model_id']}")
        print(f"  Mô tả    : {config['desc']}")
        print(f"{'─'*65}")

        # Trích xuất embedding
        extractor = EmbeddingExtractor(embed_name, config)
        try:
            X = extractor.extract_all(signals, cache_key=cache_key)
        except RuntimeError as exc:
            print(f"  [ERROR] {exc}")
            print(f"  → Bỏ qua {embed_name}, chuyển sang embedding tiếp theo.")
            continue

        embeddings_dict[embed_name] = X
        print(f"  Embedding shape: {X.shape}")

        # ML models
        for model_name, model in get_ml_models().items():
            acc_mean, acc_std, f1_mac, f1_wt = evaluate_ml(X, y_enc, model)
            print(
                f"  [{model_name:18s}]  "
                f"Acc={acc_mean*100:5.1f}%±{acc_std*100:.1f}%  "
                f"F1-macro={f1_mac*100:.1f}%  "
                f"F1-wtd={f1_wt*100:.1f}%"
            )
            results.append({
                "Embedding":           embed_name,
                "Feature/Embedding":   embed_name,
                "Model":               model_name,
                "Accuracy(%)":         round(acc_mean * 100, 2),
                "Std(%)":              round(acc_std  * 100, 2),
                "F1_macro(%)":         round(f1_mac   * 100, 2),
                "F1_weighted(%)":      round(f1_wt    * 100, 2),
                "Dim":                 X.shape[1],
                "Category":            "ML",
            })

        # DL MLP head
        # Lưu ý: DL dùng 1 hold-out split (80/20), ML dùng 5-fold CV
        # → Kết quả không hoàn toàn so sánh được. Ghi chú trong report.
        if DL_BACKEND != "none":
            acc, elapsed = evaluate_dl_mlp(X, y_enc, n_cls)
            if acc is not None:
                print(
                    f"  [{'MLP (DL)':18s}]  "
                    f"Acc={acc*100:5.1f}%  ({elapsed:.0f}s)"
                    f"  [hold-out split, không so sánh trực tiếp với CV]"
                )
                results.append({
                    "Embedding":           embed_name,
                    "Feature/Embedding":   embed_name,
                    "Model":               "MLP (DL)",
                    "Accuracy(%)":         round(acc * 100, 2),
                    "Std(%)":              0.0,
                    "F1_macro(%)":         0.0,   # N/A cho DL hold-out
                    "F1_weighted(%)":      0.0,   # N/A cho DL hold-out
                    "Dim":                 X.shape[1],
                    "Category":            "DL",
                })

        # Confusion matrix cho embedding này
        plot_confusion_matrix(X, y_enc, le, embed_name)

    if not results:
        print("\n[ERROR] Không có kết quả nào. Kiểm tra kết nối mạng và cài đặt.")
        return

    # ── 3. Tổng hợp kết quả ──────────────────────────────────────────────────
    df    = pd.DataFrame(results)
    pivot_acc = df.pivot_table(index="Embedding", columns="Model",
                               values="Accuracy(%)", aggfunc="mean").round(1)
    pivot_f1  = df[df["F1_macro(%)"] > 0].pivot_table(
        index="Embedding", columns="Model",
        values="F1_macro(%)", aggfunc="mean"
    ).round(1)

    print("\n" + "=" * 65)
    print("  KẾT QUẢ TỔNG HỢP – ACCURACY (%)")
    print("=" * 65)
    print(pivot_acc.to_string())

    if not pivot_f1.empty:
        print("\n  KẾT QUẢ TỔNG HỢP – F1-MACRO (%)")
        print("  (F1-macro: công bằng với từng cảm xúc, không bị neutral 96 samples kéo lệch)")
        print(pivot_f1.to_string())

    ml_df = df[df["F1_macro(%)"] > 0]
    top5 = (ml_df.nlargest(5, "F1_macro(%)")
                 [["Embedding", "Model", "Accuracy(%)", "F1_macro(%)", "F1_weighted(%)"]]
                 .reset_index(drop=True))
    top5.index += 1
    print(f"\n  Top 5 kết hợp tốt nhất (theo F1-macro):")
    print(top5.to_string())

    # So sánh với mục 1 (baseline)
    baseline_valid = {k: v for k, v in BASELINE_RESULTS.items() if v > 0}
    if baseline_valid:
        print(f"\n  So sánh với Hand-crafted Features (Mục 1):")
        print(f"  {'Feature/Embedding':<30} {'Avg Accuracy':>12}")
        print(f"  {'─'*44}")
        avg_embed = df.groupby("Embedding")["Accuracy(%)"].mean()
        for name, acc in avg_embed.items():
            print(f"  {'[Emb] ' + name:<30} {acc:>11.1f}%")
        for name, acc in baseline_valid.items():
            print(f"  {'[HC]  ' + name:<30} {acc:>11.1f}%")

    # ── 4. Lưu CSV ───────────────────────────────────────────────────────────
    df.to_csv("results_embedding_full.csv", index=False)
    pivot_acc.to_csv("results_embedding_pivot_acc.csv")
    if not pivot_f1.empty:
        pivot_f1.to_csv("results_embedding_pivot_f1.csv")
    print(f"\n[INFO] Đã lưu: results_embedding_full.csv, "
          f"results_embedding_pivot_acc.csv, results_embedding_pivot_f1.csv")

    # ── 5. Visualization ─────────────────────────────────────────────────────
    baseline_df = None
    if baseline_valid:
        baseline_df = pd.DataFrame([
            {"Feature/Embedding": k, "Model": "SVM", "Accuracy(%)": v,
             "Embedding": k, "Std(%)": 0.0, "Dim": 0, "Category": "ML"}
            for k, v in baseline_valid.items()
        ])

    plot_embedding_comparison(df, baseline_df)

    if embeddings_dict:
        plot_embedding_space(embeddings_dict, y_enc, le)

    print("\n" + "=" * 65)
    print("  HOÀN THÀNH! Output files:")
    print("  • results_embedding_full.csv")
    print("  • results_embedding_pivot_acc.csv   – pivot Accuracy")
    print("  • results_embedding_pivot_f1.csv    – pivot F1-macro (ML only)")
    print("  • results_embedding_comparison.png")
    print("  • embedding_tsne.png")
    print("  • confusion_wav2vec2_svm.png")
    print("  • confusion_hubert_svm.png")
    print("  • confusion_whisper_svm.png")
    print("  • confusion_mert_svm.png")
    print("=" * 65 + "\n")

    # ── 6. Gợi ý kết luận ─────────────────────────────────────────────────────
    # Xếp hạng theo F1-macro (ML only) thay vì accuracy
    ml_only = df[df["F1_macro(%)"] > 0]
    if ml_only.empty:
        ml_only = df
    best_embed = ml_only.groupby("Embedding")["F1_macro(%)"].mean().idxmax()
    best_model = ml_only[ml_only["Embedding"] == best_embed].nlargest(1, "F1_macro(%)")
    print(f"  Kết luận: Embedding tốt nhất là [{best_embed}]")
    print(f"  Mô hình tốt nhất: "
          f"{best_model['Model'].values[0]} – "
          f"F1-macro={best_model['F1_macro(%)'].values[0]:.1f}%  "
          f"Acc={best_model['Accuracy(%)'].values[0]:.1f}%")
    print(f"  → Sử dụng [{best_embed}] cho Module Phân tích Cảm xúc")
    print(f"    trong hệ thống gợi ý của luận văn.\n")


if __name__ == "__main__":
    main()
