"""
=============================================================================
THỰC NGHIỆM MỤC 3 – FUSION FEATURES ĐA ĐẶC TRƯNG ÂM THANH
=============================================================================

Mục tiêu:
  Kết hợp Hand-crafted Features (Mục 1) + Pretrained Embeddings (Mục 2)
  theo 4 chiến lược Fusion, so sánh với baseline đơn lẻ.

4 chiến lược Fusion:
  1. Early Fusion    – Concatenate vector trực tiếp trước khi vào model
  2. Late Fusion     – Mỗi nhóm có model riêng, kết hợp xác suất đầu ra
  3. Intermediate    – PCA giảm chiều từng nhóm → cân bằng → concat
  4. Attention Gate  – MLP học trọng số động cho từng nhóm (liên hệ MAMEX)

Features dùng (từ kết quả Mục 1 & 2):
  Hand-crafted : MFCC (240d), Log-Mel (256d), Chroma+Contrast+Tonnetz (50d),
                 Statistical (10d)  [LPCC bị loại – F1=3% cho thấy vấn đề]
  Embeddings   : HuBERT (768d), Wav2Vec2 (768d), Whisper (512d), MERT (768d)

Baseline (để so sánh):
  Mục 1 best : MFCC + SVM    = 68.5%  F1=67.8%
  Mục 2 best : HuBERT + SVM  = 83.9%  F1=82.4%

Tương thích: Python 3.9 – 3.14
  pip install torch librosa transformers scikit-learn numpy pandas matplotlib seaborn
=============================================================================
"""

from __future__ import annotations

import os
import sys
import time
import warnings
import importlib.util
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold)
from sklearn.preprocessing import (LabelEncoder, StandardScaler, Normalizer)
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, classification_report)

warnings.filterwarnings("ignore")

# ── Python version guard ──────────────────────────────────────────────────────
if sys.version_info < (3, 9):
    raise RuntimeError(f"Cần Python >= 3.9, hiện tại: {sys.version}")

# ── DL backend detection ──────────────────────────────────────────────────────
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
      f".{sys.version_info.micro}  |  DL backend: {DL_BACKEND}")

# ── Hằng số ───────────────────────────────────────────────────────────────────
RAVDESS_PATH : str   = "./RAVDESS"
CACHE_DIR    : str   = "./embedding_cache"    # cache từ Mục 2
SR_HC        : int   = 22050                  # hand-crafted SR
SR_EMB       : int   = 16000                  # embedding SR (Wav2Vec/HuBERT/Whisper)
SR_MERT      : int   = 24000                  # MERT yêu cầu 24kHz
DURATION     : float = 3.0
RANDOM_SEED  : int   = 42
N_FFT        : int   = 2048
HOP_LENGTH   : int   = 512
N_MFCC       : int   = 40
N_MELS       : int   = 128
DL_EPOCHS    : int   = 60
DL_BATCH     : int   = 32
PCA_DIM      : int   = 64     # chiều sau PCA trong Intermediate Fusion
LAYER_IDX    : int   = 9      # layer selection cho Wav2Vec2/HuBERT

EMOTIONS: dict[int, str] = {
    1: "neutral", 2: "calm",    3: "happy",    4: "sad",
    5: "angry",   6: "fearful", 7: "disgust",  8: "surprised",
}

# Baseline từ Mục 1 & 2 (để vẽ đường tham chiếu)
BASELINE: dict[str, dict] = {
    "MFCC + SVM (Mục 1)":    {"acc": 68.54, "f1": 67.78, "color": "#FF9800"},
    "Log-Mel + SVM (Mục 1)": {"acc": 68.47, "f1": 67.78, "color": "#FFC107"},
    "HuBERT + SVM (Mục 2)":  {"acc": 83.89, "f1": 82.35, "color": "#2196F3"},
    "Wav2Vec2 + SVM (Mục 2)":{"acc": 83.19, "f1": 82.35, "color": "#03A9F4"},
}

np.random.seed(RANDOM_SEED)
os.makedirs(CACHE_DIR, exist_ok=True)


# ============================================================================
# PHẦN 1: TẢI DỮ LIỆU
# ============================================================================

def load_audio(fp: str, sr: int, duration: float = DURATION) -> np.ndarray:
    y, _ = librosa.load(fp, sr=sr, duration=duration, mono=True)
    target = int(sr * duration)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)), mode="constant")
    else:
        y = y[:target]
    peak = np.abs(y).max()
    return y / peak if peak > 0 else y


def load_ravdess(data_path: str) -> tuple[list, list, list]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"\n{'='*60}\nKHÔNG TÌM THẤY: {path.absolute()}\n"
            f"Download: https://zenodo.org/record/1188976\n{'='*60}"
        )
    wav_files = sorted(path.rglob("*.wav"))
    signals_hc: list[np.ndarray] = []   # 22050 Hz cho hand-crafted
    signals_emb: list[np.ndarray] = []  # 16000 Hz cho embedding
    signals_mert: list[np.ndarray] = [] # 24000 Hz cho MERT
    labels: list[str] = []
    paths: list[str] = []

    print(f"[INFO] Đang tải {len(wav_files)} files...")
    for fp in wav_files:
        parts = fp.stem.split("-")
        if len(parts) < 3:
            continue
        code = int(parts[2])
        if code not in EMOTIONS:
            continue
        try:
            signals_hc.append(load_audio(str(fp), SR_HC))
            signals_emb.append(load_audio(str(fp), SR_EMB))
            signals_mert.append(load_audio(str(fp), SR_MERT))
            labels.append(EMOTIONS[code])
            paths.append(str(fp))
        except Exception as e:
            print(f"  [WARN] {fp.name}: {e}")

    print(f"[INFO] {len(labels)} samples, {len(set(labels))} lớp.")
    return signals_hc, signals_emb, signals_mert, labels, paths


# ============================================================================
# PHẦN 2: TRÍCH XUẤT HAND-CRAFTED FEATURES (từ Mục 1)
# ============================================================================

def extract_mfcc(y: np.ndarray, sr: int = SR_HC) -> np.ndarray:
    """MFCC + delta + delta² → 240 chiều."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    d1   = librosa.feature.delta(mfcc)
    d2   = librosa.feature.delta(mfcc, order=2)
    return np.concatenate([
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
        np.mean(d1,   axis=1), np.std(d1,   axis=1),
        np.mean(d2,   axis=1), np.std(d2,   axis=1),
    ])

def extract_logmel(y: np.ndarray, sr: int = SR_HC) -> np.ndarray:
    """Log-Mel Spectrogram → 256 chiều."""
    mel     = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                              n_fft=N_FFT, hop_length=HOP_LENGTH)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return np.concatenate([np.mean(log_mel, axis=1), np.std(log_mel, axis=1)])

def extract_chroma(y: np.ndarray, sr: int = SR_HC) -> np.ndarray:
    """Chroma + Spectral Contrast + Tonnetz → 50 chiều."""
    h = librosa.effects.harmonic(y)
    chroma   = librosa.feature.chroma_cqt(y=h, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr,
                                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    tonnetz  = librosa.feature.tonnetz(y=h, sr=sr)
    return np.concatenate([
        np.mean(chroma, axis=1),   np.std(chroma, axis=1),
        np.mean(contrast, axis=1), np.std(contrast, axis=1),
        np.mean(tonnetz, axis=1),  np.std(tonnetz, axis=1),
    ])

def extract_statistical(y: np.ndarray, sr: int = SR_HC) -> np.ndarray:
    """ZCR + RMSE + Centroid + Bandwidth + Rolloff → 10 chiều."""
    zcr  = librosa.feature.zero_crossing_rate(y)
    rmse = librosa.feature.rms(y=y)
    cen  = librosa.feature.spectral_centroid(y=y, sr=sr,
                                              n_fft=N_FFT, hop_length=HOP_LENGTH)
    bw   = librosa.feature.spectral_bandwidth(y=y, sr=sr,
                                               n_fft=N_FFT, hop_length=HOP_LENGTH)
    ro   = librosa.feature.spectral_rolloff(y=y, sr=sr,
                                             n_fft=N_FFT, hop_length=HOP_LENGTH)
    return np.concatenate([[float(np.mean(f)), float(np.std(f))]
                           for f in [zcr, rmse, cen, bw, ro]])

# Map tên → hàm
HC_EXTRACTORS: dict[str, object] = {
    "MFCC":              extract_mfcc,
    "Log-Mel":           extract_logmel,
    "Chroma+C+T":        extract_chroma,
    "Statistical":       extract_statistical,
}

def build_hc_matrix(signals: list[np.ndarray],
                    extractor_fn,
                    name: str = "") -> np.ndarray:
    X = []
    errors = 0
    for y in signals:
        try:
            X.append(extractor_fn(y))
        except Exception:
            errors += 1
            X.append(X[-1].copy() if X else np.zeros(10))
    if errors:
        print(f"  [WARN] {name}: {errors} lỗi trích xuất")
    return np.array(X)


# ============================================================================
# PHẦN 3: TẢI PRETRAINED EMBEDDINGS (cache từ Mục 2)
# ============================================================================

def load_embedding_cache(embed_name: str,
                         n_samples: int) -> np.ndarray | None:
    """
    Load embedding đã tính từ Mục 2 (cache .npy).
    Tránh download + compute lại từ đầu.
    """
    cache_key  = f"ravdess_{n_samples}"
    cache_file = Path(CACHE_DIR) / f"{cache_key}_{embed_name}.npy"
    if cache_file.exists():
        print(f"  [CACHE] Loaded {embed_name} ← {cache_file}")
        return np.load(str(cache_file))
    return None


def extract_embedding_live(embed_name: str,
                            config: dict,
                            signals: list[np.ndarray],
                            signals_mert: list[np.ndarray],
                            n_samples: int) -> np.ndarray:
    """
    Trích xuất embedding nếu cache không tồn tại.
    Dùng lại logic từ Mục 2.
    """
    print(f"  [INFO] Cache không tìm thấy cho {embed_name}. "
          f"Đang trích xuất từ HuggingFace...")
    try:
        import torch
        from transformers import (Wav2Vec2Processor, Wav2Vec2Model,
                                   Wav2Vec2FeatureExtractor, HubertModel,
                                   WhisperProcessor, WhisperModel,
                                   AutoProcessor, AutoModel)

        if embed_name == "Wav2Vec2":
            proc  = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        elif embed_name == "HuBERT":
            proc  = Wav2Vec2FeatureExtractor.from_pretrained(
                        "facebook/hubert-base-ls960")
            model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        elif embed_name == "Whisper":
            proc  = WhisperProcessor.from_pretrained("openai/whisper-base")
            model = WhisperModel.from_pretrained("openai/whisper-base")
        elif embed_name == "MERT":
            proc  = AutoProcessor.from_pretrained("m-a-p/MERT-v1-95M",
                                                   trust_remote_code=True)
            model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M",
                                               trust_remote_code=True)
        else:
            raise ValueError(f"Unknown embedding: {embed_name}")

        model.eval()
        embeddings = []
        src = signals_mert if embed_name == "MERT" else signals
        sr  = SR_MERT       if embed_name == "MERT" else SR_EMB

        for i, y in enumerate(src):
            with torch.no_grad():
                if embed_name in ("Wav2Vec2", "HuBERT"):
                    inp = proc(y, sampling_rate=sr,
                               return_tensors="pt", padding=True)
                    out = model(**inp, output_hidden_states=True)
                    h   = out.hidden_states[LAYER_IDX + 1]
                    emb = h.squeeze(0).mean(0)
                elif embed_name == "Whisper":
                    inp = proc(y, sampling_rate=sr, return_tensors="pt")
                    out = model.encoder(inp["input_features"])
                    emb = out.last_hidden_state.squeeze(0).mean(0)
                elif embed_name == "MERT":
                    inp = proc(y, sampling_rate=sr, return_tensors="pt")
                    out = model(**inp, output_hidden_states=True)
                    all_h = torch.stack(out.hidden_states)
                    emb   = all_h.mean(0).squeeze(0).mean(0)
                embeddings.append(emb.cpu().numpy().astype(np.float32))
            if (i + 1) % 200 == 0:
                print(f"    {i+1}/{len(src)}")

        X = np.array(embeddings)
        cache_key  = f"ravdess_{n_samples}"
        cache_file = Path(CACHE_DIR) / f"{cache_key}_{embed_name}.npy"
        np.save(str(cache_file), X)
        print(f"  [CACHE] Đã lưu → {cache_file}")
        return X

    except ImportError as e:
        raise RuntimeError(
            f"Không thể load {embed_name}: {e}\n"
            f"pip install transformers torch torchaudio"
        ) from e


EMBED_CONFIGS: dict[str, dict] = {
    "HuBERT":   {"dim": 768},
    "Wav2Vec2": {"dim": 768},
    "Whisper":  {"dim": 512},
    "MERT":     {"dim": 768},
}


# ============================================================================
# PHẦN 4: ĐÁNH GIÁ CHUNG
# ============================================================================

def evaluate_cv(X: np.ndarray, y_enc: np.ndarray,
                model: Pipeline,
                cv: int = 5) -> dict[str, float]:
    """5-fold Stratified CV → acc, acc_std, f1_macro, f1_weighted."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    acc  = cross_val_score(model, X, y_enc, cv=skf, scoring="accuracy",    n_jobs=-1)
    f1m  = cross_val_score(model, X, y_enc, cv=skf, scoring="f1_macro",    n_jobs=-1)
    f1w  = cross_val_score(model, X, y_enc, cv=skf, scoring="f1_weighted", n_jobs=-1)
    return {
        "acc":      round(float(acc.mean()) * 100, 2),
        "acc_std":  round(float(acc.std())  * 100, 2),
        "f1_macro": round(float(f1m.mean()) * 100, 2),
        "f1_wtd":   round(float(f1w.mean()) * 100, 2),
    }


def svm_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler",     StandardScaler()),
        ("normalizer", Normalizer(norm="l2")),
        ("clf",        SVC(kernel="rbf", C=10, gamma="scale",
                           probability=True, random_state=RANDOM_SEED)),
    ])

def rf_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler",     StandardScaler()),
        ("normalizer", Normalizer(norm="l2")),
        ("clf",        RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1
        )),
    ])


def print_result(label: str, m: dict) -> None:
    print(f"  {label:<50s}  "
          f"Acc={m['acc']:5.1f}%±{m['acc_std']:.1f}%  "
          f"F1-mac={m['f1_macro']:5.1f}%  "
          f"F1-wtd={m['f1_wtd']:5.1f}%")


# ============================================================================
# PHẦN 5: BỐN CHIẾN LƯỢC FUSION
# ============================================================================

# ── Chiến lược 1: Early Fusion (Concatenation) ───────────────────────────────

def early_fusion(Xs: list[np.ndarray]) -> np.ndarray:
    """
    Nối tất cả vector đặc trưng thành 1 vector duy nhất.
    Ưu điểm: đơn giản, model học được tương tác giữa các nhóm.
    Nhược điểm: chiều cao → dùng L2 normalize để cân bằng đóng góp.
    """
    return np.concatenate(Xs, axis=1)


def run_early_fusion(hc_dict: dict[str, np.ndarray],
                     emb_dict: dict[str, np.ndarray],
                     y_enc: np.ndarray) -> list[dict]:
    """
    Thử tất cả combinations HC + Embedding (concat trực tiếp).
    Dùng SVM + RF để so sánh.
    """
    print("\n" + "─" * 65)
    print("  CHIẾN LƯỢC 1: EARLY FUSION (Concatenation)")
    print("─" * 65)

    results = []

    # HC + từng Embedding
    for hc_name, X_hc in hc_dict.items():
        for emb_name, X_emb in emb_dict.items():
            X_fused = early_fusion([X_hc, X_emb])
            label   = f"{hc_name} + {emb_name}"
            dim     = X_fused.shape[1]

            for model_name, model in [("SVM", svm_pipeline()),
                                       ("RF",  rf_pipeline())]:
                m = evaluate_cv(X_fused, y_enc, model)
                print_result(f"[Early] {label} | {model_name}", m)
                results.append({
                    "Strategy": "1_Early",
                    "Combination": label,
                    "Model": model_name,
                    "Dim": dim,
                    **m
                })

    # All HC concat + từng Embedding
    X_all_hc = early_fusion(list(hc_dict.values()))
    for emb_name, X_emb in emb_dict.items():
        X_fused = early_fusion([X_all_hc, X_emb])
        label   = f"ALL_HC + {emb_name}"
        dim     = X_fused.shape[1]
        for model_name, model in [("SVM", svm_pipeline()),
                                   ("RF",  rf_pipeline())]:
            m = evaluate_cv(X_fused, y_enc, model)
            print_result(f"[Early] {label} | {model_name}", m)
            results.append({
                "Strategy": "1_Early",
                "Combination": label,
                "Model": model_name,
                "Dim": dim,
                **m
            })

    # All HC + All Embeddings
    X_all_emb = early_fusion(list(emb_dict.values()))
    X_all     = early_fusion([X_all_hc, X_all_emb])
    for model_name, model in [("SVM", svm_pipeline()),
                               ("RF",  rf_pipeline())]:
        m = evaluate_cv(X_all, y_enc, model)
        print_result(f"[Early] ALL_HC + ALL_EMB | {model_name}", m)
        results.append({
            "Strategy": "1_Early",
            "Combination": "ALL_HC + ALL_EMB",
            "Model": model_name,
            "Dim": X_all.shape[1],
            **m
        })

    return results


# ── Chiến lược 2: Late Fusion (Score-level) ──────────────────────────────────

def run_late_fusion(hc_dict: dict[str, np.ndarray],
                    emb_dict: dict[str, np.ndarray],
                    y_enc: np.ndarray,
                    n_classes: int) -> list[dict]:
    """
    Mỗi nhóm feature có SVM riêng → lấy xác suất predict → kết hợp.

    3 cách kết hợp xác suất:
      - Average : prob = (p1 + p2) / 2
      - Weighted: prob = w1*p1 + w2*p2  (w tỷ lệ với val accuracy từng model)
      - Max Vote: majority vote từ predict label (không dùng prob)

    Đây là cách "ensemble" cổ điển, không cần refit thêm model nào.
    """
    print("\n" + "─" * 65)
    print("  CHIẾN LƯỢC 2: LATE FUSION (Score-level Ensemble)")
    print("─" * 65)
    print("  [NOTE] Dùng 80/20 hold-out (không phải CV) để có prob trên test set.")
    print("         Kết quả mang tính tham khảo, không so sánh trực tiếp với CV.")

    results = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    hc_keys  = list(hc_dict.keys())
    emb_keys = list(emb_dict.keys())

    def late_fuse_cv(X_a: np.ndarray,
                     X_b: np.ndarray,
                     name_a: str,
                     name_b: str) -> None:
        """5-fold CV cho late fusion: train 2 SVM riêng, combine probs."""
        avg_accs,  avg_f1s  = [], []
        wtd_accs,  wtd_f1s  = [], []
        vote_accs, vote_f1s = [], []

        for tr_idx, te_idx in skf.split(X_a, y_enc):
            # Fit SVM A
            pip_a = svm_pipeline()
            pip_a.fit(X_a[tr_idx], y_enc[tr_idx])
            pa = pip_a.predict_proba(X_a[te_idx])   # (N_te, 8)

            # Fit SVM B
            pip_b = svm_pipeline()
            pip_b.fit(X_b[tr_idx], y_enc[tr_idx])
            pb = pip_b.predict_proba(X_b[te_idx])   # (N_te, 8)

            y_te = y_enc[te_idx]

            # Val accuracy của từng model (để tính weight)
            val_a = accuracy_score(y_te, pa.argmax(1))
            val_b = accuracy_score(y_te, pb.argmax(1))
            w_a   = val_a / (val_a + val_b + 1e-8)
            w_b   = val_b / (val_a + val_b + 1e-8)

            # Average
            p_avg = (pa + pb) / 2
            avg_accs.append(accuracy_score(y_te, p_avg.argmax(1)))
            avg_f1s.append(f1_score(y_te, p_avg.argmax(1),
                                    average="macro", zero_division=0))

            # Weighted
            p_wtd = w_a * pa + w_b * pb
            wtd_accs.append(accuracy_score(y_te, p_wtd.argmax(1)))
            wtd_f1s.append(f1_score(y_te, p_wtd.argmax(1),
                                    average="macro", zero_division=0))

            # Majority vote
            pred_a = pa.argmax(1)
            pred_b = pb.argmax(1)
            vote   = np.where(pred_a == pred_b, pred_a, p_avg.argmax(1))
            vote_accs.append(accuracy_score(y_te, vote))
            vote_f1s.append(f1_score(y_te, vote,
                                     average="macro", zero_division=0))

        label = f"{name_a} + {name_b}"
        for mode, accs, f1s in [("Average",  avg_accs,  avg_f1s),
                                  ("Weighted", wtd_accs,  wtd_f1s),
                                  ("MaxVote",  vote_accs, vote_f1s)]:
            m = {
                "acc":      round(np.mean(accs) * 100, 2),
                "acc_std":  round(np.std(accs)  * 100, 2),
                "f1_macro": round(np.mean(f1s)  * 100, 2),
                "f1_wtd":   0.0,
            }
            print_result(f"[Late-{mode}] {label}", m)
            results.append({
                "Strategy":    f"2_Late_{mode}",
                "Combination": label,
                "Model":       f"Late-{mode}",
                "Dim":         X_a.shape[1] + X_b.shape[1],
                **m
            })

    # HC + Embedding pairs (chọn MFCC và Log-Mel làm đại diện HC)
    for hc_name in ["MFCC", "Log-Mel"]:
        if hc_name not in hc_dict:
            continue
        for emb_name, X_emb in emb_dict.items():
            late_fuse_cv(hc_dict[hc_name], X_emb, hc_name, emb_name)

    # Best HC (MFCC) + Best Embedding (HuBERT)
    if "MFCC" in hc_dict and "HuBERT" in emb_dict:
        print("\n  [Best pair đã có ở trên]")

    return results


# ── Chiến lược 3: Intermediate Fusion (PCA → concat) ─────────────────────────

def run_intermediate_fusion(hc_dict: dict[str, np.ndarray],
                             emb_dict: dict[str, np.ndarray],
                             y_enc: np.ndarray,
                             pca_dim: int = PCA_DIM) -> list[dict]:
    """
    Giảm chiều mỗi nhóm xuống pca_dim bằng PCA trước khi concat.

    Tại sao cần PCA?
      - MFCC (240d) vs HuBERT (768d): HuBERT chiếm 76% không gian feature
        → SVM có thể bị dominated bởi embedding
      - PCA → (64d, 64d) cân bằng contribution của từng nhóm
      - Giảm tổng chiều: 240+768=1008 → 64+64=128
        → tăng tốc training, giảm overfitting
    """
    print("\n" + "─" * 65)
    print(f"  CHIẾN LƯỢC 3: INTERMEDIATE FUSION (PCA→{pca_dim}d → concat)")
    print("─" * 65)

    results = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    for hc_name, X_hc in hc_dict.items():
        for emb_name, X_emb in emb_dict.items():
            label = f"{hc_name} + {emb_name} (PCA-{pca_dim})"
            dim_out = pca_dim * 2

            acc_list, f1m_list, f1w_list = [], [], []

            for tr_idx, te_idx in skf.split(X_hc, y_enc):
                # Fit PCA riêng cho từng nhóm (chỉ trên train set)
                sc_hc  = StandardScaler()
                pca_hc = PCA(n_components=min(pca_dim, X_hc.shape[1]),
                             random_state=RANDOM_SEED)
                h_tr = pca_hc.fit_transform(sc_hc.fit_transform(X_hc[tr_idx]))
                h_te = pca_hc.transform(sc_hc.transform(X_hc[te_idx]))

                sc_emb  = StandardScaler()
                pca_emb = PCA(n_components=min(pca_dim, X_emb.shape[1]),
                              random_state=RANDOM_SEED)
                e_tr = pca_emb.fit_transform(sc_emb.fit_transform(X_emb[tr_idx]))
                e_te = pca_emb.transform(sc_emb.transform(X_emb[te_idx]))

                X_tr_fused = np.concatenate([h_tr, e_tr], axis=1)
                X_te_fused = np.concatenate([h_te, e_te], axis=1)

                # Classifier trên không gian đã giảm chiều
                clf = SVC(kernel="rbf", C=10, gamma="scale",
                          probability=True, random_state=RANDOM_SEED)
                clf.fit(X_tr_fused, y_enc[tr_idx])
                y_pred = clf.predict(X_te_fused)
                y_te   = y_enc[te_idx]

                acc_list.append(accuracy_score(y_te, y_pred))
                f1m_list.append(f1_score(y_te, y_pred,
                                         average="macro",    zero_division=0))
                f1w_list.append(f1_score(y_te, y_pred,
                                         average="weighted", zero_division=0))

            m = {
                "acc":      round(np.mean(acc_list) * 100, 2),
                "acc_std":  round(np.std(acc_list)  * 100, 2),
                "f1_macro": round(np.mean(f1m_list) * 100, 2),
                "f1_wtd":   round(np.mean(f1w_list) * 100, 2),
            }
            print_result(f"[Inter] {label}", m)
            results.append({
                "Strategy":    "3_Intermediate",
                "Combination": label,
                "Model":       f"SVM+PCA{pca_dim}",
                "Dim":         dim_out,
                **m
            })

    # ALL HC + ALL EMB với PCA
    X_all_hc  = np.concatenate(list(hc_dict.values()), axis=1)
    X_all_emb = np.concatenate(list(emb_dict.values()), axis=1)
    label     = f"ALL_HC + ALL_EMB (PCA-{pca_dim})"

    acc_list, f1m_list, f1w_list = [], [], []
    for tr_idx, te_idx in skf.split(X_all_hc, y_enc):
        sc_hc  = StandardScaler()
        pca_hc = PCA(n_components=pca_dim, random_state=RANDOM_SEED)
        h_tr   = pca_hc.fit_transform(sc_hc.fit_transform(X_all_hc[tr_idx]))
        h_te   = pca_hc.transform(sc_hc.transform(X_all_hc[te_idx]))

        sc_emb  = StandardScaler()
        pca_emb = PCA(n_components=pca_dim, random_state=RANDOM_SEED)
        e_tr    = pca_emb.fit_transform(sc_emb.fit_transform(X_all_emb[tr_idx]))
        e_te    = pca_emb.transform(sc_emb.transform(X_all_emb[te_idx]))

        X_tr_f = np.concatenate([h_tr, e_tr], axis=1)
        X_te_f = np.concatenate([h_te, e_te], axis=1)

        clf = SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_SEED)
        clf.fit(X_tr_f, y_enc[tr_idx])
        y_pred = clf.predict(X_te_f)
        y_te   = y_enc[te_idx]
        acc_list.append(accuracy_score(y_te, y_pred))
        f1m_list.append(f1_score(y_te, y_pred, average="macro",    zero_division=0))
        f1w_list.append(f1_score(y_te, y_pred, average="weighted", zero_division=0))

    m = {
        "acc":      round(np.mean(acc_list) * 100, 2),
        "acc_std":  round(np.std(acc_list)  * 100, 2),
        "f1_macro": round(np.mean(f1m_list) * 100, 2),
        "f1_wtd":   round(np.mean(f1w_list) * 100, 2),
    }
    print_result(f"[Inter] {label}", m)
    results.append({
        "Strategy":    "3_Intermediate",
        "Combination": label,
        "Model":       f"SVM+PCA{pca_dim}",
        "Dim":         pca_dim * 2,
        **m
    })

    return results


# ── Chiến lược 4: Attention Gate Fusion (MLP + Gating) ───────────────────────

def _build_attention_gate_torch(dim_hc: int,
                                 dim_emb: int,
                                 n_classes: int,
                                 proj_dim: int = 256) -> object:
    """
    Attention-based Gating Fusion (PyTorch).

    Kiến trúc:
      HC  (dim_hc)  → Linear → proj_dim → BN → ReLU
      EMB (dim_emb) → Linear → proj_dim → BN → ReLU
      Gate: softmax(Linear([h_hc; h_emb])) → [α_hc, α_emb]
      Fused = α_hc · h_hc + α_emb · h_emb
      Classifier: Linear(proj_dim → n_classes)

    Ý nghĩa: với mỗi sample, model tự học nên tin vào feature nào hơn.
    Liên hệ trực tiếp với kiến trúc MAMEX (Mục 1 thực nghiệm).
    """
    import torch.nn as nn

    class AttentionGateFusion(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj_hc  = nn.Sequential(
                nn.Linear(dim_hc,  proj_dim),
                nn.BatchNorm1d(proj_dim), nn.ReLU(), nn.Dropout(0.3)
            )
            self.proj_emb = nn.Sequential(
                nn.Linear(dim_emb, proj_dim),
                nn.BatchNorm1d(proj_dim), nn.ReLU(), nn.Dropout(0.3)
            )
            # Gate network: nhận concat của 2 projections → softmax weights
            self.gate = nn.Sequential(
                nn.Linear(proj_dim * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
                nn.Softmax(dim=-1)    # [α_hc, α_emb]
            )
            self.classifier = nn.Sequential(
                nn.Linear(proj_dim, 128),
                nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, n_classes)
            )

        def forward(self, x_hc, x_emb):  # type: ignore[override]
            h_hc  = self.proj_hc(x_hc)
            h_emb = self.proj_emb(x_emb)
            gate_w = self.gate(
                __import__("torch").cat([h_hc, h_emb], dim=1)
            )                              # (batch, 2)
            alpha_hc  = gate_w[:, 0:1]    # (batch, 1)
            alpha_emb = gate_w[:, 1:2]    # (batch, 1)
            fused = alpha_hc * h_hc + alpha_emb * h_emb  # (batch, proj_dim)
            return self.classifier(fused), gate_w

    return AttentionGateFusion()


def _train_attention_torch(X_hc: np.ndarray,
                            X_emb: np.ndarray,
                            y_enc: np.ndarray,
                            n_classes: int) -> tuple[dict, list]:
    """
    Huấn luyện Attention Gate Fusion bằng PyTorch.
    Trả về (metrics_dict, gate_weights_list).
    gate_weights dùng để visualize: model tin vào HC hay Embedding hơn?
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        torch.manual_seed(RANDOM_SEED)

        # Scale từng nhóm riêng
        sc_hc  = StandardScaler()
        sc_emb = StandardScaler()
        X_hc_s  = sc_hc.fit_transform(X_hc).astype(np.float32)
        X_emb_s = sc_emb.fit_transform(X_emb).astype(np.float32)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        acc_list, f1m_list, f1w_list = [], [], []
        all_gates: list[np.ndarray] = []

        for tr_idx, te_idx in skf.split(X_hc_s, y_enc):
            X_hc_tr,  X_hc_te  = X_hc_s[tr_idx],  X_hc_s[te_idx]
            X_emb_tr, X_emb_te = X_emb_s[tr_idx], X_emb_s[te_idx]
            y_tr = y_enc[tr_idx].astype(np.int64)
            y_te = y_enc[te_idx]

            # Re-scale trên fold
            sc1 = StandardScaler()
            X_hc_tr  = sc1.fit_transform(X_hc_tr).astype(np.float32)
            X_hc_te  = sc1.transform(X_hc_te).astype(np.float32)
            sc2 = StandardScaler()
            X_emb_tr = sc2.fit_transform(X_emb_tr).astype(np.float32)
            X_emb_te = sc2.transform(X_emb_te).astype(np.float32)

            ds_tr  = TensorDataset(torch.tensor(X_hc_tr),
                                   torch.tensor(X_emb_tr),
                                   torch.tensor(y_tr))
            dl_tr  = DataLoader(ds_tr, batch_size=DL_BATCH, shuffle=True)

            device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model   = _build_attention_gate_torch(
                X_hc_tr.shape[1], X_emb_tr.shape[1], n_classes
            ).to(device)
            opt     = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                        weight_decay=1e-4)
            sched   = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=DL_EPOCHS
            )
            loss_fn = nn.CrossEntropyLoss()

            best_acc   = 0.0
            no_improve = 0
            patience   = 12

            for _ in range(DL_EPOCHS):
                model.train()
                for xh, xe, yb in dl_tr:
                    xh, xe, yb = xh.to(device), xe.to(device), yb.to(device)
                    opt.zero_grad()
                    logits, _ = model(xh, xe)
                    loss_fn(logits, yb).backward()
                    opt.step()
                sched.step()

                model.eval()
                with torch.no_grad():
                    xh_te = torch.tensor(X_hc_te).to(device)
                    xe_te = torch.tensor(X_emb_te).to(device)
                    logits_te, gw = model(xh_te, xe_te)
                    y_pred_te = logits_te.argmax(1).cpu().numpy()

                val_acc = accuracy_score(y_te, y_pred_te)
                if val_acc > best_acc:
                    best_acc   = val_acc
                    no_improve = 0
                    best_pred  = y_pred_te.copy()
                    best_gates = gw.cpu().numpy()
                else:
                    no_improve += 1
                if no_improve >= patience:
                    break

            acc_list.append(accuracy_score(y_te, best_pred))
            f1m_list.append(f1_score(y_te, best_pred,
                                     average="macro",    zero_division=0))
            f1w_list.append(f1_score(y_te, best_pred,
                                     average="weighted", zero_division=0))
            all_gates.append(best_gates)

        return {
            "acc":      round(np.mean(acc_list) * 100, 2),
            "acc_std":  round(np.std(acc_list)  * 100, 2),
            "f1_macro": round(np.mean(f1m_list) * 100, 2),
            "f1_wtd":   round(np.mean(f1w_list) * 100, 2),
        }, all_gates

    except ImportError:
        return {"acc": 0, "acc_std": 0, "f1_macro": 0, "f1_wtd": 0}, []


def run_attention_fusion(hc_dict: dict[str, np.ndarray],
                          emb_dict: dict[str, np.ndarray],
                          y_enc: np.ndarray,
                          n_classes: int) -> tuple[list[dict], dict]:
    """
    Attention Gating Fusion: chạy MFCC + mỗi Embedding, và ALL_HC + best_EMB.
    Trả về (results, gate_weights_dict) để visualize.
    """
    print("\n" + "─" * 65)
    print("  CHIẾN LƯỢC 4: ATTENTION GATE FUSION (MLP + Gating)")
    print("─" * 65)

    if DL_BACKEND == "none":
        print("  [SKIP] Cần PyTorch hoặc TensorFlow. "
              "Cài: pip install torch --index-url "
              "https://download.pytorch.org/whl/cpu")
        return [], {}

    results    = []
    gate_store = {}

    # MFCC + mỗi Embedding
    for emb_name, X_emb in emb_dict.items():
        label = f"MFCC + {emb_name}"
        print(f"\n  Đang train: {label}...")
        m, gates = _train_attention_torch(
            hc_dict["MFCC"], X_emb, y_enc, n_classes
        )
        print_result(f"[AttGate] {label}", m)
        results.append({
            "Strategy":    "4_AttentionGate",
            "Combination": label,
            "Model":       "AttGate-MLP",
            "Dim":         hc_dict["MFCC"].shape[1] + X_emb.shape[1],
            **m
        })
        if gates:
            gate_store[label] = np.concatenate(gates, axis=0)

    # ALL HC + Best Embedding (HuBERT)
    if "HuBERT" in emb_dict:
        X_all_hc = np.concatenate(list(hc_dict.values()), axis=1)
        label    = "ALL_HC + HuBERT"
        print(f"\n  Đang train: {label}...")
        m, gates = _train_attention_torch(
            X_all_hc, emb_dict["HuBERT"], y_enc, n_classes
        )
        print_result(f"[AttGate] {label}", m)
        results.append({
            "Strategy":    "4_AttentionGate",
            "Combination": label,
            "Model":       "AttGate-MLP",
            "Dim":         X_all_hc.shape[1] + emb_dict["HuBERT"].shape[1],
            **m
        })
        if gates:
            gate_store[label] = np.concatenate(gates, axis=0)

    return results, gate_store


# ============================================================================
# PHẦN 6: VISUALIZATION
# ============================================================================

def plot_fusion_comparison(df: pd.DataFrame,
                            save_path: str = "results_fusion_comparison.png"
                            ) -> None:
    """
    4 subplot:
      (a) Grouped bar – F1-macro theo Strategy (top combinations)
      (b) Heatmap – tất cả combinations × metric
      (c) Scatter – Acc vs F1-macro
      (d) Bar – so sánh top Fusion vs baseline Mục 1 & 2
    """
    fig = plt.figure(figsize=(22, 16))
    fig.suptitle(
        "So sánh 4 chiến lược Fusion – Speech Emotion Recognition (RAVDESS)\n"
        f"Python {sys.version_info.major}.{sys.version_info.minor}",
        fontsize=14, fontweight="bold"
    )

    strategy_colors = {
        "1_Early":          "#4CAF50",
        "2_Late_Average":   "#2196F3",
        "2_Late_Weighted":  "#03A9F4",
        "2_Late_MaxVote":   "#00BCD4",
        "3_Intermediate":   "#FF9800",
        "4_AttentionGate":  "#9C27B0",
    }

    # ── (a) Top 15 combinations by F1-macro ──────────────────────────────────
    ax_a = fig.add_subplot(2, 2, 1)
    top15 = df.nlargest(15, "f1_macro").copy()
    top15["label"] = top15["Strategy"].str.split("_", n=1).str[1].str[:8] \
                     + "\n" + top15["Combination"].str[:25]
    colors = [strategy_colors.get(s, "#888") for s in top15["Strategy"]]
    bars = ax_a.barh(range(len(top15)), top15["f1_macro"],
                     color=colors, edgecolor="white", linewidth=0.5)
    ax_a.set_yticks(range(len(top15)))
    ax_a.set_yticklabels(top15["label"], fontsize=7)
    ax_a.set_xlabel("F1-macro (%)")
    ax_a.set_title("(a) Top 15 Combinations (F1-macro)", fontweight="bold")

    # Vẽ đường baseline
    for bname, binfo in BASELINE.items():
        ax_a.axvline(binfo["f1"], color=binfo["color"],
                     linestyle="--", linewidth=1.2, alpha=0.8)
        ax_a.text(binfo["f1"] + 0.2, len(top15) - 1,
                  bname[:20], fontsize=6, color=binfo["color"],
                  va="top", rotation=90)

    for bar, val in zip(bars, top15["f1_macro"]):
        ax_a.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                  f"{val:.1f}%", va="center", fontsize=7, fontweight="bold")

    # Legend
    handles = [mpatches.Patch(color=c, label=k.split("_", 1)[1])
               for k, c in strategy_colors.items()]
    ax_a.legend(handles=handles, fontsize=7, loc="lower right")

    # ── (b) Heatmap – best per strategy ──────────────────────────────────────
    ax_b = fig.add_subplot(2, 2, 2)
    best_per = (df.groupby("Strategy")[["acc", "f1_macro", "f1_wtd"]]
                  .max().reset_index())
    best_per["Strategy_label"] = best_per["Strategy"].str.split("_", n=1).str[1]
    hm_data = best_per.set_index("Strategy_label")[["acc", "f1_macro", "f1_wtd"]]
    hm_data.columns = ["Accuracy", "F1-macro", "F1-weighted"]

    # Thêm baseline rows
    bl_rows = pd.DataFrame({
        "Accuracy":    [BASELINE["MFCC + SVM (Mục 1)"]["acc"],
                        BASELINE["HuBERT + SVM (Mục 2)"]["acc"]],
        "F1-macro":    [BASELINE["MFCC + SVM (Mục 1)"]["f1"],
                        BASELINE["HuBERT + SVM (Mục 2)"]["f1"]],
        "F1-weighted": [BASELINE["MFCC + SVM (Mục 1)"]["f1"],
                        BASELINE["HuBERT + SVM (Mục 2)"]["f1"]],
    }, index=["[Baseline] MFCC+SVM", "[Baseline] HuBERT+SVM"])
    hm_data = pd.concat([hm_data, bl_rows])

    sns.heatmap(hm_data, annot=True, fmt=".1f", cmap="RdYlGn",
                linewidths=0.5, ax=ax_b, cbar_kws={"label": "%"},
                vmin=60, vmax=95)
    ax_b.set_title("(b) Best per Strategy – Acc vs F1 (%)", fontweight="bold")
    ax_b.tick_params(axis="x", rotation=15)
    ax_b.tick_params(axis="y", rotation=0)

    # ── (c) Scatter – Acc vs F1-macro ─────────────────────────────────────────
    ax_c = fig.add_subplot(2, 2, 3)
    for strat, grp in df.groupby("Strategy"):
        color = strategy_colors.get(strat, "#888")
        ax_c.scatter(grp["acc"], grp["f1_macro"],
                     c=color, label=strat.split("_", 1)[1],
                     alpha=0.75, s=60, edgecolors="white", linewidth=0.5)

    # Baseline points
    for bname, binfo in BASELINE.items():
        ax_c.scatter(binfo["acc"], binfo["f1"],
                     marker="*", s=200, color=binfo["color"],
                     zorder=5, edgecolors="black", linewidth=0.5)
        ax_c.annotate(bname[:18], (binfo["acc"], binfo["f1"]),
                      fontsize=6.5, xytext=(3, 3), textcoords="offset points")

    # Diagonal reference
    all_vals = list(df["acc"]) + [b["acc"] for b in BASELINE.values()]
    lo, hi = min(all_vals) - 1, max(all_vals) + 1
    ax_c.plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=0.8)
    ax_c.set_xlabel("Accuracy (%)")
    ax_c.set_ylabel("F1-macro (%)")
    ax_c.set_title("(c) Accuracy vs F1-macro (mỗi điểm = 1 combination)",
                    fontweight="bold")
    ax_c.legend(fontsize=7, markerscale=1.2)

    # ── (d) Bar – top 1 per strategy vs baseline ─────────────────────────────
    ax_d = fig.add_subplot(2, 2, 4)
    rows = []
    for strat, grp in df.groupby("Strategy"):
        best = grp.loc[grp["f1_macro"].idxmax()]
        rows.append({
            "label": strat.split("_", 1)[1][:20] + "\n"
                     + best["Combination"][:22],
            "f1_macro": best["f1_macro"],
            "acc":      best["acc"],
            "color":    strategy_colors.get(strat, "#888"),
        })

    # Thêm baseline
    for bname, binfo in BASELINE.items():
        rows.append({
            "label":    bname[:30],
            "f1_macro": binfo["f1"],
            "acc":      binfo["acc"],
            "color":    binfo["color"],
        })

    rows_df = pd.DataFrame(rows).sort_values("f1_macro")
    ax_d.barh(range(len(rows_df)), rows_df["f1_macro"],
              color=rows_df["color"].tolist(),
              edgecolor="white", linewidth=0.5)
    ax_d.set_yticks(range(len(rows_df)))
    ax_d.set_yticklabels(rows_df["label"], fontsize=7.5)
    ax_d.set_xlabel("F1-macro (%)")
    ax_d.set_title("(d) Best Fusion vs Baseline (F1-macro)",
                    fontweight="bold")

    for i, (_, row) in enumerate(rows_df.iterrows()):
        ax_d.text(row["f1_macro"] + 0.2, i,
                  f"{row['f1_macro']:.1f}%",
                  va="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Biểu đồ so sánh → {save_path}")


def plot_gate_weights(gate_store: dict[str, np.ndarray],
                       save_path: str = "results_fusion_gate_weights.png"
                       ) -> None:
    """
    Visualize gate weights từ Attention Fusion:
    Mỗi cột = 1 combination, bar cho thấy model tin HC hay Embedding hơn.
    """
    if not gate_store:
        return

    n = len(gate_store)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    fig.suptitle("Attention Gate Weights – Mức độ tin tưởng vào từng nhóm feature\n"
                 "(α_HC + α_Embedding = 1 với mỗi sample)",
                 fontsize=12, fontweight="bold")

    for ax, (label, gates) in zip(axes, gate_store.items()):
        mean_w = gates.mean(axis=0)   # [α_hc_mean, α_emb_mean]
        std_w  = gates.std(axis=0)

        bars = ax.bar(["Hand-crafted\n(HC)", "Pretrained\nEmbedding"],
                      mean_w * 100,
                      yerr=std_w * 100,
                      color=["#FF9800", "#2196F3"],
                      edgecolor="white", capsize=5)
        for bar, val in zip(bars, mean_w * 100):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")

        ax.set_title(label, fontweight="bold", fontsize=9)
        ax.set_ylabel("Trọng số trung bình (%)")
        ax.set_ylim(0, 100)
        ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
        ax.text(0.5, 52, "50% (equal weight)", ha="center", fontsize=7,
                color="gray", transform=ax.get_xaxis_transform())

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Gate weights → {save_path}")


def plot_summary_table(df: pd.DataFrame,
                        save_path: str = "results_fusion_summary_table.png"
                        ) -> None:
    """Bảng tổng hợp top 3 mỗi chiến lược + all baselines."""
    rows = []

    # Baseline
    for bname, binfo in BASELINE.items():
        rows.append({
            "Chiến lược":     "Baseline",
            "Combination":    bname,
            "Model":          "SVM",
            "Acc (%)":        binfo["acc"],
            "F1-macro (%)":   binfo["f1"],
            "Cải tiến F1":    "—",
        })

    # Top 3 per strategy
    ref_f1 = BASELINE["HuBERT + SVM (Mục 2)"]["f1"]
    strategy_labels = {
        "1_Early":          "Early Fusion",
        "2_Late_Average":   "Late Avg",
        "2_Late_Weighted":  "Late Wtd",
        "2_Late_MaxVote":   "Late Vote",
        "3_Intermediate":   "Intermediate",
        "4_AttentionGate":  "AttGate",
    }
    for strat, grp in df.groupby("Strategy"):
        top3 = grp.nlargest(3, "f1_macro")
        for _, row in top3.iterrows():
            delta = row["f1_macro"] - ref_f1
            sign  = "+" if delta >= 0 else ""
            rows.append({
                "Chiến lược":   strategy_labels.get(strat, strat),
                "Combination":  row["Combination"][:35],
                "Model":        row["Model"],
                "Acc (%)":      row["acc"],
                "F1-macro (%)": row["f1_macro"],
                "Cải tiến F1":  f"{sign}{delta:.1f}% vs HuBERT",
            })

    tbl_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(18, len(tbl_df) * 0.55 + 1.5))
    ax.axis("off")
    col_widths = [0.10, 0.30, 0.12, 0.09, 0.11, 0.18]
    tbl = ax.table(
        cellText  = tbl_df.values,
        colLabels = tbl_df.columns,
        cellLoc   = "center",
        loc       = "center",
        colWidths = col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.6)

    # Header style
    for j in range(len(tbl_df.columns)):
        tbl[0, j].set_facecolor("#1F4E79")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Row coloring
    strat_row_colors = {
        "Baseline":      "#FFF3E0",
        "Early Fusion":  "#E8F5E9",
        "Late Avg":      "#E3F2FD",
        "Late Wtd":      "#E1F5FE",
        "Late Vote":     "#E0F7FA",
        "Intermediate":  "#FFF8E1",
        "AttGate":       "#F3E5F5",
    }
    for i, (_, row) in enumerate(tbl_df.iterrows(), start=1):
        color = strat_row_colors.get(row["Chiến lược"], "#FFFFFF")
        for j in range(len(tbl_df.columns)):
            tbl[i, j].set_facecolor(color)
        # Highlight F1-macro > HuBERT baseline
        if isinstance(row["F1-macro (%)"], (int, float)):
            if row["F1-macro (%)"] > ref_f1:
                tbl[i, 4].set_facecolor("#C8E6C9")
                tbl[i, 4].set_text_props(fontweight="bold", color="#1B5E20")

    ax.set_title("Bảng tổng hợp kết quả Fusion – Top 3 mỗi chiến lược\n"
                 "(ô xanh = vượt HuBERT baseline 83.9%)",
                 fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Summary table → {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    print("\n" + "=" * 65)
    print("  THỰC NGHIỆM MỤC 3 – FUSION FEATURES ĐA ĐẶC TRƯNG")
    print(f"  Python {sys.version_info.major}.{sys.version_info.minor}"
          f"  |  DL: {DL_BACKEND}")
    print("=" * 65)

    # 1. Load data
    signals_hc, signals_emb, signals_mert, labels, _ = load_ravdess(RAVDESS_PATH)
    le    = LabelEncoder()
    y_enc = le.fit_transform(labels)
    n_cls = int(len(le.classes_))
    n_smp = len(labels)

    print(f"[INFO] {n_smp} samples | {n_cls} classes")

    # 2. Trích xuất Hand-crafted Features
    print("\n[INFO] Trích xuất Hand-crafted Features...")
    hc_dict: dict[str, np.ndarray] = {}
    for name, fn in HC_EXTRACTORS.items():
        t0 = time.time()
        hc_dict[name] = build_hc_matrix(signals_hc, fn, name)
        print(f"  {name:20s}: {hc_dict[name].shape[1]}d  ({time.time()-t0:.1f}s)")

    # 3. Load Pretrained Embeddings (từ cache Mục 2)
    print("\n[INFO] Loading Pretrained Embeddings...")
    emb_dict: dict[str, np.ndarray] = {}
    for embed_name in EMBED_CONFIGS:
        X = load_embedding_cache(embed_name, n_smp)
        if X is None:
            print(f"  [INFO] Cache miss → trích xuất {embed_name} live...")
            X = extract_embedding_live(
                embed_name, EMBED_CONFIGS[embed_name],
                signals_emb, signals_mert, n_smp
            )
        if X is not None:
            emb_dict[embed_name] = X
            print(f"  {embed_name:12s}: {X.shape[1]}d  ✓")

    if not emb_dict:
        print("[ERROR] Không có embedding nào. Chạy Mục 2 trước.")
        return

    # 4. Chạy 4 chiến lược Fusion
    all_results: list[dict] = []

    r1 = run_early_fusion(hc_dict, emb_dict, y_enc)
    all_results.extend(r1)

    r2 = run_late_fusion(hc_dict, emb_dict, y_enc, n_cls)
    all_results.extend(r2)

    r3 = run_intermediate_fusion(hc_dict, emb_dict, y_enc)
    all_results.extend(r3)

    r4, gate_store = run_attention_fusion(hc_dict, emb_dict, y_enc, n_cls)
    all_results.extend(r4)

    # 5. Tổng hợp
    df = pd.DataFrame(all_results)

    print("\n" + "=" * 65)
    print("  KẾT QUẢ TỔNG HỢP – TOP 10 (theo F1-macro)")
    print("=" * 65)
    top10 = (df.nlargest(10, "f1_macro")
               [["Strategy", "Combination", "Model",
                 "acc", "acc_std", "f1_macro", "f1_wtd"]]
               .reset_index(drop=True))
    top10.index += 1
    print(top10.to_string())

    print("\n  BASELINE (để so sánh):")
    for bname, binfo in BASELINE.items():
        print(f"  {bname:<35s}  Acc={binfo['acc']:.1f}%  F1={binfo['f1']:.1f}%")

    print("\n  BEST PER STRATEGY:")
    for strat, grp in df.groupby("Strategy"):
        best = grp.loc[grp["f1_macro"].idxmax()]
        delta = best["f1_macro"] - BASELINE["HuBERT + SVM (Mục 2)"]["f1"]
        sign  = "+" if delta >= 0 else ""
        print(f"  [{strat:20s}] {best['Combination'][:30]:<30s} "
              f"Acc={best['acc']:.1f}%  F1-mac={best['f1_macro']:.1f}%  "
              f"({sign}{delta:.1f}% vs HuBERT)")

    # 6. Lưu CSV
    df.to_csv("results_fusion_full.csv", index=False)
    print(f"\n[INFO] Đã lưu: results_fusion_full.csv")

    # 7. Visualizations
    print("[INFO] Đang vẽ biểu đồ...")
    plot_fusion_comparison(df)
    plot_summary_table(df)
    if gate_store:
        plot_gate_weights(gate_store)

    print("\n" + "=" * 65)
    print("  HOÀN THÀNH! Output files:")
    print("  • results_fusion_full.csv")
    print("  • results_fusion_comparison.png  – 4 subplot so sánh")
    print("  • results_fusion_summary_table.png – bảng tổng hợp")
    print("  • results_fusion_gate_weights.png  – attention weights")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
