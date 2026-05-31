"""
=============================================================================
BÀI 4: Comparative Analysis of Pretrained Audio Representations
Tích hợp vào: Speech Emotion Recognition → Product Recommendation
=============================================================================
"""

import os, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

# Import từ file cũ
from feature_experiment import (
    load_ravdess, extract_mfcc, build_feature_matrix,
    evaluate_ml as _evaluate_ml_tuple, get_ml_models, RAVDESS_PATH, SR, RANDOM_SEED
)
from sklearn.metrics import f1_score as _f1_score
from sklearn.model_selection import StratifiedKFold as _SKF


def evaluate_ml(X: np.ndarray, y_enc: np.ndarray, model, cv: int = 5) -> dict:
    """Wrapper that returns a metrics dict compatible with pretrained_embeddings usage."""
    skf = _SKF(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    acc_scores, f1_scores = [], []
    for tr, te in skf.split(X, y_enc):
        model.fit(X[tr], y_enc[tr])
        y_pred = model.predict(X[te])
        acc_scores.append((y_pred == y_enc[te]).mean())
        f1_scores.append(_f1_score(y_enc[te], y_pred, average="macro", zero_division=0))
    return {
        "accuracy_mean": float(np.mean(acc_scores)),
        "accuracy_std":  float(np.std(acc_scores)),
        "f1_mean":       float(np.mean(f1_scores)),
        "f1_std":        float(np.std(f1_scores)),
    }

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)

# ============================================================
# PHẦN 1: EXTRACT EMBEDDINGS TỪ PRETRAINED MODELS
# ============================================================

# ── 1.1 Wav2Vec2 (Facebook) ──────────────────────────────────
def extract_wav2vec2(signals: list, sr: int = SR,
                     model_name: str = "facebook/wav2vec2-base") -> np.ndarray:
    """
    Wav2Vec2: Self-supervised speech model từ Facebook.
    Output: embedding 768 chiều (base) hoặc 1024 (large).
    Phù hợp nhất cho speech vì được train trên dữ liệu tiếng nói.
    """
    from transformers import Wav2Vec2Processor, Wav2Vec2Model

    print(f"  [Wav2Vec2] Loading {model_name}...")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model     = Wav2Vec2Model.from_pretrained(model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    embeddings = []
    TARGET_SR  = 16000  # Wav2Vec2 yêu cầu 16kHz

    for i, y in enumerate(signals):
        if i % 100 == 0:
            print(f"    Đang xử lý {i}/{len(signals)}...")
        try:
            # Resample nếu cần
            if sr != TARGET_SR:
                y_rs = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
            else:
                y_rs = y

            inputs = processor(
                y_rs, sampling_rate=TARGET_SR,
                return_tensors="pt", padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model(**inputs)

            # Mean pooling qua time dimension → (768,)
            emb = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            embeddings.append(emb)

        except Exception as e:
            print(f"    [WARN] Sample {i}: {e}")
            embeddings.append(np.zeros(768))

    return np.array(embeddings)


# ── 1.2 HuBERT (Facebook) ────────────────────────────────────
def extract_hubert(signals: list, sr: int = SR) -> np.ndarray:
    """
    HuBERT: Hidden-Unit BERT cho speech.
    Tương tự Wav2Vec2 nhưng dùng offline clustering.
    Output: 768 chiều.
    """
    from transformers import HubertModel, Wav2Vec2Processor

    print("  [HuBERT] Loading facebook/hubert-base-ls960...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model     = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    model.eval()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)
    TARGET_SR = 16000

    embeddings = []
    for i, y in enumerate(signals):
        if i % 100 == 0:
            print(f"    Đang xử lý {i}/{len(signals)}...")
        try:
            y_rs = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR) \
                   if sr != TARGET_SR else y
            inputs = processor(y_rs, sampling_rate=TARGET_SR,
                               return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
            emb = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            embeddings.append(emb)
        except Exception as e:
            embeddings.append(np.zeros(768))

    return np.array(embeddings)


# ── 1.3 MERT (Music Foundation Model) ───────────────────────
def extract_mert(signals: list, sr: int = SR) -> np.ndarray:
    """
    MERT: Music-specific pretrained model (m-a-p/MERT-v1-95M).
    Tương tự Wav2Vec2 nhưng được pretrain trên nhạc.
    Bài báo gốc dùng model này như một trong các baseline chính.
    Output: 768 chiều.
    """
    from transformers import AutoModel, AutoProcessor

    print("  [MERT] Loading m-a-p/MERT-v1-95M...")
    try:
        processor = AutoProcessor.from_pretrained(
            "m-a-p/MERT-v1-95M", trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-95M", trust_remote_code=True
        )
        model.eval()
    except Exception as e:
        print(f"  [WARN] Không load được MERT: {e}")
        print("  Bỏ qua MERT, dùng placeholder zeros.")
        return np.zeros((len(signals), 768))

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)
    TARGET_SR = 24000  # MERT dùng 24kHz

    embeddings = []
    for i, y in enumerate(signals):
        if i % 100 == 0:
            print(f"    Đang xử lý {i}/{len(signals)}...")
        try:
            y_rs = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR) \
                   if sr != TARGET_SR else y
            inputs = processor(y_rs, sampling_rate=TARGET_SR,
                               return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            # Lấy trung bình tất cả hidden states (theo bài báo)
            all_layers = torch.stack(out.hidden_states)  # (L, B, T, D)
            emb = all_layers.mean(dim=0).mean(dim=1).squeeze().cpu().numpy()
            embeddings.append(emb)
        except Exception as e:
            embeddings.append(np.zeros(768))

    return np.array(embeddings)


# ── 1.4 OpenL3 ───────────────────────────────────────────────
def extract_openl3(signals: list, sr: int = SR) -> np.ndarray:
    """
    OpenL3: Look, Listen and Learn embedding.
    Train trên audio-visual correspondence.
    Output: 512 chiều (music content type).
    """
    try:
        import openl3
        print("  [OpenL3] Extracting embeddings...")

        embeddings = []
        for i, y in enumerate(signals):
            if i % 100 == 0:
                print(f"    Đang xử lý {i}/{len(signals)}...")
            try:
                emb, _ = openl3.get_audio_embedding(
                    y, sr,
                    content_type="music",
                    embedding_size=512,
                    hop_size=0.1
                )
                # Mean pooling qua time → (512,)
                embeddings.append(emb.mean(axis=0))
            except Exception as e:
                embeddings.append(np.zeros(512))

        return np.array(embeddings)

    except ImportError:
        print("  [WARN] openl3 chưa cài. pip install openl3")
        return np.zeros((len(signals), 512))


# ── 1.5 Panns (CNN14) ────────────────────────────────────────
def extract_panns_cnn14(signals: list, sr: int = SR) -> np.ndarray:
    """
    PANNs CNN14: Pretrained Audio Neural Networks.
    Train trên AudioSet (527 classes, 2M clips).
    Output: 2048 chiều embedding từ penultimate layer.
    Download: https://zenodo.org/record/3987831
    """
    try:
        import panns_inference
        from panns_inference import AudioTagging

        print("  [PANNs CNN14] Loading model...")
        at = AudioTagging(checkpoint_path=None, device="cpu")

        embeddings = []
        TARGET_SR  = 32000

        for i, y in enumerate(signals):
            if i % 100 == 0:
                print(f"    Đang xử lý {i}/{len(signals)}...")
            try:
                y_rs = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR) \
                       if sr != TARGET_SR else y
                y_rs = y_rs[np.newaxis, :]  # (1, samples)
                _, emb = at.inference(y_rs)  # emb: (1, 2048)
                embeddings.append(emb.squeeze())
            except Exception:
                embeddings.append(np.zeros(2048))

        return np.array(embeddings)

    except ImportError:
        print("  [WARN] panns_inference chưa cài. pip install panns-inference")
        return np.zeros((len(signals), 2048))


# ── 1.6 Whisper (OpenAI) ─────────────────────────────────────
def extract_whisper(signals: list, sr: int = SR,
                    model_size: str = "base") -> np.ndarray:
    """
    Whisper (OpenAI) – encoder embedding.
    Được train trên 680k giờ speech đa ngôn ngữ (supervised).
    Khác với Wav2Vec2/HuBERT (self-supervised), Whisper dùng
    weak supervision từ transcript → rất mạnh trên speech tasks.

    Các model_size khả dụng (tăng dần theo độ chính xác & RAM):
      tiny  (39M params, ~1GB RAM)  → nhanh nhất
      base  (74M params, ~1GB RAM)  → khuyến nghị cho thực nghiệm
      small (244M params, ~2GB RAM) → tốt hơn, vẫn chạy được CPU
      medium / large → cần GPU

    Output: 512 chiều (tiny/base) | 768 chiều (small) | 1024 (medium/large)
    Cài đặt: pip install openai-whisper
    """
    try:
        import whisper
    except ImportError:
        print("  [WARN] openai-whisper chưa cài.")
        print("         Chạy: pip install openai-whisper")
        return np.zeros((len(signals), 512))

    print(f"  [Whisper-{model_size}] Loading model...")
    try:
        model = whisper.load_model(model_size)
    except Exception as e:
        print(f"  [WARN] Không load được Whisper: {e}")
        return np.zeros((len(signals), 512))

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    TARGET_SR  = 16000   # Whisper yêu cầu 16kHz
    # Kích thước embedding tuỳ theo model_size
    EMB_DIMS   = {"tiny": 384, "base": 512,
                  "small": 768, "medium": 1024, "large": 1280}
    emb_dim    = EMB_DIMS.get(model_size, 512)

    embeddings = []
    for i, y in enumerate(signals):
        if i % 100 == 0:
            print(f"    Đang xử lý {i}/{len(signals)}...")
        try:
            # 1. Resample về 16kHz
            y_rs = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR) \
                   if sr != TARGET_SR else y
            y_rs = y_rs.astype(np.float32)

            # 2. Pad / trim về đúng 30s (chuẩn của Whisper)
            audio = whisper.pad_or_trim(y_rs)

            # 3. Log-Mel spectrogram (80 mel bins, cố định của Whisper)
            mel = whisper.log_mel_spectrogram(audio).to(device)  # (80, 3000)

            # 4. Chạy encoder → lấy hidden states
            with torch.no_grad():
                enc_out = model.encoder(mel.unsqueeze(0))  # (1, T', emb_dim)

            # 5. Mean pooling qua time axis → (emb_dim,)
            emb = enc_out.mean(dim=1).squeeze().cpu().numpy()
            embeddings.append(emb)

        except Exception as e:
            print(f"    [WARN] Sample {i}: {e}")
            embeddings.append(np.zeros(emb_dim))

    arr = np.array(embeddings)
    print(f"  [Whisper-{model_size}] Done – shape: {arr.shape}")
    return arr


# ============================================================
# PHẦN 2: SO SÁNH EMBEDDINGS VỚI CLASSIFIER
# ============================================================

PRETRAINED_EXTRACTORS = {
    "Wav2Vec2-base":      extract_wav2vec2,
    "HuBERT-base":        extract_hubert,
    "MERT-v1-95M":        extract_mert,
    "Whisper-base":       extract_whisper,          # ← MỚI
    "OpenL3-512":         extract_openl3,
    "PANNs-CNN14":        extract_panns_cnn14,
}

# Classifiers đơn giản phù hợp với embedding lớn
CLASSIFIERS = {
    "SVM (RBF)": Pipeline([
        ("sc",  StandardScaler()),
        ("clf", SVC(kernel="rbf", C=10, gamma="scale",
                    random_state=RANDOM_SEED)),
    ]),
    "Random Forest": Pipeline([
        ("sc",  StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200,
                                       random_state=RANDOM_SEED, n_jobs=-1)),
    ]),
}


def compare_embeddings(
    signals: list, y_enc: np.ndarray
) -> tuple[pd.DataFrame, dict]:
    """
    Chạy toàn bộ extractor × classifier.
    Trả về (DataFrame kết quả, embedding_cache).
    embedding_cache dùng lại cho fusion experiments (tránh extract 2 lần).
    """
    results:         list[dict]        = []
    embedding_cache: dict[str, np.ndarray] = {}

    # ── Pretrained embeddings ────────────────────────────────────
    for emb_name, extractor in PRETRAINED_EXTRACTORS.items():
        print(f"\n{'─'*55}")
        print(f"  EMBEDDING: {emb_name}")
        print(f"{'─'*55}")

        t0 = time.time()
        X  = extractor(signals)
        elapsed = time.time() - t0
        print(f"  Shape: {X.shape}  |  Thời gian: {elapsed:.1f}s")

        # Bỏ qua nếu extractor trả về toàn zero (chưa cài thư viện)
        if np.all(X == 0):
            print("  [SKIP] Embedding toàn zero – bỏ qua")
            continue

        embedding_cache[emb_name] = X   # ← lưu cache

        for clf_name, clf in CLASSIFIERS.items():
            metrics = evaluate_ml(X, y_enc, clf)
            print(
                f"  [{clf_name:15s}]  "
                f"Acc={metrics['accuracy_mean']*100:.1f}%"
                f"±{metrics['accuracy_std']*100:.1f}%  "
                f"F1={metrics['f1_mean']*100:.1f}%"
            )
            results.append({
                "Embedding":   emb_name,
                "Classifier":  clf_name,
                "Accuracy(%)": round(metrics["accuracy_mean"] * 100, 2),
                "Std(%)":      round(metrics["accuracy_std"]  * 100, 2),
                "F1_macro(%)": round(metrics["f1_mean"]        * 100, 2),
                "Dim":         X.shape[1],
                "Category":    "Pretrained",
            })

    # ── Baseline MFCC ────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("  BASELINE: MFCC (từ feature_experiment.py)")
    print(f"{'─'*55}")

    X_mfcc = build_feature_matrix(signals, extract_mfcc, "MFCC")
    embedding_cache["MFCC (baseline)"] = X_mfcc   # ← lưu cache

    for clf_name, clf in CLASSIFIERS.items():
        metrics = evaluate_ml(X_mfcc, y_enc, clf)
        print(
            f"  [{clf_name:15s}]  "
            f"Acc={metrics['accuracy_mean']*100:.1f}%  "
            f"F1={metrics['f1_mean']*100:.1f}%"
        )
        results.append({
            "Embedding":   "MFCC (baseline)",
            "Classifier":  clf_name,
            "Accuracy(%)": round(metrics["accuracy_mean"] * 100, 2),
            "Std(%)":      round(metrics["accuracy_std"]  * 100, 2),
            "F1_macro(%)": round(metrics["f1_mean"]        * 100, 2),
            "Dim":         X_mfcc.shape[1],
            "Category":    "Baseline",
        })

    return pd.DataFrame(results), embedding_cache


# ============================================================
# PHẦN 3: KẾT HỢP EMBEDDINGS (FUSION)
# ============================================================

def extract_combined_embedding(signals: list,
                                extractors: dict,
                                sr: int = SR) -> np.ndarray:
    """
    Late fusion: concatenate nhiều embeddings.
    Ví dụ: MFCC + Wav2Vec2 → richer representation.
    """
    parts = []
    for name, fn in extractors.items():
        print(f"  Extracting {name}...")
        X = fn(signals)
        # Normalize từng phần trước khi concat
        sc = StandardScaler()
        X  = sc.fit_transform(X)
        parts.append(X)

    return np.concatenate(parts, axis=1)

def run_fusion_experiments(
    y_enc: np.ndarray,
    embedding_cache: dict[str, np.ndarray],
) -> list[dict]:
    """
    Late-fusion: concatenate các embeddings đã normalize,
    sau đó đánh giá bằng SVM (RBF) – 5-fold CV.

    Trả về list[dict] để append vào DataFrame chính.
    Chỉ chạy fusion nếu TẤT CẢ embedding thành phần đều có trong cache.
    """
    # ── Định nghĩa các cấu hình fusion ──────────────────────────
    fusion_configs: dict[str, list[str]] = {
        # Best + 2nd best (từ kết quả đơn lẻ)
        "HuBERT + MERT":              ["HuBERT-base",     "MERT-v1-95M"],
        # Best pretrained + Whisper
        "HuBERT + Whisper":           ["HuBERT-base",     "Whisper-base"],
        # Best + MFCC baseline
        "HuBERT + MFCC":              ["HuBERT-base",     "MFCC (baseline)"],
        # Top 3 pretrained
        "HuBERT + MERT + Whisper":    ["HuBERT-base",     "MERT-v1-95M",
                                       "Whisper-base"],
        # Tất cả speech models
        "All Speech Models":          ["HuBERT-base",     "MERT-v1-95M",
                                       "Wav2Vec2-base",   "Whisper-base"],
        # Best + MFCC (kết hợp pretrained + handcrafted)
        "HuBERT + MERT + MFCC":       ["HuBERT-base",     "MERT-v1-95M",
                                       "MFCC (baseline)"],
    }

    clf = Pipeline([
        ("sc",  StandardScaler()),
        ("clf", SVC(kernel="rbf", C=10, gamma="scale",
                    random_state=RANDOM_SEED)),
    ])

    results: list[dict] = []

    print(f"\n{'─'*55}")
    print("  FUSION EXPERIMENTS (Late Fusion – Concat + SVM)")
    print(f"{'─'*55}")

    for fusion_name, keys in fusion_configs.items():
        # Kiểm tra đủ embedding trong cache
        missing = [k for k in keys if k not in embedding_cache]
        if missing:
            print(f"  [SKIP] '{fusion_name}': thiếu {missing}")
            continue

        # Normalize từng phần rồi concatenate
        parts: list[np.ndarray] = []
        for k in keys:
            X_part = embedding_cache[k].copy().astype(np.float32)
            sc_part = StandardScaler()
            parts.append(sc_part.fit_transform(X_part))
        X_fused = np.concatenate(parts, axis=1)

        t0      = time.time()
        metrics = evaluate_ml(X_fused, y_enc, clf)
        elapsed = time.time() - t0

        print(
            f"  [Fusion: {fusion_name:30s}]  "
            f"Acc={metrics['accuracy_mean']*100:.1f}%"
            f"±{metrics['accuracy_std']*100:.1f}%  "
            f"F1={metrics['f1_mean']*100:.1f}%  "
            f"Dim={X_fused.shape[1]}  ({elapsed:.0f}s)"
        )
        results.append({
            "Embedding":   f"[Fusion] {fusion_name}",
            "Classifier":  "SVM (RBF)",
            "Accuracy(%)": round(metrics["accuracy_mean"] * 100, 2),
            "Std(%)":      round(metrics["accuracy_std"]  * 100, 2),
            "F1_macro(%)": round(metrics["f1_mean"]        * 100, 2),
            "Dim":         X_fused.shape[1],
            "Category":    "Fusion",
        })

    return results



def plot_embedding_comparison(
    df: pd.DataFrame,
    save_path: str = "results_pretrained_embeddings.png",
) -> None:
    """
    Vẽ 3 biểu đồ:
      (A) Heatmap Accuracy × (Embedding, Classifier)  – chỉ single embeddings
      (B) Bar chart F1-macro trung bình – phân màu 3 nhóm
      (C) Bar chart riêng cho Fusion experiments (nếu có)
    """
    COLOR_MAP = {
        "Pretrained": "#4ECDC4",   # xanh ngọc
        "Baseline":   "#FF6B6B",   # đỏ cam
        "Fusion":     "#6C5CE7",   # tím
    }

    if "Category" not in df.columns:
        df = df.copy()
        df["Category"] = df["Embedding"].apply(
            lambda n: "Fusion" if "[Fusion]" in n
                      else ("Baseline" if "baseline" in n else "Pretrained")
        )

    df_single = df[df["Category"] != "Fusion"]
    df_fusion  = df[df["Category"] == "Fusion"]
    has_fusion = len(df_fusion) > 0

    n_cols = 3 if has_fusion else 2
    fig_w  = 26 if has_fusion else 17
    fig_h  = max(7, df_single["Embedding"].nunique() * 0.75 + 2)

    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h))
    fig.suptitle(
        "So sánh Pretrained Audio Embeddings – Speech Emotion Recognition\n"
        "(Bài báo 4: Comparative Analysis of Pretrained Audio Representations)",
        fontsize=13, fontweight="bold",
    )

    # ── (A) Heatmap ──────────────────────────────────────────────
    ax_heat = axes[0]
    pivot = df_single.pivot_table(index="Embedding", columns="Classifier",
                                   values="Accuracy(%)", aggfunc="mean")
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, ax=ax_heat, cbar_kws={"label": "Accuracy (%)"})
    ax_heat.set_title("(A) Accuracy (%) – Embedding × Classifier", fontweight="bold")
    ax_heat.set_xlabel("Classifier")
    ax_heat.set_ylabel("Embedding")
    ax_heat.tick_params(axis="x", rotation=20)
    ax_heat.tick_params(axis="y", rotation=0)

    # ── (B) Bar chart F1 single embeddings ───────────────────────
    ax_bar = axes[1]
    avg_single = (df_single.groupby(["Embedding", "Category"])["F1_macro(%)"]
                  .mean().reset_index().sort_values("F1_macro(%)"))

    bar_colors = [COLOR_MAP.get(c, "#888") for c in avg_single["Category"]]
    bars = ax_bar.barh(avg_single["Embedding"], avg_single["F1_macro(%)"],
                       color=bar_colors, edgecolor="grey", height=0.6)
    for bar, val in zip(bars, avg_single["F1_macro(%)"]):
        ax_bar.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")

    mfcc_f1 = df[df["Embedding"] == "MFCC (baseline)"]["F1_macro(%)"].mean()
    ax_bar.axvline(mfcc_f1, color="#E17055", linestyle="--",
                   linewidth=1.8, label=f"MFCC baseline ({mfcc_f1:.1f}%)")
    ax_bar.set_xlabel("F1-macro trung bình (%)")
    ax_bar.set_title("(B) Xếp hạng Embedding – avg F1", fontweight="bold")
    ax_bar.set_xlim(0, avg_single["F1_macro(%)"].max() * 1.18)

    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=COLOR_MAP[k], label=k)
                  for k in ["Pretrained", "Baseline"]
                  if k in avg_single["Category"].values]
    legend_els.append(plt.Line2D([0], [0], color="#E17055", linestyle="--",
                                  linewidth=1.8,
                                  label=f"MFCC baseline ({mfcc_f1:.1f}%)"))
    ax_bar.legend(handles=legend_els, loc="lower right", fontsize=8)

    # ── (C) Bar chart Fusion ──────────────────────────────────────
    if has_fusion:
        ax_fuse = axes[2]
        avg_fusion = (df_fusion.groupby("Embedding")["F1_macro(%)"]
                      .mean().reset_index().sort_values("F1_macro(%)"))
        avg_fusion["Label"] = avg_fusion["Embedding"].str.replace(
            r"^\[Fusion\] ", "", regex=True)

        bars_f = ax_fuse.barh(avg_fusion["Label"], avg_fusion["F1_macro(%)"],
                               color=COLOR_MAP["Fusion"], edgecolor="grey", height=0.6)
        for bar, val in zip(bars_f, avg_fusion["F1_macro(%)"]):
            ax_fuse.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                         f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")

        best_single_f1   = df_single["F1_macro(%)"].max()
        best_single_name = df_single.loc[df_single["F1_macro(%)"].idxmax(), "Embedding"]
        ax_fuse.axvline(best_single_f1, color="#4ECDC4", linestyle="--",
                        linewidth=1.8,
                        label=f"Best single: {best_single_name} ({best_single_f1:.1f}%)")
        ax_fuse.axvline(mfcc_f1, color="#E17055", linestyle=":",
                        linewidth=1.5, label=f"MFCC baseline ({mfcc_f1:.1f}%)")

        ax_fuse.set_xlabel("F1-macro (%)")
        ax_fuse.set_title("(C) Fusion Experiments – F1-macro", fontweight="bold")
        x_max = max(avg_fusion["F1_macro(%)"].max(), best_single_f1) * 1.18
        ax_fuse.set_xlim(0, x_max)
        ax_fuse.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Biểu đồ → {save_path}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("\n" + "=" * 65)
    print("  BÀI 4: PRETRAINED AUDIO REPRESENTATIONS + FUSION")
    print("=" * 65)

    signals, labels, _ = load_ravdess(RAVDESS_PATH)
    le    = LabelEncoder()
    y_enc = le.fit_transform(labels)

    # ── Bước 1: So sánh từng embedding đơn lẻ ───────────────────
    df, embedding_cache = compare_embeddings(signals, y_enc)

    # ── Bước 2: Fusion experiments ───────────────────────────────
    fusion_rows = run_fusion_experiments(y_enc, embedding_cache)
    if fusion_rows:
        df = pd.concat([df, pd.DataFrame(fusion_rows)], ignore_index=True)

    # ── Bước 3: In bảng tổng hợp ─────────────────────────────────
    print("\n" + "=" * 65)
    print("  KẾT QUẢ TỔNG HỢP – ACCURACY (%)")
    print("=" * 65)
    pivot_acc = df[df["Category"] != "Fusion"].pivot_table(
        index="Embedding", columns="Classifier",
        values="Accuracy(%)", aggfunc="mean"
    ).round(1)
    print(pivot_acc.to_string())

    print("\n  KẾT QUẢ TỔNG HỢP – F1-MACRO (%)")
    pivot_f1 = df[df["Category"] != "Fusion"].pivot_table(
        index="Embedding", columns="Classifier",
        values="F1_macro(%)", aggfunc="mean"
    ).round(1)
    print(pivot_f1.to_string())

    if fusion_rows:
        print("\n  FUSION RESULTS:")
        df_fuse_print = (df[df["Category"] == "Fusion"]
                         [["Embedding", "Accuracy(%)", "Std(%)", "F1_macro(%)", "Dim"]]
                         .sort_values("Accuracy(%)", ascending=False)
                         .reset_index(drop=True))
        df_fuse_print.index += 1
        print(df_fuse_print.to_string())

    # Top 5 tổng
    top5 = (df.nlargest(5, "Accuracy(%)")
              [["Embedding", "Classifier", "Accuracy(%)", "Std(%)", "F1_macro(%)"]]
              .reset_index(drop=True))
    top5.index += 1
    print(f"\n  TOP 5 KẾT HỢP TỐT NHẤT:")
    print(top5.to_string())

    # ── Bước 4: Lưu kết quả ──────────────────────────────────────
    df.to_csv("results_pretrained_embeddings.csv", index=False)
    plot_embedding_comparison(df)

    print("\n" + "=" * 65)
    print("  OUTPUT:")
    print("  • results_pretrained_embeddings.csv")
    print("  • results_pretrained_embeddings.png")
    print("=" * 65)


if __name__ == "__main__":
    main()