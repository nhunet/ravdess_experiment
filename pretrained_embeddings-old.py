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


# ============================================================
# PHẦN 2: SO SÁNH EMBEDDINGS VỚI CLASSIFIER
# ============================================================

PRETRAINED_EXTRACTORS = {
    "Wav2Vec2-base":   extract_wav2vec2,
    "HuBERT-base":     extract_hubert,
    "MERT-v1-95M":     extract_mert,
    "OpenL3-512":      extract_openl3,
    "PANNs-CNN14":     extract_panns_cnn14,
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


def compare_embeddings(signals: list, y_enc: np.ndarray) -> pd.DataFrame:
    """
    Chạy toàn bộ extractor × classifier, trả về DataFrame kết quả.
    """
    results = []
    le = LabelEncoder()

    for emb_name, extractor in PRETRAINED_EXTRACTORS.items():
        print(f"\n{'─'*55}")
        print(f"  EMBEDDING: {emb_name}")
        print(f"{'─'*55}")

        t0 = time.time()
        X  = extractor(signals)
        print(f"  Shape: {X.shape}  |  Thời gian: {time.time()-t0:.1f}s")

        if X.sum() == 0:
            print("  [SKIP] Embedding toàn zero – bỏ qua")
            continue

        for clf_name, clf in CLASSIFIERS.items():
            metrics = evaluate_ml(X, y_enc, clf)
            print(
                f"  [{clf_name:15s}]  "
                f"Acc={metrics['accuracy_mean']*100:.1f}%±{metrics['accuracy_std']*100:.1f}%  "
                f"F1={metrics['f1_mean']*100:.1f}%"
            )
            results.append({
                "Embedding":    emb_name,
                "Classifier":   clf_name,
                "Accuracy(%)":  round(metrics["accuracy_mean"] * 100, 2),
                "Std(%)":       round(metrics["accuracy_std"]  * 100, 2),
                "F1_macro(%)":  round(metrics["f1_mean"]        * 100, 2),
                "Dim":          X.shape[1],
            })

    # Thêm baseline MFCC từ bài cũ để so sánh
    print(f"\n{'─'*55}")
    print("  BASELINE: MFCC (từ feature_experiment.py)")
    print(f"{'─'*55}")

    X_mfcc = build_feature_matrix(signals, extract_mfcc, "MFCC")
    for clf_name, clf in CLASSIFIERS.items():
        metrics = evaluate_ml(X_mfcc, y_enc, clf)
        print(f"  [{clf_name:15s}]  "
              f"Acc={metrics['accuracy_mean']*100:.1f}%  F1={metrics['f1_mean']*100:.1f}%")
        results.append({
            "Embedding":   "MFCC (baseline)",
            "Classifier":  clf_name,
            "Accuracy(%)": round(metrics["accuracy_mean"] * 100, 2),
            "Std(%)":      round(metrics["accuracy_std"]  * 100, 2),
            "F1_macro(%)": round(metrics["f1_mean"]        * 100, 2),
            "Dim":         X_mfcc.shape[1],
        })

    return pd.DataFrame(results)


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


# ============================================================
# VISUALIZATION
# ============================================================

def plot_embedding_comparison(df: pd.DataFrame,
                               save_path: str = "results_pretrained_embeddings.png"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "So sánh Pretrained Audio Embeddings – Speech Emotion Recognition\n"
        "(Bài báo 4: Comparative Analysis of Pretrained Audio Representations)",
        fontsize=12, fontweight="bold"
    )

    # Heatmap
    pivot = df.pivot(index="Embedding", columns="Classifier", values="Accuracy(%)")
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, ax=axes[0])
    axes[0].set_title("Accuracy (%) – Embedding × Classifier", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=20)

    # Bar chart so sánh F1
    avg = df.groupby("Embedding")["F1_macro(%)"].mean().sort_values()
    colors = ["#FF6B6B" if "baseline" in x else "#4ECDC4" for x in avg.index]
    bars = axes[1].barh(avg.index, avg.values, color=colors, edgecolor="grey")
    for bar, val in zip(bars, avg.values):
        axes[1].text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}%", va="center", fontsize=9)
    axes[1].set_xlabel("F1-macro trung bình (%)")
    axes[1].set_title("Xếp hạng Embedding (avg F1)", fontweight="bold")
    axes[1].axvline(
        df[df["Embedding"] == "MFCC (baseline)"]["F1_macro(%)"].mean(),
        color="red", linestyle="--", linewidth=1.5, label="MFCC baseline"
    )
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Biểu đồ → {save_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 65)
    print("  BÀI 4: PRETRAINED AUDIO REPRESENTATIONS")
    print("=" * 65)

    signals, labels, _ = load_ravdess(RAVDESS_PATH)
    le    = LabelEncoder()
    y_enc = le.fit_transform(labels)

    # So sánh embeddings
    df = compare_embeddings(signals, y_enc)

    # In bảng tổng hợp
    print("\n" + "=" * 65)
    print("  KẾT QUẢ TỔNG HỢP")
    print("=" * 65)
    pivot = df.pivot_table(
        index="Embedding", columns="Classifier",
        values=["Accuracy(%)", "F1_macro(%)"], aggfunc="mean"
    ).round(1)
    print(pivot.to_string())

    # Lưu kết quả
    df.to_csv("results_pretrained_embeddings.csv", index=False)
    plot_embedding_comparison(df)

    print("\n[INFO] Output:")
    print("  • results_pretrained_embeddings.csv")
    print("  • results_pretrained_embeddings.png")


if __name__ == "__main__":
    main()