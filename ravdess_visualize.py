"""
=============================================================================
RAVDESS DATASET VISUALIZER
Đề tài: Hệ thống gợi ý sản phẩm dựa trên phân tích giọng nói và cảm xúc
Học viên: Nguyễn Tấn Nhu | GVHD: TS. Bùi Thanh Hùng
=============================================================================

Mô tả:
  Script này phân tích và trực quan hóa toàn diện dataset RAVDESS,
  giúp hiểu rõ phân phối dữ liệu TRƯỚC KHI chạy thực nghiệm.

Output (8 file PNG):
  1. ravdess_01_class_distribution.png   – Phân phối cảm xúc (bar + pie)
  2. ravdess_02_actor_analysis.png       – Phân tích theo actor & giới tính
  3. ravdess_03_duration_analysis.png    – Phân phối thời lượng & intensity
  4. ravdess_04_waveform_gallery.png     – Waveform 8 cảm xúc (1 sample/emotion)
  5. ravdess_05_spectrogram_gallery.png  – Log-Mel spectrogram 8 cảm xúc
  6. ravdess_06_mfcc_gallery.png         – MFCC 8 cảm xúc (40 coeff)
  7. ravdess_07_feature_stats.png        – Boxplot thống kê đặc trưng theo cảm xúc
  8. ravdess_08_correlation.png          – Tương quan MFCC trung bình giữa các cảm xúc

Dataset: RAVDESS – https://zenodo.org/record/1188976
  Giải nén vào: ./RAVDESS/
  Naming: 03-01-{emotion}-{intensity}-{statement}-{repetition}-{actor}.wav
  Emotion codes: 01=neutral 02=calm 03=happy 04=sad
                 05=angry  06=fearful 07=disgust 08=surprised
=============================================================================
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Hằng số – giữ đồng nhất với feature_experiment.py ──────────────────────
RAVDESS_PATH = "./RAVDESS"
SR           = 22050
DURATION     = 3.0
N_MFCC       = 40
N_MELS       = 128
N_FFT        = 2048
HOP_LENGTH   = 512

EMOTIONS: dict[int, str] = {
    1: "neutral", 2: "calm",    3: "happy",    4: "sad",
    5: "angry",   6: "fearful", 7: "disgust",  8: "surprised",
}

# Màu sắc nhất quán cho 8 cảm xúc
EMOTION_COLORS: dict[str, str] = {
    "neutral":   "#9B9B9B",
    "calm":      "#5DADE2",
    "happy":     "#F4D03F",
    "sad":       "#85C1E9",
    "angry":     "#E74C3C",
    "fearful":   "#8E44AD",
    "disgust":   "#27AE60",
    "surprised": "#E67E22",
}

EMOTION_EMOJIS: dict[str, str] = {
    "neutral":   "😐", "calm":      "😌", "happy":    "😄", "sad":      "😢",
    "angry":     "😠", "fearful":   "😨", "disgust":  "🤢", "surprised":"😲",
}

sns.set_theme(style="whitegrid", font_scale=1.05)

# ── Thư mục output ───────────────────────────────────────────────────────────
OUT_DIR = Path("data_visualization")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# PHẦN 1: QUÉT VÀ PHÂN TÍCH METADATA
# ============================================================================

def scan_ravdess(data_path: str) -> pd.DataFrame:
    """
    Quét toàn bộ RAVDESS, trả về DataFrame metadata.
    Cột: path, emotion, emotion_code, intensity, statement,
         repetition, actor, gender, duration_s
    """
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

    print(f"[INFO] Tìm thấy {len(wav_files)} files .wav trong {path}")

    rows = []
    for fp in wav_files:
        parts = fp.stem.split("-")
        if len(parts) < 7:
            continue
        try:
            modality    = int(parts[0])   # 03 = audio-only
            vocal_ch    = int(parts[1])   # 01 = speech
            emo_code    = int(parts[2])
            intensity   = int(parts[3])   # 01=normal, 02=strong
            statement   = int(parts[4])   # 01="Kids are talking...", 02="Dogs are sitting..."
            repetition  = int(parts[5])   # 01, 02
            actor       = int(parts[6])

            if emo_code not in EMOTIONS:
                continue

            # Actor 1–12 = female, 13–24 = male (RAVDESS convention: odd=male, even=female)
            gender = "Female" if actor % 2 == 0 else "Male"

            # Thời lượng thực tế
            dur = librosa.get_duration(path=str(fp))

            rows.append({
                "path":        str(fp),
                "filename":    fp.name,
                "emotion":     EMOTIONS[emo_code],
                "emotion_code": emo_code,
                "intensity":   "normal" if intensity == 1 else "strong",
                "statement":   f"stmt{statement:02d}",
                "repetition":  repetition,
                "actor":       actor,
                "gender":      gender,
                "duration_s":  round(dur, 3),
            })
        except (ValueError, IndexError):
            continue

    df = pd.DataFrame(rows)
    print(f"[INFO] Parse được {len(df)} samples  |  "
          f"{df['emotion'].nunique()} cảm xúc  |  "
          f"{df['actor'].nunique()} actors")
    return df


# ============================================================================
# PHẦN 2: BIỂU ĐỒ PHÂN PHỐI LỚP
# ============================================================================

def plot_class_distribution(df: pd.DataFrame,
                             save_path: str = str(OUT_DIR / "ravdess_01_class_distribution.png")) -> None:
    """
    Biểu đồ 1: Phân phối số lượng samples theo cảm xúc.
    Panel trái: horizontal bar chart | Panel phải: pie chart
    """
    counts = df["emotion"].value_counts().reindex(EMOTIONS.values(), fill_value=0)
    colors = [EMOTION_COLORS[e] for e in counts.index]
    labels_emoji = [f"{EMOTION_EMOJIS[e]} {e.capitalize()}" for e in counts.index]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"RAVDESS – Phân phối cảm xúc\n"
        f"({len(df)} samples  |  {df['actor'].nunique()} actors  |  "
        f"{df['emotion'].nunique()} classes)",
        fontsize=13, fontweight="bold",
    )

    # ── Bar chart ──
    bars = axes[0].barh(labels_emoji[::-1], counts.values[::-1],
                        color=colors[::-1], edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, counts.values[::-1]):
        axes[0].text(val + 2, bar.get_y() + bar.get_height() / 2,
                     f"  {val}", va="center", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Số lượng samples")
    axes[0].set_title("Số lượng theo cảm xúc", fontweight="bold")
    axes[0].axvline(counts.mean(), color="red", linestyle="--",
                    linewidth=1.5, label=f"Mean = {counts.mean():.0f}")
    axes[0].legend()
    axes[0].set_xlim(0, counts.max() * 1.2)

    # ── Pie chart ──
    wedges, texts, autotexts = axes[1].pie(
        counts.values, labels=labels_emoji,
        colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.80,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")
    axes[1].set_title("Tỷ lệ phần trăm", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] {save_path}")


# ============================================================================
# PHẦN 3: PHÂN TÍCH THEO ACTOR & GIỚI TÍNH
# ============================================================================

def plot_actor_analysis(df: pd.DataFrame,
                        save_path: str = str(OUT_DIR / "ravdess_02_actor_analysis.png")) -> None:
    """
    Biểu đồ 2: Số samples per actor, màu theo giới tính.
    Panel dưới: stacked bar – emotion distribution theo gender.
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle("RAVDESS – Phân tích Actor & Giới tính",
                 fontsize=13, fontweight="bold")

    # ── Panel 1: samples per actor ──
    actor_counts = df.groupby(["actor", "gender"]).size().reset_index(name="count")
    actor_counts = actor_counts.sort_values("actor")
    bar_colors   = ["#E8A0BF" if g == "Female" else "#6495ED"
                    for g in actor_counts["gender"]]
    bars = axes[0].bar(actor_counts["actor"].astype(str),
                       actor_counts["count"],
                       color=bar_colors, edgecolor="white", linewidth=0.6)
    axes[0].set_xlabel("Actor ID")
    axes[0].set_ylabel("Số samples")
    axes[0].set_title("Samples per Actor  (hồng = Female, xanh = Male)", fontweight="bold")
    axes[0].axhline(actor_counts["count"].mean(), color="red",
                    linestyle="--", linewidth=1.5,
                    label=f"Mean = {actor_counts['count'].mean():.0f}")
    axes[0].legend()
    for bar, row in zip(bars, actor_counts.itertuples()):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(row.count), ha="center", va="bottom", fontsize=8)

    # ── Panel 2: emotion distribution per gender ──
    gender_emo = df.groupby(["gender", "emotion"]).size().unstack(fill_value=0)
    gender_emo = gender_emo.reindex(columns=list(EMOTIONS.values()), fill_value=0)
    emo_colors = [EMOTION_COLORS[e] for e in gender_emo.columns]
    gender_emo.plot(kind="bar", stacked=True, ax=axes[1],
                    color=emo_colors, edgecolor="white", linewidth=0.5,
                    width=0.5)
    axes[1].set_xlabel("Giới tính")
    axes[1].set_ylabel("Số samples")
    axes[1].set_title("Phân phối cảm xúc theo giới tính (stacked)", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(title="Emotion", bbox_to_anchor=(1.01, 1), loc="upper left",
                   fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] {save_path}")


# ============================================================================
# PHẦN 4: PHÂN TÍCH THỜI LƯỢNG & INTENSITY
# ============================================================================

def plot_duration_analysis(df: pd.DataFrame,
                           save_path: str = str(OUT_DIR / "ravdess_03_duration_analysis.png")) -> None:
    """
    Biểu đồ 3: Boxplot thời lượng theo cảm xúc + phân phối intensity.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("RAVDESS – Phân tích Thời lượng & Intensity",
                 fontsize=13, fontweight="bold")

    # ── Boxplot duration per emotion ──
    emotion_order = list(EMOTIONS.values())
    palette       = {e: EMOTION_COLORS[e] for e in emotion_order}
    sns.boxplot(data=df, x="emotion", y="duration_s", order=emotion_order,
                palette=palette, ax=axes[0], linewidth=1.2)
    axes[0].set_title("Thời lượng (s) theo Cảm xúc", fontweight="bold")
    axes[0].set_xlabel("Cảm xúc")
    axes[0].set_ylabel("Thời lượng (s)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].axhline(DURATION, color="red", linestyle="--",
                    linewidth=1.5, label=f"Crop threshold = {DURATION}s")
    axes[0].legend(fontsize=9)

    # ── Histogram tổng duration ──
    axes[1].hist(df["duration_s"], bins=40, color="#5DADE2",
                 edgecolor="white", linewidth=0.5)
    axes[1].axvline(df["duration_s"].mean(), color="red", linestyle="--",
                    linewidth=1.5, label=f"Mean = {df['duration_s'].mean():.2f}s")
    axes[1].axvline(DURATION, color="orange", linestyle="--",
                    linewidth=1.5, label=f"Crop = {DURATION}s")
    axes[1].set_xlabel("Thời lượng (s)")
    axes[1].set_ylabel("Số files")
    axes[1].set_title("Phân phối thời lượng tổng thể", fontweight="bold")
    axes[1].legend(fontsize=9)

    # ── Intensity × Emotion heatmap ──
    hm_data = df.groupby(["emotion", "intensity"]).size().unstack(fill_value=0)
    hm_data = hm_data.reindex(emotion_order, fill_value=0)
    sns.heatmap(hm_data, annot=True, fmt="d", cmap="Blues",
                linewidths=0.5, ax=axes[2],
                cbar_kws={"label": "Số samples"})
    axes[2].set_title("Intensity × Emotion", fontweight="bold")
    axes[2].set_xlabel("Intensity")
    axes[2].set_ylabel("Cảm xúc")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] {save_path}")


# ============================================================================
# PHẦN 5: AUDIO LOADING (dùng lại logic từ feature_experiment.py)
# ============================================================================

def load_audio(file_path: str,
               sr: int = SR,
               duration: float = DURATION) -> np.ndarray:
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


def pick_one_per_emotion(df: pd.DataFrame) -> dict[str, str]:
    """Chọn 1 file đại diện cho mỗi cảm xúc (intensity=strong nếu có)."""
    samples: dict[str, str] = {}
    for emo in EMOTIONS.values():
        subset = df[df["emotion"] == emo]
        strong = subset[subset["intensity"] == "strong"]
        row    = strong.iloc[0] if not strong.empty else subset.iloc[0]
        samples[emo] = row["path"]
    return samples


# ============================================================================
# PHẦN 6: WAVEFORM GALLERY
# ============================================================================

def plot_waveform_gallery(samples: dict[str, str],
                          save_path: str = str(OUT_DIR / "ravdess_04_waveform_gallery.png")) -> None:
    """
    Biểu đồ 4: Waveform của 1 sample cho mỗi cảm xúc (2×4 grid).
    """
    emotions = list(EMOTIONS.values())
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    fig.suptitle(
        "RAVDESS – Waveform Gallery (1 sample/emotion, intensity=strong)",
        fontsize=13, fontweight="bold",
    )
    t = np.linspace(0, DURATION, int(SR * DURATION))

    for ax, emo in zip(axes.flat, emotions):
        y  = load_audio(samples[emo])
        ax.plot(t, y, color=EMOTION_COLORS[emo], linewidth=0.6, alpha=0.9)
        ax.set_title(f"{EMOTION_EMOJIS[emo]} {emo.capitalize()}",
                     fontweight="bold", fontsize=11)
        ax.set_xlabel("Thời gian (s)", fontsize=8)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlim(0, DURATION)

        # Highlight RMS energy
        frame_len = N_FFT
        rms  = librosa.feature.rms(y=y, frame_length=frame_len,
                                   hop_length=HOP_LENGTH)[0]
        t_rms = librosa.frames_to_time(np.arange(len(rms)), sr=SR,
                                        hop_length=HOP_LENGTH)
        ax2   = ax.twinx()
        ax2.plot(t_rms, rms, color="grey", linewidth=1.0,
                 alpha=0.5, linestyle="--")
        ax2.set_ylabel("RMS", fontsize=7, color="grey")
        ax2.tick_params(axis="y", colors="grey", labelsize=7)
        ax2.set_ylim(0, rms.max() * 2.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] {save_path}")


# ============================================================================
# PHẦN 7: SPECTROGRAM GALLERY
# ============================================================================

def plot_spectrogram_gallery(samples: dict[str, str],
                              save_path: str = str(OUT_DIR / "ravdess_05_spectrogram_gallery.png")) -> None:
    """
    Biểu đồ 5: Log-Mel Spectrogram 8 cảm xúc (2×4 grid).
    """
    emotions = list(EMOTIONS.values())
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    fig.suptitle(
        f"RAVDESS – Log-Mel Spectrogram Gallery\n"
        f"(n_mels={N_MELS}, n_fft={N_FFT}, hop={HOP_LENGTH})",
        fontsize=13, fontweight="bold",
    )

    for ax, emo in zip(axes.flat, emotions):
        y   = load_audio(samples[emo])
        mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS,
                                             n_fft=N_FFT, hop_length=HOP_LENGTH)
        db  = librosa.power_to_db(mel, ref=np.max)
        img = librosa.display.specshow(db, sr=SR, hop_length=HOP_LENGTH,
                                        x_axis="time", y_axis="mel", ax=ax)
        ax.set_title(f"{EMOTION_EMOJIS[emo]} {emo.capitalize()}",
                     fontweight="bold", fontsize=11)
        ax.set_xlabel("Thời gian (s)", fontsize=8)
        ax.set_ylabel("Mel (Hz)", fontsize=8)
        fig.colorbar(img, ax=ax, format="%+2.0f dB",
                     pad=0.02).ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] {save_path}")


# ============================================================================
# PHẦN 8: MFCC GALLERY
# ============================================================================

def plot_mfcc_gallery(samples: dict[str, str],
                      save_path: str = str(OUT_DIR / "ravdess_06_mfcc_gallery.png")) -> None:
    """
    Biểu đồ 6: MFCC (40 coeff) của 8 cảm xúc kèm delta overlay.
    """
    emotions = list(EMOTIONS.values())
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    fig.suptitle(
        f"RAVDESS – MFCC Gallery (n_mfcc={N_MFCC}) + Δ overlay",
        fontsize=13, fontweight="bold",
    )

    for ax, emo in zip(axes.flat, emotions):
        y    = load_audio(samples[emo])
        mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC,
                                     n_fft=N_FFT, hop_length=HOP_LENGTH)
        img  = librosa.display.specshow(mfcc, sr=SR, hop_length=HOP_LENGTH,
                                         x_axis="time", ax=ax)
        ax.set_title(f"{EMOTION_EMOJIS[emo]} {emo.capitalize()}",
                     fontweight="bold", fontsize=11)
        ax.set_xlabel("Thời gian (s)", fontsize=8)
        ax.set_ylabel("MFCC coeff", fontsize=8)
        fig.colorbar(img, ax=ax, pad=0.02).ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] {save_path}")


# ============================================================================
# PHẦN 9: BOXPLOT THỐNG KÊ ĐẶC TRƯNG THEO CẢM XÚC
# ============================================================================

def _extract_stats(df: pd.DataFrame,
                   max_samples: int = 100) -> pd.DataFrame:
    """
    Trích xuất ZCR, RMSE, Spectral Centroid, Bandwidth,
    MFCC-1, MFCC-2 cho mỗi file (giới hạn max_samples/emotion).
    """
    rows = []
    emotions = list(EMOTIONS.values())
    print("[INFO] Trích xuất thống kê đặc trưng (có thể mất vài phút)...")

    for emo in emotions:
        subset = df[df["emotion"] == emo].head(max_samples)
        for _, row in subset.iterrows():
            try:
                y = load_audio(row["path"])
                zcr      = float(np.mean(librosa.feature.zero_crossing_rate(y)))
                rmse     = float(np.mean(librosa.feature.rms(y=y)))
                centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=SR)))
                bwidth   = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=SR)))
                rolloff  = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=SR)))
                mfcc     = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC,
                                                 n_fft=N_FFT, hop_length=HOP_LENGTH)
                rows.append({
                    "emotion":    emo,
                    "gender":     row["gender"],
                    "intensity":  row["intensity"],
                    "ZCR":        zcr,
                    "RMSE":       rmse,
                    "Centroid":   centroid,
                    "Bandwidth":  bwidth,
                    "Rolloff":    rolloff,
                    "MFCC_1":     float(np.mean(mfcc[0])),
                    "MFCC_2":     float(np.mean(mfcc[1])),
                    "MFCC_mean_all": float(np.mean(mfcc)),
                })
            except Exception:
                continue

    print(f"[INFO] Đã trích xuất {len(rows)} samples")
    return pd.DataFrame(rows)


def plot_feature_stats(stats_df: pd.DataFrame,
                       save_path: str = str(OUT_DIR / "ravdess_07_feature_stats.png")) -> None:
    """
    Biểu đồ 7: Boxplot 6 đặc trưng quan trọng theo cảm xúc.
    """
    features   = ["ZCR", "RMSE", "Centroid", "Bandwidth", "Rolloff", "MFCC_mean_all"]
    feat_names = {
        "ZCR":          "Zero Crossing Rate",
        "RMSE":         "RMS Energy",
        "Centroid":     "Spectral Centroid (Hz)",
        "Bandwidth":    "Spectral Bandwidth (Hz)",
        "Rolloff":      "Spectral Rolloff (Hz)",
        "MFCC_mean_all": "MFCC Mean (all coeff)",
    }
    emotion_order = list(EMOTIONS.values())
    palette       = {e: EMOTION_COLORS[e] for e in emotion_order}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "RAVDESS – Boxplot Đặc trưng âm thanh theo Cảm xúc\n"
        "(đồng nhất với feature_experiment.py)",
        fontsize=13, fontweight="bold",
    )

    for ax, feat in zip(axes.flat, features):
        sns.boxplot(data=stats_df, x="emotion", y=feat,
                    order=emotion_order, palette=palette,
                    ax=ax, linewidth=1.0, fliersize=3)
        ax.set_title(feat_names[feat], fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=35)
        ax.set_ylabel(feat)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] {save_path}")


# ============================================================================
# PHẦN 10: TƯƠNG QUAN MFCC GIỮA CÁC CẢM XÚC
# ============================================================================

def plot_mfcc_correlation(samples: dict[str, str],
                           save_path: str = str(OUT_DIR / "ravdess_08_correlation.png")) -> None:
    """
    Biểu đồ 8: Heatmap tương quan cosine giữa MFCC mean vectors (8 emotions).
    Giúp thấy cảm xúc nào "gần nhau" về đặc trưng MFCC.
    """
    emotions   = list(EMOTIONS.values())
    mfcc_means = {}

    for emo in emotions:
        y    = load_audio(samples[emo])
        mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC,
                                     n_fft=N_FFT, hop_length=HOP_LENGTH)
        mfcc_means[emo] = np.mean(mfcc, axis=1)   # (N_MFCC,)

    # Cosine similarity matrix
    mat = np.array([mfcc_means[e] for e in emotions])  # (8, N_MFCC)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    mat_n = mat / norms
    corr  = mat_n @ mat_n.T   # (8, 8) cosine similarity

    labels_emoji = [f"{EMOTION_EMOJIS[e]} {e}" for e in emotions]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "RAVDESS – Tương quan MFCC giữa các cảm xúc",
        fontsize=13, fontweight="bold",
    )

    # ── Heatmap cosine similarity ──
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=labels_emoji, yticklabels=labels_emoji,
                linewidths=0.5, ax=axes[0], vmin=-1, vmax=1,
                cbar_kws={"label": "Cosine Similarity"})
    axes[0].set_title("Cosine Similarity (MFCC mean vectors)",
                      fontweight="bold")
    axes[0].tick_params(axis="x", rotation=40)
    axes[0].tick_params(axis="y", rotation=0)

    # ── Line plot: MFCC mean profile per emotion ──
    for emo in emotions:
        axes[1].plot(range(N_MFCC), mfcc_means[emo],
                     color=EMOTION_COLORS[emo],
                     label=f"{EMOTION_EMOJIS[emo]} {emo}",
                     linewidth=1.5, alpha=0.85)
    axes[1].set_xlabel("MFCC Coefficient Index")
    axes[1].set_ylabel("Mean Value")
    axes[1].set_title("MFCC Mean Profile (mỗi cảm xúc)", fontweight="bold")
    axes[1].legend(fontsize=8, loc="upper right", ncol=2)
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    print("\n" + "=" * 65)
    print("  RAVDESS DATASET VISUALIZER")
    print(f"  Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print("=" * 65)

    # 1. Quét metadata
    df = scan_ravdess(RAVDESS_PATH)

    # Tóm tắt nhanh
    print(f"\n{'─'*45}")
    print("  TỔNG QUAN DATASET")
    print(f"{'─'*45}")
    print(f"  Tổng samples     : {len(df)}")
    print(f"  Số cảm xúc       : {df['emotion'].nunique()}")
    print(f"  Số actors        : {df['actor'].nunique()} "
          f"({df[df['gender']=='Female']['actor'].nunique()} F / "
          f"{df[df['gender']=='Male']['actor'].nunique()} M)")
    print(f"  Thời lượng TB    : {df['duration_s'].mean():.2f}s "
          f"(min={df['duration_s'].min():.2f}s, max={df['duration_s'].max():.2f}s)")
    print(f"  Samples cần crop : {(df['duration_s'] > DURATION).sum()} "
          f"/ {len(df)} (>{DURATION}s)")
    print(f"\n  Phân phối cảm xúc:")
    for emo, cnt in df["emotion"].value_counts().items():
        bar = "█" * (cnt // 10)
        print(f"    {emo:12s}: {cnt:4d}  {bar}")

    # 2. Chọn 1 sample/emotion để visualize audio
    samples = pick_one_per_emotion(df)

    print(f"\n{'─'*45}")
    print("  ĐANG VẼ BIỂU ĐỒ...")
    print(f"{'─'*45}")

    # 3. Vẽ tuần tự
    plot_class_distribution(df)
    plot_actor_analysis(df)
    plot_duration_analysis(df)
    plot_waveform_gallery(samples)
    plot_spectrogram_gallery(samples)
    plot_mfcc_gallery(samples)

    # Trích xuất stats cho tất cả files (giới hạn 80/emotion để nhanh)
    stats_df = _extract_stats(df, max_samples=80)
    plot_feature_stats(stats_df)
    plot_mfcc_correlation(samples)

    print(f"\n{'='*65}")
    print("  HOÀN THÀNH! Đã tạo 8 file PNG:")
    print("  1. ravdess_01_class_distribution.png")
    print("  2. ravdess_02_actor_analysis.png")
    print("  3. ravdess_03_duration_analysis.png")
    print("  4. ravdess_04_waveform_gallery.png")
    print("  5. ravdess_05_spectrogram_gallery.png")
    print("  6. ravdess_06_mfcc_gallery.png")
    print("  7. ravdess_07_feature_stats.png")
    print("  8. ravdess_08_correlation.png")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
