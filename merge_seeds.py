"""
=============================================================================
GỘP KẾT QUẢ ĐA SEED CHO CẢI TIẾN #3 (speaker_adversarial.py)
Đề tài: Hệ thống gợi ý sản phẩm dựa trên phân tích giọng nói và cảm xúc
Học viên: Nguyễn Tấn Nhu | GVHD: TS. Bùi Thanh Hùng
=============================================================================

TẠI SAO CẦN FILE NÀY

  run_seed_sweep.sh chạy 3 seed × 4 config × 6 fold = 72 lần train, mỗi seed
  ghi checkpoint vào một thư mục RIÊNG (seed_42/, seed_43/, seed_44/) để
  không lẫn nhau. Hệ quả: mỗi thư mục chỉ tự tổng hợp được 6 fold của riêng
  nó. Không có bước gộp thì 72 điểm dữ liệu nằm rời rạc và toàn bộ lý do
  chạy đa seed (tăng statistical power) mất sạch.

  File này gộp lại và tính thống kê trên toàn bộ 72 điểm.

VÌ SAO ĐA SEED LẠI QUAN TRỌNG (lập luận để viết vào bài báo)

  Với n = 6 fold, paired t-test chỉ phát hiện được chênh lệch lớn; Δ ≤ 3%
  gần như chắc chắn cho ra p > 0.05 dù có thật. Wilcoxon còn tệ hơn: p sàn
  lý thuyết = 2/2^6 = 0.0312, tức KHÔNG BAO GIỜ xuống dưới 0.05 kể cả khi
  một cấu hình thắng cả 6/6 fold. Với n = 18 (3 seed × 6 fold), p sàn của
  Wilcoxon rơi xuống ~7.6e-6 và t-test đủ power cho Δ ≥ 1%.

HAI MỨC TỔNG HỢP — BÁO CÁO CẢ HAI

  (a) Pooled (n = n_seed × n_fold): coi mỗi lần train là một quan sát.
      Dùng cho kiểm định ghép cặp, vì mỗi cặp (Seed, Fold) là một điều kiện
      giống hệt nhau giữa các cấu hình → ghép cặp hợp lệ.
  (b) Seed-level (n = n_seed): lấy trung bình 6 fold trong từng seed trước,
      rồi mới tính std GIỮA các seed. Con số này trả lời câu hỏi khác:
      "chạy lại toàn bộ thí nghiệm thì kết quả xê dịch bao nhiêu?"

  Std của (a) luôn lớn hơn (b) vì chứa cả biến thiên giữa các nhóm speaker.
  Trong bài báo nên ghi rõ đang dùng loại nào — người phản biện sẽ hỏi.

CÁCH DÙNG

  python merge_seeds.py
  python merge_seeds.py --base-dir /content/drive/MyDrive/ravdess_experiment/outputs_speaker_adversarial
  python merge_seeds.py --dirs out_seed42 out_seed43 out_seed44

KHÔNG PHỤ THUỘC torch / transformers / sklearn — chỉ numpy, pandas,
matplotlib, scipy. Chạy được trên máy cá nhân, không cần GPU, vài giây.
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Thứ tự cấu hình trong ablation (khớp CONFIGS của speaker_adversarial.py)
CONFIG_ORDER = ["base", "aug", "grl", "aug+grl"]

# Nhãn lớp: speaker_adversarial.py dùng LabelEncoder trên chuỗi tên cảm xúc
# (EMOTIONS của hubert_finetune.py), mà LabelEncoder sắp xếp theo ALPHABET.
# Vì vậy chỉ số 0..7 trong y_true/y_pred ánh xạ theo đúng thứ tự dưới đây.
# (Hardcode để file này không phải import torch chỉ để lấy 8 chuỗi.)
CLASSES = ["angry", "calm", "disgust", "fearful",
           "happy", "neutral", "sad", "surprised"]

# Thứ tự dễ đọc khi vẽ (theo nhóm arousal/valence, không theo alphabet)
EMO_ORDER = ["neutral", "calm", "happy", "sad",
             "angry", "fearful", "disgust", "surprised"]

METRICS_FOR_TESTS = [
    ("Accuracy(%)",      "Accuracy"),
    ("F1_macro(%)",      "F1-macro"),
    ("spk_probe_lr(%)",  "Speaker-probe (LR, tuyến tính)"),
    ("spk_probe_mlp(%)", "Speaker-probe (MLP, phi tuyến)"),
]


# ============================================================================
# PHẦN 1: THU THẬP CHECKPOINT
# ============================================================================

def _infer_seed(path: Path, default: int | None = None) -> int | None:
    """Suy seed từ tên thư mục dạng seed_43 / seed43 / out_seed43."""
    for part in reversed(path.resolve().parts):
        m = re.search(r"seed[_-]?(\d+)", part, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return default


def collect(dirs: list[Path]) -> tuple[pd.DataFrame, dict]:
    """
    Quét mọi _adv_*.json trong các thư mục đã cho.

    Trả về (df các dòng metric, dict predictions gộp theo config).
    Checkpoint cũ không có trường "Seed" → suy từ tên thư mục.
    """
    rows, preds = [], {}
    n_files = 0

    for d in dirs:
        for ck in sorted(d.glob("_adv_*.json")):
            try:
                s = json.loads(ck.read_text())
            except json.JSONDecodeError:
                print(f"  [WARN] Bỏ qua file hỏng: {ck}")
                continue

            row = dict(s["row"])
            if "Seed" not in row:
                seed = _infer_seed(ck)
                if seed is None:
                    print(f"  [WARN] Không suy được seed cho {ck} → bỏ qua. "
                          f"Đặt checkpoint trong thư mục tên dạng 'seed_43'.")
                    continue
                row["Seed"] = seed
            rows.append(row)
            n_files += 1

            if s.get("y_true") is not None:
                key = (row["Config"], row["Seed"])
                p = preds.setdefault(key, {"true": [], "pred": []})
                p["true"].extend(s["y_true"])
                p["pred"].extend(s["y_pred"])

    print(f"  Đã đọc {n_files} checkpoint từ {len(dirs)} thư mục.")
    return pd.DataFrame(rows), preds


def check_completeness(df: pd.DataFrame) -> None:
    """In ma trận (Config × Seed) → thấy ngay chỗ nào còn thiếu fold."""
    print("\n" + "=" * 76)
    print("  ĐỘ ĐẦY ĐỦ CỦA SWEEP (số fold có sẵn cho mỗi Config × Seed)")
    print("=" * 76)

    piv = df.pivot_table(index="Config", columns="Seed", values="Fold",
                         aggfunc="nunique", fill_value=0)
    piv = piv.reindex([c for c in CONFIG_ORDER if c in piv.index])
    print(piv.to_string())

    expected = int(df["Fold"].nunique())
    missing = (piv != expected)
    if missing.values.any():
        print(f"\n  [CẢNH BÁO] Sweep CHƯA đầy đủ (kỳ vọng {expected} fold mỗi ô).")
        for cfg in piv.index:
            for seed in piv.columns:
                got = int(piv.loc[cfg, seed])
                if got != expected:
                    print(f"    • {cfg:8s} seed {seed}: {got}/{expected} fold"
                          f"  →  SEED={seed} python speaker_adversarial.py "
                          f"--configs {cfg}")
        print("  Các con số dưới đây vẫn tính được, nhưng KHÔNG cân bằng giữa")
        print("  các cấu hình — kiểm định ghép cặp sẽ chỉ dùng phần giao nhau.")
    else:
        print(f"\n  ✓ Đầy đủ: {len(piv.index)} config × {len(piv.columns)} seed "
              f"× {expected} fold = {len(df)} lần train.")


# ============================================================================
# PHẦN 2: TỔNG HỢP HAI MỨC
# ============================================================================

def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    (a) pooled  : mean ± std trên TOÀN BỘ (seed × fold)
    (b) by-seed : trung bình từng seed trước, std GIỮA các seed
    """
    pooled = df.groupby("Config").agg(
        n=("Accuracy(%)", "size"),
        Acc=("Accuracy(%)", "mean"),
        Acc_std=("Accuracy(%)", "std"),
        F1=("F1_macro(%)", "mean"),
        F1_std=("F1_macro(%)", "std"),
    ).round(2)

    per_seed = df.groupby(["Config", "Seed"])["Accuracy(%)"].mean().reset_index()
    by_seed = per_seed.groupby("Config").agg(
        n_seed=("Accuracy(%)", "size"),
        Acc=("Accuracy(%)", "mean"),
        Acc_std=("Accuracy(%)", "std"),
    ).round(2)

    order = [c for c in CONFIG_ORDER if c in pooled.index]
    return pooled.reindex(order), by_seed.reindex(order)


def print_ablation(pooled: pd.DataFrame, by_seed: pd.DataFrame,
                   df: pd.DataFrame) -> None:
    print("\n" + "=" * 76)
    print("  ABLATION STUDY – GỘP TOÀN BỘ SEED")
    print("=" * 76)

    base = pooled.loc["base", "Acc"] if "base" in pooled.index else None
    print(f"  {'Cấu hình':10s} {'n':>4s} {'Accuracy (pooled)':>20s} "
          f"{'F1':>9s} {'Δ base':>9s}")
    print("  " + "─" * 60)
    for c in pooled.index:
        a, s = pooled.loc[c, "Acc"], pooled.loc[c, "Acc_std"]
        d = f"{a - base:+.2f}%" if base is not None and c != "base" else "—"
        print(f"  {c:10s} {int(pooled.loc[c,'n']):4d} "
              f"{a:12.2f} ± {s:<5.2f} {pooled.loc[c,'F1']:8.2f}% {d:>9s}")

    print(f"\n  {'Cấu hình':10s} {'n_seed':>7s} {'Accuracy (seed-level)':>24s}")
    print("  " + "─" * 45)
    for c in by_seed.index:
        s = by_seed.loc[c, "Acc_std"]
        s_txt = f"± {s:.2f}" if pd.notna(s) else "±  n/a"
        print(f"  {c:10s} {int(by_seed.loc[c,'n_seed']):7d} "
              f"{by_seed.loc[c,'Acc']:16.2f} {s_txt}")

    print("\n  • pooled     : mỗi lần train là một quan sát (gồm cả biến thiên")
    print("                 giữa các nhóm speaker) → std lớn hơn.")
    print("  • seed-level : chạy lại cả thí nghiệm thì xê dịch bao nhiêu.")
    print("\n  Đối chiếu mốc cũ: HuBERT fine-tuned không aug/grl = 72.98% (LOSGO,")
    print("  1 seed) — chính là cấu hình 'base'.")

    # Ổn định theo từng nhóm speaker: fold nào khó nhất?
    if df["Fold"].nunique() > 1:
        print("\n  ── Accuracy theo nhóm speaker (trung bình mọi seed & config) ──")
        fold_mean = df.groupby("Fold")["Accuracy(%)"].agg(["mean", "std"]).round(2)
        for f in fold_mean.index:
            m, s = fold_mean.loc[f, "mean"], fold_mean.loc[f, "std"]
            bar = "█" * int(m / 3)
            print(f"    {f:20s} {m:6.2f} ± {s:5.2f}  {bar}")
        hardest = fold_mean["mean"].idxmin()
        easiest = fold_mean["mean"].idxmax()
        print(f"    → Khó nhất: {hardest} ({fold_mean.loc[hardest,'mean']:.2f}%)"
              f"  |  Dễ nhất: {easiest} ({fold_mean.loc[easiest,'mean']:.2f}%)")


# ============================================================================
# PHẦN 3: KIỂM ĐỊNH GHÉP CẶP TRÊN (SEED, FOLD)
# ============================================================================

def paired_tests(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """
    Ghép cặp theo (Seed, Fold): cùng seed, cùng nhóm speaker test → hai cấu
    hình chỉ khác nhau ở aug/grl. Đây là điều kiện để paired test hợp lệ.
    """
    from scipy import stats

    piv = df.pivot_table(index=["Seed", "Fold"], columns="Config",
                         values=metric_col, aggfunc="mean")
    piv = piv.reindex(columns=[c for c in CONFIG_ORDER if c in piv.columns])
    piv = piv.dropna()
    if piv.empty or piv.shape[1] < 2:
        return pd.DataFrame()

    rows = []
    cfgs = list(piv.columns)
    for i in range(len(cfgs)):
        for j in range(i + 1, len(cfgs)):
            a_name, b_name = cfgs[i], cfgs[j]
            a, b = piv[a_name].values, piv[b_name].values
            diff = b - a
            n = len(diff)

            p_t = stats.ttest_rel(b, a).pvalue
            try:
                p_w = stats.wilcoxon(b, a).pvalue
            except ValueError:      # mọi diff = 0
                p_w = 1.0
            sd = float(diff.std(ddof=1))
            d_val = float(diff.mean()) / sd if sd > 0 else 0.0

            rows.append({
                # Quy ước: Δ = (cấu hình SAU) − (cấu hình TRƯỚC), dương nghĩa là
                # cấu hình đứng sau tốt hơn. Cột "Thắng" đếm số cặp Δ > 0.
                "So sánh":    f"{a_name} vs {b_name}",
                "Δ (sau−trước)": round(float(diff.mean()), 2),
                "t-test p":   round(float(p_t), 5),
                "Wilcoxon p": round(float(p_w), 5),
                "Cohen_d":    round(d_val, 2),
                "n":          n,
                "Thắng":      f"{int((diff > 0).sum())}/{n}",
                "Kết luận":   ("có ý nghĩa (p<0.05)" if p_t < 0.05
                               else "KHÔNG đủ bằng chứng"),
            })
    return pd.DataFrame(rows)


def run_all_tests(df: pd.DataFrame, out_dir: Path) -> None:
    try:
        import scipy  # noqa: F401
    except ImportError:
        print("\n  [WARN] Cần scipy để kiểm định: pip install scipy")
        return

    n_pairs = len(df[["Seed", "Fold"]].drop_duplicates())
    n_seed = df["Seed"].nunique()
    n_fold = df["Fold"].nunique()

    print("\n" + "=" * 76)
    print(f"  KIỂM ĐỊNH THỐNG KÊ – paired trên (Seed, Fold)")
    print(f"  n = {n_pairs}  ({n_seed} seed × {n_fold} fold)")
    print("=" * 76)

    all_tests = []
    for col, label in METRICS_FOR_TESTS:
        if col not in df.columns or df[col].isna().all():
            continue
        st = paired_tests(df, col)
        if st.empty:
            continue
        st.insert(0, "Metric", label)
        all_tests.append(st)

        print(f"\n  ── {label} ──")
        print(st.drop(columns=["Metric"]).to_string(index=False))

    if not all_tests:
        print("  Không đủ dữ liệu để kiểm định.")
        return

    st_all = pd.concat(all_tests, ignore_index=True)
    st_all.to_csv(out_dir / "statistical_tests_merged.csv", index=False)

    n_tests = len(st_all)
    w_floor = 2.0 ** (-n_pairs) * 2
    print("\n  ── Ghi chú đọc kết quả ──")
    print(f"  • Wilcoxon p sàn lý thuyết với n={n_pairs}: {w_floor:.2e}")
    if w_floor > 0.05:
        print(f"    → VẪN > 0.05: Wilcoxon không thể đạt ý nghĩa dù thắng "
              f"{n_pairs}/{n_pairs}. Cần thêm seed.")
    else:
        print(f"    → < 0.05: n đã đủ để Wilcoxon dùng được, không còn bị chặn")
        print(f"      như hồi n=6 (p sàn 0.0312).")
    print(f"  • Multiple comparisons: {n_tests} test → ngưỡng Bonferroni "
          f"= 0.05/{n_tests} ≈ {0.05/n_tests:.5f}")
    print(f"  • Cohen's d: 0.2 nhỏ | 0.5 vừa | 0.8 lớn")
    print(f"\n  → {out_dir / 'statistical_tests_merged.csv'}")


# ============================================================================
# PHẦN 4: PER-EMOTION GỘP MỌI SEED
# ============================================================================

def _confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    cm = np.zeros((k, k), dtype=float)
    np.add.at(cm, (y_true, y_pred), 1.0)
    return cm


def per_emotion(preds: dict, out_dir: Path) -> None:
    """
    Gộp predictions của MỌI seed cho từng config → per-emotion + confusion.

    Gộp nhiều seed làm bảng per-emotion ổn định hơn hẳn: mỗi lớp có
    n_seed × 180 mẫu thay vì 180, nên F1 từng lớp bớt nhiễu.
    """
    if not preds:
        print("\n  [INFO] Checkpoint không kèm predictions → bỏ qua per-emotion.")
        return

    print("\n" + "=" * 76)
    print("  PHÂN TÍCH PER-EMOTION (gộp mọi seed, mọi fold)")
    print("=" * 76)

    by_cfg: dict[str, dict[str, list]] = {}
    for (cfg, _seed), d in preds.items():
        p = by_cfg.setdefault(cfg, {"true": [], "pred": []})
        p["true"].extend(d["true"])
        p["pred"].extend(d["pred"])

    k = len(CLASSES)
    emo_rows, pair_rows = [], []

    for cfg in [c for c in CONFIG_ORDER if c in by_cfg]:
        yt = np.asarray(by_cfg[cfg]["true"], dtype=int)
        yp = np.asarray(by_cfg[cfg]["pred"], dtype=int)
        if yt.size == 0:
            continue
        if yt.max() >= k or yp.max() >= k:
            print(f"  [WARN] {cfg}: nhãn vượt {k} lớp → bỏ qua")
            continue

        cm = _confusion(yt, yp, k)
        support = cm.sum(axis=1)
        pred_tot = cm.sum(axis=0)
        diag = np.diag(cm)

        recall = np.divide(diag, support, out=np.zeros(k), where=support > 0)
        precision = np.divide(diag, pred_tot, out=np.zeros(k), where=pred_tot > 0)
        denom = precision + recall
        f1 = np.divide(2 * precision * recall, denom,
                       out=np.zeros(k), where=denom > 0)
        acc = diag.sum() / cm.sum() * 100

        for i, emo in enumerate(CLASSES):
            emo_rows.append({
                "Config": cfg, "Emotion": emo,
                "Precision(%)": round(precision[i] * 100, 2),
                "Recall(%)":    round(recall[i] * 100, 2),
                "F1(%)":        round(f1[i] * 100, 2),
                "Support":      int(support[i]),
            })

        print(f"\n  [{cfg}]  Acc gộp = {acc:.2f}%  (n = {int(cm.sum())} mẫu)")
        order = np.argsort(f1)
        for i in order:
            bar = "█" * int(f1[i] * 100 / 4)
            print(f"      {CLASSES[i]:10s} F1={f1[i]*100:5.1f}%  {bar}")

        cm_pct = cm / np.clip(support[:, None], 1e-9, None) * 100
        pairs = sorted(((CLASSES[i], CLASSES[j], cm_pct[i, j])
                        for i in range(k) for j in range(k) if i != j),
                       key=lambda x: -x[2])
        print("      3 cặp nhầm nặng nhất:")
        for a, b, v in pairs[:3]:
            print(f"        {a:10s} → {b:10s}: {v:5.1f}%")

        i_n, i_c = CLASSES.index("neutral"), CLASSES.index("calm")
        pair_rows.append({
            "Config": cfg,
            "Accuracy(%)": round(acc, 2),
            "neutral→calm": round(float(cm_pct[i_n, i_c]), 2),
            "calm→neutral": round(float(cm_pct[i_c, i_n]), 2),
            "top_pair": f"{pairs[0][0]}→{pairs[0][1]}",
            "top_pair(%)": round(float(pairs[0][2]), 2),
        })

        _plot_confusion(cm_pct, cfg, acc, out_dir)

    if emo_rows:
        df_emo = pd.DataFrame(emo_rows)
        df_emo.to_csv(out_dir / "per_emotion_merged.csv", index=False)
        pd.DataFrame(pair_rows).to_csv(out_dir / "confusion_pairs_merged.csv",
                                       index=False)
        _plot_per_emotion(df_emo, out_dir)
        print(f"\n  → {out_dir / 'per_emotion_merged.csv'}")


def _plot_confusion(cm_pct: np.ndarray, cfg: str, acc: float,
                    out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            v = cm_pct[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 50 else "black")
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(f"Confusion Matrix – {cfg}\n"
                 f"LOSGO, gộp mọi seed (Acc = {acc:.2f}%)", fontweight="bold")
    fig.colorbar(im, ax=ax, label="Tỷ lệ theo hàng (%)")
    plt.tight_layout()
    plt.savefig(out_dir / f"confusion_merged_{cfg.replace('+','_')}.png",
                dpi=150, bbox_inches="tight")
    plt.close()


def _plot_per_emotion(df_emo: pd.DataFrame, out_dir: Path) -> None:
    piv = df_emo.pivot_table(index="Emotion", columns="Config", values="F1(%)")
    piv = piv.reindex([e for e in EMO_ORDER if e in piv.index])
    piv = piv.reindex(columns=[c for c in CONFIG_ORDER if c in piv.columns])

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(piv.index))
    w = 0.8 / max(1, len(piv.columns))
    for i, c in enumerate(piv.columns):
        ax.bar(x + i * w - 0.4 + w / 2, piv[c].values, w, label=c,
               edgecolor="grey", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index, rotation=20, ha="right")
    ax.set_ylabel("F1 (%)")
    ax.set_title("F1 theo từng cảm xúc – so sánh các cấu hình (gộp mọi seed)",
                 fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "per_emotion_merged.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# PHẦN 5: BIỂU ĐỒ ABLATION
# ============================================================================

def plot_ablation(df: pd.DataFrame, pooled: pd.DataFrame,
                  out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle("Ablation aug / GRL – gộp nhiều seed (LOSGO)",
                 fontsize=13, fontweight="bold")

    # Trái: bar pooled ± std, chấm = trung bình từng seed
    cfgs = list(pooled.index)
    x = np.arange(len(cfgs))
    axes[0].bar(x, pooled["Acc"].values,
                yerr=np.nan_to_num(pooled["Acc_std"].values, nan=0.0),
                capsize=5, color="#2E7D5B", edgecolor="grey", alpha=0.85)

    per_seed = df.groupby(["Config", "Seed"])["Accuracy(%)"].mean().reset_index()
    for si, seed in enumerate(sorted(per_seed["Seed"].unique())):
        sub = per_seed[per_seed.Seed == seed].set_index("Config")
        vals = [sub.loc[c, "Accuracy(%)"] if c in sub.index else np.nan
                for c in cfgs]
        axes[0].scatter(x, vals, zorder=3, s=45, label=f"seed {seed}")

    for i, c in enumerate(cfgs):
        axes[0].text(i, pooled.loc[c, "Acc"] + 1.2,
                     f"{pooled.loc[c,'Acc']:.2f}", ha="center",
                     fontweight="bold", fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cfgs)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Pooled mean ± std (chấm = trung bình mỗi seed)")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    # Phải: từng fold, từng config → thấy fold nào khó
    fold_piv = df.pivot_table(index="Fold", columns="Config",
                              values="Accuracy(%)", aggfunc="mean")
    fold_piv = fold_piv.reindex(columns=[c for c in CONFIG_ORDER
                                         if c in fold_piv.columns])
    for c in fold_piv.columns:
        axes[1].plot(range(1, len(fold_piv) + 1), fold_piv[c].values,
                     "o-", linewidth=2, label=c)
    axes[1].set_xticks(range(1, len(fold_piv) + 1))
    axes[1].set_xticklabels(fold_piv.index, rotation=20, ha="right", fontsize=8)
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Ổn định qua các nhóm speaker (trung bình mọi seed)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "ablation_merged.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out_dir / 'ablation_merged.png'}")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Gộp kết quả đa seed của speaker_adversarial.py")
    ap.add_argument("--base-dir", default="./outputs_speaker_adversarial",
                    help="Thư mục cha chứa các thư mục con seed_42/, seed_43/...")
    ap.add_argument("--dirs", nargs="+", default=None,
                    help="Chỉ định thẳng danh sách thư mục (bỏ qua --base-dir)")
    ap.add_argument("--out-dir", default=None,
                    help="Nơi ghi kết quả gộp (mặc định: <base-dir>/merged)")
    args = ap.parse_args()

    print("\n" + "=" * 76)
    print("  GỘP KẾT QUẢ ĐA SEED – CẢI TIẾN #3 (aug / speaker-adversarial)")
    print("=" * 76)

    if args.dirs:
        dirs = [Path(d) for d in args.dirs]
        base = Path(args.dirs[0]).parent
    else:
        base = Path(args.base_dir)
        dirs = sorted(d for d in base.glob("seed_*") if d.is_dir())
        if base.exists() and list(base.glob("_adv_*.json")):
            # Trường hợp chạy 1 seed thẳng vào base-dir, không có thư mục con
            dirs.append(base)

    dirs = [d for d in dirs if d.exists()]
    if not dirs:
        print(f"\n  [LỖI] Không tìm thấy thư mục kết quả nào.")
        print(f"        Đã tìm trong: {Path(args.base_dir).absolute()}")
        print(f"        Chỉ định thẳng bằng: --dirs <thư_mục_1> <thư_mục_2> ...")
        sys.exit(1)

    print(f"  Thư mục nguồn ({len(dirs)}):")
    for d in dirs:
        print(f"    • {d}")

    df, preds = collect(dirs)
    if df.empty:
        print("\n  [LỖI] Không có checkpoint _adv_*.json nào hợp lệ.")
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else base / "merged"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df.sort_values(["Config", "Seed", "Fold"]).reset_index(drop=True)
    df.to_csv(out_dir / "merged_all_runs.csv", index=False)

    check_completeness(df)

    pooled, by_seed = summarize(df)
    pooled.to_csv(out_dir / "summary_pooled.csv")
    by_seed.to_csv(out_dir / "summary_by_seed.csv")
    print_ablation(pooled, by_seed, df)

    run_all_tests(df, out_dir)

    # Speaker-probe: GRL có thật sự xoá được danh tính không?
    if "spk_probe_lr(%)" in df.columns and df["spk_probe_lr(%)"].notna().any():
        print("\n" + "=" * 76)
        print("  SPEAKER-PROBE TRÊN EMBEDDING ĐÓNG BĂNG (gộp seed)")
        print("=" * 76)
        cols = [c for c in ("spk_probe_lr(%)", "spk_probe_mlp(%)",
                            "spk_chance(%)", "perm_p_lr", "perm_p_mlp")
                if c in df.columns]
        pr = df.groupby("Config")[cols].mean().round(3)
        pr = pr.reindex([c for c in CONFIG_ORDER if c in pr.index])
        print(pr.to_string())
        pr.to_csv(out_dir / "speaker_probe_merged.csv")
        if "base" in pr.index and "grl" in pr.index:
            base_lr = pr.loc["base", "spk_probe_lr(%)"]
            d_lr = pr.loc["grl", "spk_probe_lr(%)"] - base_lr
            chance = pr["spk_chance(%)"].mean() if "spk_chance(%)" in pr else 0.0
            print(f"\n  Δ LR-probe (grl − base) = {d_lr:+.2f}%  "
                  f"(base={base_lr:.1f}%, chance={chance:.1f}%)")
            # Diễn giải phải bám số liệu, không in cứng một kết luận.
            if d_lr > -2:
                print("  → GRL gần như KHÔNG làm giảm probe. Thông tin danh tính")
                print("    vẫn nằm trong embedding; GRL chỉ đánh bại head đối kháng")
                print("    chứ không xoá được — đúng hạn chế đã biết của DANN/GRL.")
            elif pr.loc["grl", "spk_probe_lr(%)"] > chance + 15:
                print("  → GRL làm probe GIẢM đáng kể nhưng vẫn cao hơn hẳn mức")
                print("    ngẫu nhiên → xoá được MỘT PHẦN danh tính, chưa triệt để.")
            else:
                print("  → GRL kéo probe xuống gần mức ngẫu nhiên → thật sự xoá")
                print("    được danh tính. Đây là kết quả mạnh, cần nhấn trong bài.")
            if "spk_probe_mlp(%)" in pr.columns:
                gap = pr.loc["grl", "spk_probe_mlp(%)"] - pr.loc["grl", "spk_probe_lr(%)"]
                if gap > 5:
                    print(f"  → MLP-probe cao hơn LR-probe {gap:.1f} điểm: danh tính")
                    print("    còn được mã hoá PHI TUYẾN, LR-probe đánh giá thấp mức")
                    print("    rò rỉ thật. Báo cáo cả hai con số.")

    per_emotion(preds, out_dir)
    plot_ablation(df, pooled, out_dir)

    print("\n" + "=" * 76)
    print("  HOÀN THÀNH – Output tại:", out_dir.absolute())
    print("=" * 76)
    for f in sorted(out_dir.iterdir()):
        print(f"    • {f.name}")
    print()


if __name__ == "__main__":
    main()
