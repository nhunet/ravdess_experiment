"""
=============================================================================
CẢI TIẾN #3: SPEAKER-ADVERSARIAL + AUGMENTATION + PHÂN TÍCH PER-EMOTION
Đề tài: Hệ thống gợi ý sản phẩm dựa trên phân tích giọng nói và cảm xúc
Học viên: Nguyễn Tấn Nhu | GVHD: TS. Bùi Thanh Hùng
=============================================================================

ĐỘNG CƠ – XUẤT PHÁT TỪ SỐ LIỆU CỦA CHÍNH MÌNH

  Các thực nghiệm trước đã ĐO ĐƯỢC hai điều:

    (1) Speaker leakage rất lớn:
          5-fold ngẫu nhiên : 89.58%
          LOSGO (giọng lạ)  : 72.98%
          → chênh 16.6 điểm = phần model học "giọng của actor" chứ không
            phải học "cảm xúc".

    (2) std qua các fold = 7.21% — model rất nhạy khi gặp giọng chưa từng nghe.
          Fold dễ (G2): 85.83%   |   Fold khó (G4): 65.00%

  Hai cải tiến trong file này TẤN CÔNG TRỰC TIẾP vấn đề đó:

  ─────────────────────────────────────────────────────────────────────────
  A. SPEAKER-ADVERSARIAL TRAINING (Gradient Reversal Layer – DANN)

     Ý tưởng: gắn thêm một speaker-classifier phụ lên embedding, NHƯNG đặt
     một Gradient Reversal Layer ở giữa. Khi lan truyền ngược:
        • speaker-classifier cố HỌC nhận ra actor
        • GRL ĐẢO DẤU gradient → backbone bị đẩy theo hướng NGƯỢC LẠI,
          tức là học biểu diễn mà speaker-classifier KHÔNG đoán nổi actor
     Kết quả mong đợi: embedding giữ thông tin CẢM XÚC nhưng vứt bỏ thông tin
     DANH TÍNH NGƯỜI NÓI → tổng quát hoá tốt hơn sang giọng lạ.

     Loss = CE_emotion  +  λ · CE_speaker(qua GRL)
     λ tăng dần theo lịch DANN: λ(p) = 2/(1+exp(-10p)) − 1

     ĐÂY LÀ ĐÓNG GÓP KHOA HỌC CHÍNH: phát hiện leakage bằng thực nghiệm,
     rồi THIẾT KẾ LOSS để chống lại chính nó. Không phải mượn kiến trúc.

  ─────────────────────────────────────────────────────────────────────────
  B. WAVEFORM AUGMENTATION

     • Speed perturbation 0.9× / 1.0× / 1.1×  (Ko et al. 2015)
       → đổi cả tốc độ LẪN cao độ → mô phỏng giọng người khác
       → chính là thứ cần để bền vững trước speaker mới
     • Gain ngẫu nhiên ±6 dB      (khác micro / khoảng cách)
     • Gaussian noise SNR 15-30dB (môi trường thu khác nhau)
     • Time shift ±10%            (chống model bám vào vị trí tuyệt đối)

     Augment ON-THE-FLY trong Dataset (không nhân bản RAM).
     KHÔNG augment tập val/test — chỉ train.

  ─────────────────────────────────────────────────────────────────────────
  C. PHÂN TÍCH PER-EMOTION (lấp lỗ hổng phân tích)

     Trước giờ chỉ có accuracy tổng. Reviewer sẽ hỏi: model nhầm cặp nào?
     File này xuất:
       • Per-emotion accuracy/precision/recall/F1 cho TỪNG cấu hình
       • Confusion matrix gộp toàn bộ 6 fold
       • Phân tích riêng cặp neutral ↔ calm (RAVDESS nổi tiếng khó tách)
       • So sánh per-emotion giữa các cấu hình → thấy GRL/Aug giúp lớp nào

ABLATION MATRIX (trả lời "bỏ thành phần X thì giảm bao nhiêu?")

     Cấu hình         Aug    GRL    Kỳ vọng
     ──────────────────────────────────────────────
     base              ✗      ✗     72.98% (đã biết)
     aug               ✓      ✗     +3-6%
     grl               ✗      ✓     +2-5%
     aug+grl           ✓      ✓     tốt nhất?

  → Bảng này chính là ABLATION STUDY reviewer đòi hỏi.

  ⚠️ LƯU Ý QUAN TRỌNG VỀ GRL (rút ra từ kiểm chứng trước khi chạy thật):
     Kiểm chứng trên dữ liệu tổng hợp cho thấy accuracy của speaker-head
     đối kháng (head bị GRL đánh bại) KHÔNG PHẢI bằng chứng rằng thông tin
     danh tính đã bị xoá khỏi embedding — head đó thua thì đương nhiên phải
     tụt accuracy, điều đó không chứng minh gì. Cách đo trung thực là:
     đóng băng encoder rồi train một LINEAR PROBE MỚI để đoán speaker.
     Probe-accuracy cao → GRL KHÔNG xoá được danh tính (hạn chế đã biết
     của DANN/GRL trong literature). Probe-accuracy thấp → GRL thật sự
     hiệu quả. File này dùng speaker_probe() cho mọi cấu hình (kể cả base)
     để có con số so sánh công bằng và trung thực.

GIAO THỨC: LOSGO 6-fold — giống hệt các thực nghiệm trước, so sánh trực tiếp
           được với A=62.43%  B=64.72%  B0=72.98%

CÁCH CHẠY
  python speaker_adversarial.py --quick              # 2 fold, 10 ep (~25 phút)
  python speaker_adversarial.py                      # đủ 4 cấu hình (~3 giờ)
  python speaker_adversarial.py --configs base aug   # chỉ 2 cấu hình
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix,
                             classification_report)

warnings.filterwarnings("ignore")

from hubert_finetune import (
    HubertSER, load_ravdess_16k, set_seed, free_gpu, gpu_mem,
    _evaluate, RANDOM_SEED, BATCH_SIZE, GRAD_ACCUM,
    LR_BACKBONE, LR_HEAD, WEIGHT_DECAY, WARMUP_RATIO, LABEL_SMOOTH,
    TARGET_SR, MAX_SAMPLES, EMBED_DIM,
)
from speaker_independent_benchmark import (
    losgo_splits, make_val_split, _metrics, paired_tests,
)

OUT_DIR = Path(os.environ.get("OUT_DIR", "./outputs_adversarial"))


def out(f: str) -> str:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return str(OUT_DIR / f)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS_DEFAULT = 30
PATIENCE = 6
GRL_LAMBDA_MAX = 1.0          # λ tối đa của gradient reversal
SPK_HIDDEN = 128

CONFIGS = {
    "base":    {"aug": False, "grl": False},
    "aug":     {"aug": True,  "grl": False},
    "grl":     {"aug": False, "grl": True},
    "aug+grl": {"aug": True,  "grl": True},
}

EMO_ORDER = ["neutral", "calm", "happy", "sad",
             "angry", "fearful", "disgust", "surprised"]


# ============================================================================
# PHẦN 1: WAVEFORM AUGMENTATION (B)
# ============================================================================

class AugmentedDataset(Dataset):
    """
    Augment ON-THE-FLY. Chỉ dùng cho TRAIN – val/test không bao giờ augment.

    Speed perturbation là kỹ thuật quan trọng nhất ở đây: đổi tốc độ phát
    (0.9× / 1.1×) làm thay đổi CẢ cao độ, tạo ra "giọng của một người khác".
    Đúng thứ cần để model bớt phụ thuộc vào danh tính người nói.
    """

    def __init__(self, waves: np.ndarray, y: np.ndarray,
                 spk: np.ndarray | None = None, augment: bool = False):
        self.waves = waves
        self.y = y
        self.spk = spk
        self.augment = augment
        self.rng = np.random.RandomState(RANDOM_SEED)

    def __len__(self):
        return len(self.y)

    def _speed_perturb(self, x: np.ndarray, rate: float) -> np.ndarray:
        """Resample tuyến tính → đổi tốc độ + cao độ, rồi pad/trim về đúng độ dài."""
        n = int(len(x) / rate)
        idx = np.linspace(0, len(x) - 1, n)
        y = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
        if len(y) < MAX_SAMPLES:
            y = np.pad(y, (0, MAX_SAMPLES - len(y)))
        return y[:MAX_SAMPLES]

    def _apply(self, x: np.ndarray) -> np.ndarray:
        r = self.rng

        # 1. Speed perturbation (Ko et al. 2015) – quan trọng nhất
        if r.rand() < 0.5:
            x = self._speed_perturb(x, r.choice([0.9, 1.1]))

        # 2. Time shift ±10% – chống bám vị trí tuyệt đối
        if r.rand() < 0.5:
            s = int(r.uniform(-0.1, 0.1) * MAX_SAMPLES)
            x = np.roll(x, s)

        # 3. Gain ±6 dB – mô phỏng micro/khoảng cách khác nhau
        if r.rand() < 0.5:
            x = x * float(10 ** (r.uniform(-6, 6) / 20))

        # 4. Gaussian noise SNR 15-30 dB – môi trường thu khác nhau
        if r.rand() < 0.3:
            snr = r.uniform(15, 30)
            p_sig = np.mean(x ** 2)
            if p_sig > 0:
                p_noise = p_sig / (10 ** (snr / 10))
                x = x + r.randn(len(x)).astype(np.float32) * np.sqrt(p_noise)

        peak = np.abs(x).max()
        if peak > 1.0:
            x = x / peak
        return x.astype(np.float32)

    def __getitem__(self, i):
        x = self.waves[i]
        if self.augment:
            x = self._apply(x.copy())
        out_ = [torch.from_numpy(x), torch.tensor(self.y[i], dtype=torch.long)]
        if self.spk is not None:
            out_.append(torch.tensor(self.spk[i], dtype=torch.long))
        return tuple(out_)


# ============================================================================
# PHẦN 2: GRADIENT REVERSAL LAYER (A) — ĐÓNG GÓP CHÍNH
# ============================================================================

class GradientReversal(torch.autograd.Function):
    """
    Gradient Reversal Layer (Ganin & Lempitsky 2015).

    Forward : hàm đồng nhất (không đổi gì)
    Backward: NHÂN GRADIENT VỚI −λ

    Hệ quả: speaker-classifier đặt SAU lớp này sẽ cố hết sức nhận ra actor,
    nhưng gradient chảy ngược về backbone bị ĐẢO DẤU → backbone bị đẩy theo
    hướng làm speaker-classifier THẤT BẠI.

    Kết quả: embedding giữ thông tin cảm xúc (vì emotion head vẫn học bình
    thường) nhưng VỨT BỎ thông tin danh tính người nói.
    """

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad.neg() * ctx.lambd, None


def grad_reverse(x, lambd: float):
    return GradientReversal.apply(x, lambd)


class HubertSER_Adversarial(HubertSER):
    """
    HuBERT SER + nhánh speaker-adversarial.

    Kiến trúc:
                          ┌─→ emotion head ──────→ 8 lớp cảm xúc  (học bình thường)
        waveform → embed ─┤
                          └─→ GRL(−λ) → speaker head → N actor   (gradient ĐẢO)

    Chỉ THÊM một nhánh phụ; nhánh cảm xúc giữ nguyên hoàn toàn so với model
    đã cho 72.98%, nên so sánh là công bằng.
    """

    def __init__(self, n_classes: int, n_speakers: int, **kw):
        super().__init__(n_classes, **kw)
        self.speaker_head = nn.Sequential(
            nn.Linear(EMBED_DIM, SPK_HIDDEN), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(SPK_HIDDEN, n_speakers),
        )

    def forward_adv(self, x: torch.Tensor, lambd: float):
        """Trả về (logit_cảm_xúc, logit_speaker)."""
        emb = self.embed_fc(self._backbone(x))       # (B, 256)
        emo = self.classifier(emb)
        spk = self.speaker_head(grad_reverse(emb, lambd))
        return emo, spk


def dann_lambda(step: int, total: int, max_lambda: float = GRL_LAMBDA_MAX) -> float:
    """
    Lịch tăng λ của DANN: λ(p) = 2/(1+exp(-10p)) − 1,  p = step/total.

    Tăng DẦN từ 0: giai đoạn đầu để model học cảm xúc trước đã, không thì
    adversarial loss sẽ phá hỏng biểu diễn khi nó còn chưa có gì.
    """
    p = min(1.0, step / max(1, total))
    return float(max_lambda * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0))


@torch.no_grad()
def _extract_emb(model, waves: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Trích embedding 256-d (đóng băng) cho một tập chỉ số."""
    model.eval()
    dl = DataLoader(AugmentedDataset(waves[idx], np.zeros(len(idx), dtype=np.int64)),
                    batch_size=BATCH_SIZE,
                    pin_memory=(DEVICE.type == "cuda"))
    embs = []
    for xb, _ in dl:
        xb = xb.to(DEVICE)
        with torch.autocast(device_type=DEVICE.type,
                            enabled=(DEVICE.type == "cuda")):
            e = model.get_embedding(xb)
        embs.append(e.float().cpu().numpy())
    return np.concatenate(embs)


def speaker_probe(model, waves, actors, idx) -> tuple[float, float]:
    """
    ⚠️ ĐO TRUNG THỰC: embedding CÒN GIỮ bao nhiêu thông tin danh tính?

    TẠI SAO CẦN HÀM NÀY (bài học từ kiểm chứng):
      Bản đầu định "chứng minh GRL hoạt động" bằng accuracy của chính
      speaker-head đối kháng — thấy nó tụt là kết luận GRL thành công.
      SAI. Head đó bị GRL đánh bại nên tất nhiên phải tụt; điều đó KHÔNG
      chứng minh thông tin danh tính đã bị xoá.

      Kiểm chứng trên dữ liệu tổng hợp cho thấy: GRL làm head đối kháng mù đi,
      NHƯNG nếu ĐÓNG BĂNG encoder rồi train một probe MỚI, probe vẫn đoán được
      speaker gần như hoàn hảo → thông tin vẫn còn nguyên trong embedding.
      Đây là phê bình đã biết với DANN/GRL trong literature.

    Cách đo đúng (hàm này):
      1. Đóng băng model đã train
      2. Trích embedding
      3. Train MỘT probe tuyến tính MỚI để đoán speaker
      4. Probe-accuracy CAO  → GRL KHÔNG xoá được danh tính (thất bại)
         Probe-accuracy THẤP → GRL thật sự xoá được (thành công)

    Con số này mới là thứ đưa vào báo cáo được.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    E = _extract_emb(model, waves, idx)
    spk = actors[idx]
    n_spk = len(np.unique(spk))
    if n_spk < 2:
        return float("nan"), float("nan")

    clf = LogisticRegression(max_iter=2000, multi_class="multinomial")
    acc = cross_val_score(clf, E, spk, cv=3, scoring="accuracy").mean() * 100
    chance = 100.0 / n_spk
    return float(acc), float(chance)


# ============================================================================
# PHẦN 3: HUẤN LUYỆN 1 FOLD, 1 CẤU HÌNH
# ============================================================================

def train_config(waves, y, actors, tr, te, n_cls, cfg: dict,
                 epochs: int, fold: str, cfg_name: str) -> dict:
    set_seed(RANDOM_SEED)
    t0 = time.time()
    use_aug, use_grl = cfg["aug"], cfg["grl"]

    real_tr, real_val = make_val_split(tr, y, actors, speaker_disjoint=True)

    # ── Nhãn speaker: ánh xạ actor → 0..N-1 TRONG TRAIN FOLD ────────────────
    spk_tr = None
    n_spk = 0
    if use_grl:
        uniq = np.unique(actors[real_tr])
        spk_map = {a: i for i, a in enumerate(uniq)}
        spk_tr = np.array([spk_map[a] for a in actors[real_tr]])
        n_spk = len(uniq)

    kw = dict(pin_memory=(DEVICE.type == "cuda"),
              num_workers=2 if DEVICE.type == "cuda" else 0)

    tr_ds = AugmentedDataset(waves[real_tr], y[real_tr], spk_tr, augment=use_aug)
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                       drop_last=True, **kw)
    # val/test KHÔNG augment
    val_dl = DataLoader(AugmentedDataset(waves[real_val], y[real_val]),
                        batch_size=BATCH_SIZE, **kw)
    te_dl = DataLoader(AugmentedDataset(waves[te], y[te]),
                       batch_size=BATCH_SIZE, **kw)

    if use_grl:
        model = HubertSER_Adversarial(n_cls, n_spk).to(DEVICE)
    else:
        model = HubertSER(n_cls).to(DEVICE)

    # Optimizer: backbone LR nhỏ, các head LR lớn
    backbone, head = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (backbone if name.startswith("hubert") else head).append(p)
    opt = torch.optim.AdamW(
        [{"params": backbone, "lr": LR_BACKBONE},
         {"params": head, "lr": LR_HEAD}], weight_decay=WEIGHT_DECAY)

    lossfn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    spk_lossfn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    steps = max(1, (len(tr_dl) // GRAD_ACCUM) * epochs)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[LR_BACKBONE, LR_HEAD],
        total_steps=steps, pct_start=WARMUP_RATIO)

    best_f1, best_state, bad, gstep = 0.0, None, 0, 0
    spk_acc_hist = []

    aug_txt = "aug " if use_aug else ""
    grl_txt = f"grl(N={n_spk}) " if use_grl else ""
    print(f"    [{fold}|{cfg_name}] {aug_txt}{grl_txt}training {epochs} epochs...")

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        spk_correct = spk_total = 0

        for i, batch in enumerate(tr_dl):
            if use_grl:
                xb, yb, sb = batch
                sb = sb.to(DEVICE)
            else:
                xb, yb = batch
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            lambd = dann_lambda(gstep, steps) if use_grl else 0.0

            with torch.autocast(device_type=DEVICE.type,
                                enabled=(DEVICE.type == "cuda")):
                if use_grl:
                    emo, spk = model.forward_adv(xb, lambd)
                    loss = lossfn(emo, yb) + spk_lossfn(spk, sb)
                    spk_correct += (spk.argmax(1) == sb).sum().item()
                    spk_total += sb.size(0)
                else:
                    loss = lossfn(model(xb), yb)
                loss = loss / GRAD_ACCUM

            scaler.scale(loss).backward()

            if (i + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad()
                if sch.last_epoch < steps - 1:
                    sch.step()
                gstep += 1

        if use_grl and spk_total:
            spk_acc_hist.append(100.0 * spk_correct / spk_total)

        vp, vt = _evaluate(model, val_dl)
        vf1 = f1_score(vt, vp, average="macro", zero_division=0)

        if vf1 > best_f1:
            best_f1, bad = vf1, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)

    p, t = _evaluate(model, te_dl)
    r = _metrics(t, p, time.time() - t0)

    # ── Accuracy của head đối kháng (CHỈ để tham khảo – KHÔNG phải bằng chứng) ──
    if use_grl and spk_acc_hist:
        r["adv_head_first"] = float(spk_acc_hist[0])
        r["adv_head_last"] = float(spk_acc_hist[-1])

    # ── ĐO TRUNG THỰC: probe tuyến tính MỚI trên embedding đóng băng ──────────
    # Đây mới là bằng chứng thật. Head đối kháng tụt accuracy KHÔNG chứng minh
    # gì cả — nó bị GRL đánh bại nên đương nhiên phải tụt.
    try:
        probe_acc, chance = speaker_probe(model, waves, actors, real_tr)
        r["spk_probe(%)"] = probe_acc
        r["spk_chance(%)"] = chance
        adv = ""
        if use_grl and spk_acc_hist:
            adv = f"  [head đối kháng: {spk_acc_hist[-1]:.1f}%]"
        print(f"    [{fold}|{cfg_name}] speaker-PROBE trên embedding đóng băng "
              f"= {probe_acc:.1f}%  (ngẫu nhiên={chance:.1f}%){adv}")
    except Exception as exc:
        print(f"    [WARN] probe lỗi: {exc}")

    free_gpu(model)
    return r


# ============================================================================
# PHẦN 4: PHÂN TÍCH PER-EMOTION (C)
# ============================================================================

def per_emotion_table(y_true, y_pred, le: LabelEncoder, cfg: str) -> pd.DataFrame:
    """Accuracy / Precision / Recall / F1 cho TỪNG cảm xúc."""
    rep = classification_report(y_true, y_pred, target_names=le.classes_,
                                output_dict=True, zero_division=0)
    rows = []
    for emo in le.classes_:
        d = rep[emo]
        rows.append({
            "Config":       cfg,
            "Emotion":      emo,
            "Precision(%)": round(d["precision"] * 100, 2),
            "Recall(%)":    round(d["recall"] * 100, 2),
            "F1(%)":        round(d["f1-score"] * 100, 2),
            "Support":      int(d["support"]),
        })
    return pd.DataFrame(rows)


def plot_confusion(y_true, y_pred, le, cfg: str, acc: float) -> None:
    cm = confusion_matrix(y_true, y_pred).astype(float)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Tỷ lệ (%)"})
    ax.set_title(f"Confusion Matrix – {cfg}  (LOSGO 6-fold gộp, Acc={acc:.1f}%)",
                 fontweight="bold")
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    plt.tight_layout()
    plt.savefig(out(f"confusion_{cfg.replace('+','_')}.png"), dpi=150,
                bbox_inches="tight")
    plt.close()


def analyze_confusions(y_true, y_pred, le, cfg: str) -> dict:
    """
    Tìm các cặp nhầm lẫn nặng nhất. RAVDESS nổi tiếng khó ở cặp neutral↔calm
    (ngay cả người nghe cũng khó tách) — kiểm chứng bằng số liệu.
    """
    cm = confusion_matrix(y_true, y_pred).astype(float)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    cls = list(le.classes_)

    pairs = []
    for i in range(len(cls)):
        for j in range(len(cls)):
            if i != j:
                pairs.append((cls[i], cls[j], cm_pct[i, j]))
    pairs.sort(key=lambda x: -x[2])

    print(f"\n  [{cfg}] 5 cặp nhầm lẫn nặng nhất:")
    for a, b, v in pairs[:5]:
        print(f"      {a:10s} → bị đoán thành {b:10s}: {v:5.1f}%")

    res = {"config": cfg}
    if "neutral" in cls and "calm" in cls:
        i_n, i_c = cls.index("neutral"), cls.index("calm")
        n2c, c2n = cm_pct[i_n, i_c], cm_pct[i_c, i_n]
        res["neutral→calm"] = round(float(n2c), 2)
        res["calm→neutral"] = round(float(c2n), 2)
        print(f"      ── Cặp kinh điển: neutral↔calm = "
              f"{n2c:.1f}% / {c2n:.1f}%")
    return res


def plot_per_emotion(df_emo: pd.DataFrame) -> None:
    """So sánh F1 từng cảm xúc giữa các cấu hình → thấy GRL/Aug giúp lớp nào."""
    piv = df_emo.pivot_table(index="Emotion", columns="Config", values="F1(%)")
    piv = piv.reindex([e for e in EMO_ORDER if e in piv.index])

    fig, ax = plt.subplots(figsize=(13, 6))
    piv.plot(kind="bar", ax=ax, edgecolor="grey", width=0.8)
    ax.set_ylabel("F1 (%)")
    ax.set_xlabel("Cảm xúc")
    ax.set_title("F1 theo từng cảm xúc – so sánh các cấu hình (LOSGO 6-fold)",
                 fontweight="bold")
    ax.legend(title="Cấu hình", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(out("per_emotion_f1.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Per-emotion F1 → {out('per_emotion_f1.png')}")


def plot_ablation(summ: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    order = [c for c in CONFIGS if c in summ.index]
    accs = summ.loc[order, "Acc"].values
    errs = np.nan_to_num(summ.loc[order, "Std"].values, nan=0.0)
    colors = ["#B0B0B0", "#8FA8C8", "#D4A574", "#2E7D5B"][:len(order)]

    ax.bar(order, accs, 0.6, yerr=errs, capsize=5, color=colors, edgecolor="grey")
    base = summ.loc["base", "Acc"] if "base" in summ.index else None
    for i, a in enumerate(accs):
        txt = f"{a:.1f}"
        if base and order[i] != "base":
            txt += f"\n({a-base:+.1f})"
        ax.text(i, a + 0.8, txt, ha="center", fontweight="bold", fontsize=9)
    if base:
        ax.axhline(base, color="red", linestyle=":", label=f"base ({base:.1f}%)")
        ax.legend()
    ax.set_ylabel("Accuracy (%) – LOSGO 6-fold")
    ax.set_title("ABLATION STUDY: đóng góp của Augmentation và "
                 "Speaker-Adversarial (GRL)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out("ablation_study.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Ablation → {out('ablation_study.png')}")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS),
                    choices=list(CONFIGS))
    ap.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    epochs = 10 if args.quick else args.epochs
    n_grp = 2 if args.quick else 6

    print("\n" + "=" * 74)
    print("  CẢI TIẾN #3: SPEAKER-ADVERSARIAL + AUGMENTATION")
    print(f"  Device: {DEVICE}  |  LOSGO {n_grp} fold  |  epochs={epochs}")
    print(f"  Cấu hình: {', '.join(args.configs)}")
    print(f"  Output → {OUT_DIR.absolute()}")
    print("=" * 74)

    waves, labels, actors = load_ravdess_16k("./RAVDESS")
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_cls = len(le.classes_)
    actors = np.array(actors)

    splits = losgo_splits(actors)[:n_grp]
    rows = []
    preds = {c: {"true": [], "pred": []} for c in args.configs}

    for name, tr, te, g in splits:
        print(f"\n{'─'*68}\n  {name}  |  VRAM: {gpu_mem()}\n{'─'*68}")

        for cfg_name in args.configs:
            ck = OUT_DIR / f"_adv_{cfg_name}_{name.split()[0]}.json"
            if ck.exists() and not args.force:
                s = json.loads(ck.read_text())
                print(f"  [RESUME] {cfg_name} @ {name} → bỏ qua "
                      f"({s['row']['Accuracy(%)']:.2f}%)")
                rows.append(s["row"])
                preds[cfg_name]["true"].extend(s["y_true"])
                preds[cfg_name]["pred"].extend(s["y_pred"])
                continue

            r = train_config(waves, y, actors, tr, te, n_cls,
                             CONFIGS[cfg_name], epochs, name, cfg_name)

            row = {
                "Config":       cfg_name,
                "Fold":         name,
                "Accuracy(%)":  round(r["Accuracy(%)"], 2),
                "F1_macro(%)":  round(r["F1_macro(%)"], 2),
                "Precision(%)": round(r["Precision(%)"], 2),
                "Recall(%)":    round(r["Recall(%)"], 2),
                "Time(s)":      round(r["Time(s)"], 1),
            }
            for k in ("spk_probe(%)", "spk_chance(%)",
                      "adv_head_first", "adv_head_last"):
                if k in r:
                    row[k] = round(r[k], 2)

            print(f"  [{cfg_name:8s}] Acc={r['Accuracy(%)']:.2f}%  "
                  f"F1={r['F1_macro(%)']:.2f}%  ({r['Time(s)']:.0f}s)")

            rows.append(row)
            preds[cfg_name]["true"].extend(r["y_true"].tolist())
            preds[cfg_name]["pred"].extend(r["y_pred"].tolist())

            OUT_DIR.mkdir(parents=True, exist_ok=True)
            ck.write_text(json.dumps({
                "row": row,
                "y_true": r["y_true"].tolist(),
                "y_pred": r["y_pred"].tolist(),
            }))
            pd.DataFrame(rows).to_csv(out("results_partial.csv"), index=False)

    # ── Tổng hợp ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(out("results_adversarial.csv"), index=False)

    summ = df.groupby("Config").agg(
        Acc=("Accuracy(%)", "mean"), Std=("Accuracy(%)", "std"),
        F1=("F1_macro(%)", "mean")).round(2)
    summ = summ.reindex([c for c in CONFIGS if c in summ.index])

    print("\n" + "=" * 74)
    print("  ABLATION STUDY – LOSGO 6-fold")
    print("=" * 74)
    print(f"  {'Cấu hình':10s} {'Aug':>4s} {'GRL':>4s} {'Accuracy':>16s} {'F1':>8s} {'Δ base':>8s}")
    print("  " + "─" * 62)
    base = summ.loc["base", "Acc"] if "base" in summ.index else None
    for c in summ.index:
        a, s, f = summ.loc[c, "Acc"], summ.loc[c, "Std"], summ.loc[c, "F1"]
        d = f"{a-base:+.2f}%" if base and c != "base" else "—"
        print(f"  {c:10s} {'✓' if CONFIGS[c]['aug'] else '✗':>4s} "
              f"{'✓' if CONFIGS[c]['grl'] else '✗':>4s} "
              f"{a:9.2f}±{s:<5.2f} {f:7.2f}% {d:>8s}")

    print(f"\n  Đối chiếu: HuBERT fine-tuned (B0, không aug/grl) = 72.98%")

    # ── GRL CÓ THẬT SỰ XOÁ ĐƯỢC DANH TÍNH KHÔNG? (đo bằng probe) ─────────────
    if "spk_probe(%)" in df.columns and df["spk_probe(%)"].notna().any():
        print("\n" + "=" * 74)
        print("  GRL CÓ THẬT SỰ XOÁ THÔNG TIN DANH TÍNH KHÔNG?")
        print("=" * 74)
        print("  Đo bằng LINEAR PROBE MỚI trên embedding ĐÓNG BĂNG.")
        print("  (KHÔNG dùng accuracy của head đối kháng — head đó bị GRL đánh bại")
        print("   nên đương nhiên tụt, điều đó không chứng minh gì cả.)\n")

        pr = df.groupby("Config")[["spk_probe(%)", "spk_chance(%)"]].mean().round(2)
        base_probe = pr.loc["base", "spk_probe(%)"] if "base" in pr.index else None
        chance = pr["spk_chance(%)"].mean()

        print(f"  {'Cấu hình':10s} {'speaker-probe':>15s} {'Δ vs base':>11s}")
        print("  " + "─" * 40)
        for c in pr.index:
            v = pr.loc[c, "spk_probe(%)"]
            d = f"{v - base_probe:+.2f}%" if base_probe is not None and c != "base" else "—"
            print(f"  {c:10s} {v:14.2f}% {d:>11s}")
        print(f"  {'(ngẫu nhiên)':10s} {chance:14.2f}%")

        if base_probe is not None and "grl" in pr.index:
            drop = base_probe - pr.loc["grl", "spk_probe(%)"]
            print()
            if drop > 10:
                print(f"  → GRL LÀM GIẢM probe {drop:.1f} điểm → thật sự xoá bớt")
                print("    thông tin danh tính khỏi embedding. Cơ chế hoạt động.")
            else:
                print(f"  → GRL chỉ giảm probe {drop:.1f} điểm → thông tin danh tính")
                print("    VẪN CÒN trong embedding. GRL làm head đối kháng mù đi,")
                print("    nhưng KHÔNG xoá được thông tin. Đây là hạn chế đã biết")
                print("    của DANN/GRL — phải báo cáo trung thực.")
            if "adv_head_last" in df.columns and df["adv_head_last"].notna().any():
                ah = df["adv_head_last"].mean()
                print(f"\n  (Head đối kháng cuối train chỉ đạt {ah:.1f}% — nhưng như trên,")
                print("   con số này KHÔNG phải bằng chứng.)")

    # ── Per-emotion + confusion ──────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("  PHÂN TÍCH PER-EMOTION")
    print("=" * 74)

    emo_rows, conf_rows = [], []
    for cfg in args.configs:
        yt = np.array(preds[cfg]["true"])
        yp = np.array(preds[cfg]["pred"])
        if len(yt) == 0:
            continue
        acc = accuracy_score(yt, yp) * 100

        tbl = per_emotion_table(yt, yp, le, cfg)
        emo_rows.append(tbl)
        plot_confusion(yt, yp, le, cfg, acc)
        conf_rows.append(analyze_confusions(yt, yp, le, cfg))

        print(f"\n  [{cfg}] F1 theo cảm xúc:")
        t = tbl.sort_values("F1(%)")
        for _, r_ in t.iterrows():
            bar = "█" * int(r_["F1(%)"] / 4)
            print(f"      {r_['Emotion']:10s} {r_['F1(%)']:5.1f}%  {bar}")
        print(f"      → Khó nhất: {t.iloc[0]['Emotion']} ({t.iloc[0]['F1(%)']:.1f}%)"
              f"  |  Dễ nhất: {t.iloc[-1]['Emotion']} ({t.iloc[-1]['F1(%)']:.1f}%)")

    if emo_rows:
        df_emo = pd.concat(emo_rows)
        df_emo.to_csv(out("per_emotion_results.csv"), index=False)
        plot_per_emotion(df_emo)
        pd.DataFrame(conf_rows).to_csv(out("confusion_pairs.csv"), index=False)

    # ── Kiểm định thống kê ───────────────────────────────────────────────────
    if df["Fold"].nunique() >= 3 and len(args.configs) > 1:
        print("\n" + "=" * 74)
        print(f"  KIỂM ĐỊNH THỐNG KÊ (paired, n={df['Fold'].nunique()} fold)")
        print("=" * 74)
        try:
            d2 = df.rename(columns={"Config": "Model"})
            st = paired_tests(d2)
            if not st.empty:
                print(st.to_string(index=False))
                st.to_csv(out("statistical_tests_adversarial.csv"), index=False)
                print("\n  [LƯU Ý] n=6 → Wilcoxon p sàn = 0.0312. Dùng t-test + Cohen's d.")
        except ImportError:
            print("  [WARN] Cần scipy")

    plot_ablation(summ)

    for f in OUT_DIR.glob("_adv_*.json"):
        f.unlink()
    p = Path(out("results_partial.csv"))
    if p.exists():
        p.unlink()

    print("\n" + "=" * 74)
    print("  HOÀN THÀNH – Output:")
    print(f"  • {out('results_adversarial.csv')}          (ablation)")
    print(f"  • {out('per_emotion_results.csv')}       (per-emotion)")
    print(f"  • {out('confusion_pairs.csv')}          (cặp nhầm lẫn)")
    print(f"  • {out('ablation_study.png')}")
    print(f"  • {out('per_emotion_f1.png')}")
    print(f"  • confusion_<config>.png")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()
