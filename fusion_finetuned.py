"""
=============================================================================
CẢI TIẾN #2 (v2): FUSION TRÊN EMBEDDING FINE-TUNED — BẢN ĐÃ SỬA 5 LỖI
Đề tài: Hệ thống gợi ý sản phẩm dựa trên phân tích giọng nói và cảm xúc
Học viên: Nguyễn Tấn Nhu | GVHD: TS. Bùi Thanh Hùng
=============================================================================

NĂM LỖI CỦA BẢN v1 VÀ CÁCH SỬA

  #1 SO SÁNH THIÊN VỊ F0
     v1: F0 (model C) là output TRỰC TIẾP của backbone được early-stop tối ưu
         trên val, còn F1-F3 phải học head MỚI trên embedding đóng băng của
         chính model đó. F0 có lợi thế bất công → nếu fusion thua, KHÔNG thể
         kết luận "fusion vô dụng" (có thể chỉ vì fusion bị đặt thế bất lợi).
     v2: THÊM control B0 = "FT-emb only + MLP head", huấn luyện bằng ĐÚNG quy
         trình của các model fusion (cùng optimizer, cùng epochs, cùng patience,
         cùng head).
         → So sánh F1/F2/F3 vs B0 mới cô lập đúng tác dụng của việc THÊM nhánh
           thủ công. F0 vẫn báo cáo, nhưng chỉ để đối chiếu với 73.54%.

  #2 MFCC LÀ NHÁNH QUÁ YẾU → KẾT LUẬN SAI PHẠM VI
     v1: chỉ ghép MFCC. MFCC ở LOSGO rất yếu, còn FT-emb đạt 73.5% → AttGate
         gần như chắc chắn học cách bỏ qua MFCC → ta sẽ kết luận "fusion vô
         dụng". Nhưng kết luận ĐÚNG chỉ là "fusion VỚI MFCC vô dụng".
         Hai mệnh đề rất khác nhau — phản biện sẽ chỉ ra ngay.
     v2: THÊM NHÁNH PROSODY (F0 contour, jitter, shimmer, energy dynamics).
         Đây mới là thông tin HuBERT có khả năng KHÔNG có: HuBERT pretrain cho
         ASR (nội dung ngôn ngữ) nên bị huấn luyện để BỎ QUA prosody, trong khi
         prosody chính là tín hiệu cảm xúc hàng đầu. Nếu fusion có cửa thắng,
         cửa đó nằm ở đây.

  #3 THIẾU BASELINE ĐƠN NHÁNH
     v1: không có MFCC-only / prosody-only → không diễn giải nổi gate weights.
         (Nếu MFCC-only=40% mà gate cho nó 0.3 thì đó là HỢP LÝ, không phải
         "bỏ qua".)
     v2: thêm B1 (MFCC-only) và B2 (Prosody-only).

  #4 NGÂN SÁCH TỐI ƯU HOÁ KHÔNG ĐỒNG NHẤT
     v1: fusion được PATIENCE*2, backbone chỉ PATIENCE.
     v2: MỌI model dùng chung FUSION_PATIENCE. Ghi rõ trong log.

  #5 CROSS-ATTENTION GIẢ
     v1: nn.MultiheadAttention với seq_len=1 → softmax trên 1 phần tử = 1.0
         → thoái hoá thành linear projection + residual. KHÔNG phải attention.
     v2: cross-attention THẬT trên trục thời gian: query = đặc trưng thủ công,
         key/value = chuỗi frame-level của HuBERT (T≈149 frames).
         Giờ attention mới thực sự "chọn khung thời gian nào đáng nghe".

⚠️  CHỐNG LEAKAGE (giữ nguyên từ v1 – vẫn là điều quan trọng nhất)
  KHÔNG dùng embeddings_hubert_finetuned.npy có sẵn: file đó sinh từ model đã
  train trên hầu hết actor, kể cả actor trong test fold → leakage.
  Mỗi fold: fine-tune HuBERT TỪ ĐẦU trên train fold → trích embedding bằng
  chính model đó → embedding của test fold luôn "sạch".

BẢY MÔ HÌNH ĐƯỢC SO SÁNH (tất cả dưới LOSGO 6-fold)

  Đối chứng end-to-end:
    F0. Model C (fine-tuned, end-to-end)      ← đối chiếu với 73.54%
  Baseline đơn nhánh (cùng quy trình head):
    B0. FT-emb only          ← CONTROL CHÍNH, so mọi fusion với cái này
    B1. MFCC only
    B2. Prosody only
  Fusion:
    F1. Concat (MFCC + Prosody + FT) + MLP
    F2. AttGate 3 nhánh
    F3. Cross-Attention THẬT trên trục thời gian

CÂU HỎI: fusion (F1/F2/F3) có vượt B0 không?
  CÓ    → nhánh thủ công bổ sung thông tin HuBERT thiếu → mô hình đề xuất.
  KHÔNG → backbone fine-tuned đã học đủ; kiến trúc ĐƠN GIẢN HƠN mà mạnh hơn.
  Cả hai đều là kết luận có giá trị.

CÁCH CHẠY
  python fusion_finetuned.py --quick    # 2 fold, 10 epochs (~20 phút)
  python fusion_finetuned.py            # 6 fold (~60 phút GPU)
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
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

warnings.filterwarnings("ignore")

from hubert_finetune import (
    HubertSER, RavdessDataset, load_ravdess_16k, set_seed, free_gpu, gpu_mem,
    _make_optimizer, _evaluate,
    RANDOM_SEED, BATCH_SIZE, GRAD_ACCUM, LR_BACKBONE, LR_HEAD,
    WARMUP_RATIO, LABEL_SMOOTH,
)
from speaker_independent_benchmark import (
    losgo_splits, load_ravdess_22k, extract_mfcc_hc, make_val_split,
    _metrics, paired_tests, SR_HC,
)

OUT_DIR = Path(os.environ.get("OUT_DIR", "./outputs_fusion_ft"))


def out(f: str) -> str:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return str(OUT_DIR / f)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMB_DIM        = 256
PROJ_DIM       = 256
FUSION_EPOCHS  = 80
FUSION_BATCH   = 32
FUSION_LR      = 1e-3
# FIX #4: MỌI model (kể cả baseline đơn nhánh) dùng CHUNG con số này.
FUSION_PATIENCE = 12
BACKBONE_PATIENCE = 6

# v3: entropy bonus giữ các nhánh gate "sống" đủ lâu để thực sự cạnh tranh.
#
# ⚠️ CƯỜNG ĐỘ ĐÃ ĐƯỢC HIỆU CHUẨN BẰNG THỰC NGHIỆM (không đoán):
#   quét reg ∈ {0, 0.05, 0.2, 0.5, 1.0, 2.0} trên dữ liệu tổng hợp
#   (emb rất mạnh, mfcc/prosody = nhiễu thuần) cho kết quả:
#       reg=0.00 → gate sập @epoch 2   (nhánh chết yểu)
#       reg=0.05 → gate sập @epoch 2   ← QUÁ YẾU, vô tác dụng!
#       reg=0.20 → gate sập @epoch 2 rồi hồi phục
#       reg=0.50 → KHÔNG sập, hand≈0.65  ✅
#       reg=1.00 → KHÔNG sập, hand≈0.66  ✅
#   → entropy bonus phải >= 0.5 mới thắng nổi gradient của cross-entropy.
#     (0.05 như dự định ban đầu hoàn toàn vô dụng.)
GATE_REG_DEFAULT = 0.5

LOSGO_BASELINE = {
    "A. HuBERT frozen + SVM":  62.43,
    "B. AttGate (frozen emb)": 64.72,
    "C. HuBERT fine-tuned":    73.54,
}


# ============================================================================
# PHẦN 1: ĐẶC TRƯNG PROSODY (FIX #2)
# ============================================================================

def extract_prosody(waves22k: np.ndarray) -> np.ndarray:
    """
    FIX #2: nhánh thủ công phải mang thông tin HuBERT KHÔNG CÓ.

    HuBERT pretrain cho ASR → được huấn luyện để bỏ qua prosody (cùng một câu
    nói vui hay buồn đều phải ra cùng transcript). Nhưng prosody CHÍNH LÀ tín
    hiệu cảm xúc hàng đầu. Đây là chỗ fusion có cửa thắng thật sự — khác với
    MFCC (vốn trùng lặp nhiều với cái HuBERT đã mã hoá).

    Đặc trưng (52 chiều):
      • F0 contour: mean/std/min/max/range/slope     (cao độ – vui cao, buồn thấp)
      • Jitter: biến thiên chu kỳ F0                  (giọng run → sợ hãi, lo âu)
      • Shimmer: biến thiên biên độ                   (giọng rung → xúc động)
      • Energy (RMS): mean/std/max + delta            (giận dữ = năng lượng cao)
      • Voiced ratio, pause ratio                     (nhịp nói – buồn nói chậm)
      • Spectral flux                                 (tốc độ biến đổi phổ)
    """
    print("  [Prosody] Trích F0/jitter/shimmer/energy...")
    feats = []

    for i, y in enumerate(waves22k):
        if i % 300 == 0:
            print(f"      {i}/{len(waves22k)}")

        # ── F0 contour (librosa.yin – nhanh hơn pyin nhiều lần) ─────────────
        try:
            f0 = librosa.yin(y, fmin=65, fmax=400, sr=SR_HC,
                             frame_length=1024, hop_length=256)
            f0 = np.nan_to_num(f0, nan=0.0)
            voiced = f0[(f0 > 65) & (f0 < 400)]
        except Exception:
            f0, voiced = np.zeros(10), np.array([])

        if len(voiced) > 2:
            f0_mean, f0_std = float(voiced.mean()), float(voiced.std())
            f0_min, f0_max = float(voiced.min()), float(voiced.max())
            f0_rng = f0_max - f0_min
            # Slope: xu hướng lên/xuống của cao độ (câu hỏi vs câu khẳng định)
            f0_slope = float(np.polyfit(np.arange(len(voiced)), voiced, 1)[0])
            # Jitter: |ΔF0| trung bình chuẩn hoá — giọng run
            jitter = float(np.mean(np.abs(np.diff(voiced))) / (f0_mean + 1e-8))
            voiced_ratio = float(len(voiced) / max(1, len(f0)))
        else:
            f0_mean = f0_std = f0_min = f0_max = f0_rng = 0.0
            f0_slope = jitter = voiced_ratio = 0.0

        # ── Energy / RMS ────────────────────────────────────────────────────
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
        rms_mean = float(rms.mean())
        # Shimmer: biến thiên biên độ giữa các frame liên tiếp
        shimmer = float(np.mean(np.abs(np.diff(rms))) / (rms_mean + 1e-8))
        # Pause ratio: tỷ lệ frame gần như im lặng (nhịp nói)
        pause_ratio = float(np.mean(rms < 0.1 * rms_mean)) if rms_mean > 0 else 0.0

        d_rms = np.diff(rms) if len(rms) > 1 else np.zeros(1)

        # ── Spectral flux: tốc độ biến đổi phổ ──────────────────────────────
        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
        flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0)) if S.shape[1] > 1 \
            else np.zeros(1)

        # ── Đường bao F0 & RMS rút gọn (8 điểm mỗi cái – giữ hình dạng) ──────
        def envelope(sig, n=8):
            if len(sig) < n:
                sig = np.pad(sig, (0, n - len(sig)))
            return np.array([float(c.mean()) for c in np.array_split(sig, n)])

        feats.append(np.concatenate([
            [f0_mean, f0_std, f0_min, f0_max, f0_rng, f0_slope,
             jitter, shimmer, voiced_ratio, pause_ratio],           # 10
            [rms_mean, float(rms.std()), float(rms.max()),
             float(np.abs(d_rms).mean()), float(d_rms.std())],      # 5
            [float(flux.mean()), float(flux.std()), float(flux.max())],  # 3
            envelope(voiced if len(voiced) > 0 else np.zeros(8)),   # 8  F0 contour
            envelope(rms),                                          # 8  energy contour
        ]))

    X = np.array(feats)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  [Prosody] shape={X.shape}")
    return X


# ============================================================================
# PHẦN 2: FINE-TUNE + TRÍCH EMBEDDING (pooled + frame-level)
# ============================================================================

@torch.no_grad()
def _extract_embeddings(model: HubertSER, waves: np.ndarray,
                        want_sequence: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Trích embedding từ model đã fine-tune.

    pooled   : (N, 256)          – cho MLP / AttGate
    sequence : (N, T≈149, 256)   – cho CROSS-ATTENTION THẬT (FIX #5)
                                   lưu fp16 để tiết kiệm RAM (~110MB)
    """
    model.eval()
    kw = dict(pin_memory=(DEVICE.type == "cuda"),
              num_workers=2 if DEVICE.type == "cuda" else 0)
    dl = DataLoader(RavdessDataset(waves, np.zeros(len(waves), dtype=np.int64)),
                    batch_size=BATCH_SIZE, **kw)

    pooled, seqs = [], []
    for xb, _ in dl:
        xb = xb.to(DEVICE)
        with torch.autocast(device_type=DEVICE.type,
                            enabled=(DEVICE.type == "cuda")):
            o = model.hubert(xb)
            if model.use_weighted_layers:
                w = torch.softmax(model.layer_weights, dim=0)
                h = o.hidden_states[0] * w[0]
                for i in range(1, len(o.hidden_states)):
                    h = h + o.hidden_states[i] * w[i]      # (B, T, 768)
            else:
                h = o.last_hidden_state

            # embed_fc là Linear+ReLU+Dropout → áp được trên (B, T, H)
            seq = model.embed_fc(h)                        # (B, T, 256)
            pool = model.pool(h)                           # (B, 768)
            pool = model.embed_fc(pool)                    # (B, 256)

        pooled.append(pool.float().cpu().numpy())
        if want_sequence:
            seqs.append(seq.half().cpu().numpy())          # fp16 tiết kiệm RAM

    P = np.concatenate(pooled)
    S = np.concatenate(seqs) if want_sequence else None
    return P, S


def finetune_and_extract(waves, y, tr, te, actors, n_cls, epochs, fold):
    """
    Fine-tune HuBERT trên train fold → trích embedding bằng CHÍNH model đó.
    Chống leakage: model chưa từng thấy actor trong test fold.
    """
    set_seed(RANDOM_SEED)
    t0 = time.time()

    real_tr, real_val = make_val_split(tr, y, actors, speaker_disjoint=True)

    kw = dict(pin_memory=(DEVICE.type == "cuda"),
              num_workers=2 if DEVICE.type == "cuda" else 0)
    tr_dl = DataLoader(RavdessDataset(waves[real_tr], y[real_tr]),
                       batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(RavdessDataset(waves[real_val], y[real_val]),
                        batch_size=BATCH_SIZE, **kw)
    te_dl = DataLoader(RavdessDataset(waves[te], y[te]),
                       batch_size=BATCH_SIZE, **kw)

    model = HubertSER(n_cls).to(DEVICE)
    opt = _make_optimizer(model)
    lossfn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    steps = max(1, (len(tr_dl) // GRAD_ACCUM) * epochs)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[LR_BACKBONE, LR_HEAD],
        total_steps=steps, pct_start=WARMUP_RATIO)

    best_f1, best_state, bad = 0.0, None, 0
    print(f"    [{fold}] Fine-tune HuBERT ({epochs} epochs, "
          f"patience={BACKBONE_PATIENCE})...")

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        for i, (xb, yb) in enumerate(tr_dl):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            with torch.autocast(device_type=DEVICE.type,
                                enabled=(DEVICE.type == "cuda")):
                loss = lossfn(model(xb), yb) / GRAD_ACCUM
            scaler.scale(loss).backward()
            if (i + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad()
                if sch.last_epoch < steps - 1:
                    sch.step()

        vp, vt = _evaluate(model, val_dl)
        vf1 = f1_score(vt, vp, average="macro", zero_division=0)
        if vf1 > best_f1:
            best_f1, bad = vf1, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= BACKBONE_PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)

    # F0: model C end-to-end (đối chiếu 73.54%)
    p, t = _evaluate(model, te_dl)
    m_c = _metrics(t, p, time.time() - t0)

    E_pool, E_seq = _extract_embeddings(model, waves, want_sequence=True)
    print(f"    [{fold}] F0 end-to-end acc={m_c['Accuracy(%)']:.2f}%  |  "
          f"pooled{E_pool.shape} seq{E_seq.shape} ({time.time()-t0:.0f}s)")

    free_gpu(model)
    return E_pool, E_seq, m_c


# ============================================================================
# PHẦN 3: CÁC KIẾN TRÚC
# ============================================================================

class SingleBranchMLP(nn.Module):
    """
    Baseline đơn nhánh — B0 (FT-emb), B1 (MFCC), B2 (Prosody).

    ⚠️ FIX v3 — KIẾN TRÚC PHẢI TRÙNG KHỚP F2:
      Bản v2 dùng head riêng (256→128→n_cls với BatchNorm khác chỗ), trong khi
      F2 khi gate=(0,0,1) thoái hoá thành: branch_emb(x) → clf(x).
      Hai head KHÁC kiến trúc → chênh lệch F2 vs B0 lẫn cả "nhiễu kiến trúc",
      không thuần tuý là tác dụng của fusion.

      Kết quả --quick đã phơi bày đúng lỗi này:
          F2 = 79.59%,  B0 = 77.09%,  Δ = +2.50%
          NHƯNG gate = (0.000, 0.000, 1.000)
      Gate bảo MFCC/prosody đóng góp CHÍNH XÁC BẰNG KHÔNG, mà F2 vẫn hơn B0
      2.5%?! Không thể cùng đúng. +2.5% đó đến từ KHÁC BIỆT KIẾN TRÚC HEAD.

      v3: B0 = ĐÚNG đường đi của F2 khi gate=(0,0,1):
              Linear(d,proj)+BN+ReLU+Drop → Linear(proj,128)+ReLU+Drop → Linear(128,C)
      → Giờ mọi chênh lệch F2 − B0 mới THUẦN TUÝ là tác dụng của fusion.
    """

    def __init__(self, d_in: int, n_cls: int, proj: int = PROJ_DIM):
        super().__init__()
        # Trùng khít branch() của AttGate3
        self.branch = nn.Sequential(
            nn.Linear(d_in, proj), nn.BatchNorm1d(proj),
            nn.ReLU(), nn.Dropout(0.3),
        )
        # Trùng khít clf của AttGate3
        self.clf = nn.Sequential(
            nn.Linear(proj, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_cls),
        )

    def forward(self, x, *_):
        return self.clf(self.branch(x)), None


class ConcatMLP(nn.Module):
    """F1. Ghép thẳng [MFCC ; Prosody ; FT-emb] → MLP."""

    def __init__(self, d_hc: int, d_emb: int, n_cls: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_hc + d_emb, 512), nn.BatchNorm1d(512),
            nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_cls),
        )

    def forward(self, x_hc, x_emb, *_):
        return self.net(torch.cat([x_hc, x_emb], dim=1)), None


class AttGate3(nn.Module):
    """
    F2. AttentionGate 3 nhánh: MFCC | Prosody | FT-emb.

    ⚠️ CHẨN ĐOÁN GATE COLLAPSE (v3):
      Kết quả --quick cho gate = (0.000, 0.000, 1.000) — bão hoà TUYỆT ĐỐI.
      Có HAI cách giải thích, ý nghĩa khoa học khác hẳn nhau:

        (a) Model HỌC ĐƯỢC rằng nhánh thủ công vô dụng  → kết luận khoa học thật
        (b) Softmax bão hoà NGAY epoch đầu, nhánh chết trước khi kịp học
            → bệnh lý tối ưu hoá, KHÔNG kết luận được gì

      Phân biệt bằng: theo dõi gate qua từng epoch (gate_history).
        • Nếu gate trôi DẦN về (0,0,1) qua nhiều epoch → (a), đáng tin.
        • Nếu gate nhảy về (0,0,1) trong 1-3 epoch đầu → (b), nghi ngờ.

      Thêm tuỳ chọn --gate-reg: cộng entropy bonus vào loss để GIỮ nhánh sống,
      buộc model phải thực sự chứng minh nhánh thủ công vô dụng thay vì bóp
      chết chúng sớm. Nếu gate VẪN về (0,0,1) dù có entropy bonus → bằng chứng
      (a) rất mạnh.
    """

    def __init__(self, d_mfcc: int, d_pros: int, d_emb: int, n_cls: int,
                 proj: int = PROJ_DIM):
        super().__init__()
        def branch(d):
            return nn.Sequential(nn.Linear(d, proj), nn.BatchNorm1d(proj),
                                 nn.ReLU(), nn.Dropout(0.3))
        self.b_mfcc, self.b_pros, self.b_emb = branch(d_mfcc), branch(d_pros), branch(d_emb)
        self.gate = nn.Sequential(
            nn.Linear(proj * 3, 64), nn.ReLU(),
            nn.Linear(64, 3), nn.Softmax(dim=-1))
        self.clf = nn.Sequential(
            nn.Linear(proj, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_cls))

    def forward(self, x_mfcc, x_pros, x_emb, *_):
        h1, h2, h3 = self.b_mfcc(x_mfcc), self.b_pros(x_pros), self.b_emb(x_emb)
        g = self.gate(torch.cat([h1, h2, h3], dim=1))          # (B, 3)
        fused = g[:, 0:1] * h1 + g[:, 1:2] * h2 + g[:, 2:3] * h3
        return self.clf(fused), g


class CrossAttentionTime(nn.Module):
    """
    F3. CROSS-ATTENTION THẬT (FIX #5).

    Bản v1 dùng seq_len=1 → softmax trên 1 phần tử = 1.0 → thoái hoá thành
    linear projection + residual. KHÔNG phải attention.

    Bản này: query = đặc trưng thủ công (MFCC+prosody, 1 token),
             key/value = CHUỖI FRAME-LEVEL của HuBERT (T≈149 frames).
    → Attention giờ thực sự học "khung thời gian nào đáng nghe", dựa trên
      gợi ý từ prosody. Đây mới đúng tinh thần cross-modal attention.
    """

    def __init__(self, d_hc: int, d_emb: int, n_cls: int,
                 proj: int = PROJ_DIM, heads: int = 4):
        super().__init__()
        self.q_proj = nn.Sequential(
            nn.Linear(d_hc, proj), nn.LayerNorm(proj), nn.ReLU(), nn.Dropout(0.3))
        self.kv_proj = nn.Sequential(
            nn.Linear(d_emb, proj), nn.LayerNorm(proj))
        self.attn = nn.MultiheadAttention(proj, heads, dropout=0.2,
                                          batch_first=True)
        self.norm = nn.LayerNorm(proj)
        self.clf = nn.Sequential(
            nn.Linear(proj * 2, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_cls))

    def forward(self, x_hc, x_emb_pooled, x_seq):
        # x_seq: (B, T, 256) – chuỗi frame-level, ĐÂY mới là điểm khác biệt
        q = self.q_proj(x_hc).unsqueeze(1)          # (B, 1, proj)
        kv = self.kv_proj(x_seq)                    # (B, T, proj)
        a, w = self.attn(q, kv, kv)                 # attend qua T frames
        a = self.norm(a.squeeze(1))                 # (B, proj)
        pooled = kv.mean(dim=1)                     # (B, proj)
        return self.clf(torch.cat([a, pooled], dim=1)), None


# ============================================================================
# PHẦN 4: HUẤN LUYỆN (FIX #4 – ngân sách ĐỒNG NHẤT cho mọi model)
# ============================================================================

def train_head(kind: str, Xs: dict, y, tr, te, actors, n_cls, seq=None,
               gate_reg: float = 0.0) -> dict:
    """
    Huấn luyện 1 model (baseline hoặc fusion) trên 1 fold.

    FIX #4: MỌI model dùng chung FUSION_EPOCHS / FUSION_PATIENCE / optimizer.
    Không model nào được ưu ái ngân sách tối ưu hoá hơn model khác.
    """
    set_seed(RANDOM_SEED)
    t0 = time.time()
    real_tr, real_val = make_val_split(tr, y, actors, speaker_disjoint=True)

    # Chuẩn hoá – scaler CHỈ fit trên train
    scaled = {}
    for k, X in Xs.items():
        sc = StandardScaler()
        scaled[k] = {
            "tr": sc.fit_transform(X[real_tr]).astype(np.float32),
            "va": sc.transform(X[real_val]).astype(np.float32),
            "te": sc.transform(X[te]).astype(np.float32),
        }

    def T(a):
        return torch.tensor(a).to(DEVICE)

    # ── Dựng model + xác định input ─────────────────────────────────────────
    d = {k: v["tr"].shape[1] for k, v in scaled.items()}

    if kind == "B0":
        model = SingleBranchMLP(d["emb"], n_cls)
        keys = ["emb"]
    elif kind == "B1":
        model = SingleBranchMLP(d["mfcc"], n_cls)
        keys = ["mfcc"]
    elif kind == "B2":
        model = SingleBranchMLP(d["pros"], n_cls)
        keys = ["pros"]
    elif kind == "F1":
        model = ConcatMLP(d["mfcc"] + d["pros"], d["emb"], n_cls)
        keys = ["hc", "emb"]
    elif kind == "F2":
        model = AttGate3(d["mfcc"], d["pros"], d["emb"], n_cls)
        keys = ["mfcc", "pros", "emb"]
    elif kind == "F3":
        model = CrossAttentionTime(d["mfcc"] + d["pros"], EMB_DIM, n_cls)
        keys = ["hc", "emb", "seq"]
    else:
        raise ValueError(kind)

    model = model.to(DEVICE)

    def get(split: str) -> list:
        xs = []
        for k in keys:
            if k == "hc":
                xs.append(T(np.concatenate(
                    [scaled["mfcc"][split], scaled["pros"][split]], axis=1)))
            elif k == "seq":
                idx = {"tr": real_tr, "va": real_val, "te": te}[split]
                xs.append(torch.tensor(seq[idx]).float().to(DEVICE))
            else:
                xs.append(T(scaled[k][split]))
        return xs

    opt = torch.optim.AdamW(model.parameters(), lr=FUSION_LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FUSION_EPOCHS)
    lossfn = nn.CrossEntropyLoss(label_smoothing=0.05)

    tr_in = get("tr")
    y_tr = torch.tensor(y[real_tr], dtype=torch.long).to(DEVICE)
    n = len(real_tr)

    best_f1, best_state, bad = 0.0, None, 0
    gate_history = []          # v3: theo dõi gate qua TỪNG epoch

    for ep in range(FUSION_EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        ep_gates = []
        for i in range(0, n - FUSION_BATCH + 1, FUSION_BATCH):
            b = perm[i:i + FUSION_BATCH]
            opt.zero_grad()
            args = [x[b] for x in tr_in]
            while len(args) < 3:
                args.append(None)
            logits, g = model(*args)
            loss = lossfn(logits, y_tr[b])

            # v3: entropy bonus – giữ nhánh sống, chống gate collapse sớm
            if g is not None and gate_reg > 0:
                ent = -(g * torch.log(g + 1e-9)).sum(1).mean()
                loss = loss - gate_reg * ent     # tối đa hoá entropy

            loss.backward()
            opt.step()
            if g is not None:
                ep_gates.append(g.detach().mean(0).cpu().numpy())
        sch.step()

        if ep_gates:
            gate_history.append(np.mean(ep_gates, axis=0).tolist())

        model.eval()
        with torch.no_grad():
            args = get("va")
            while len(args) < 3:
                args.append(None)
            vp = model(*args)[0].argmax(1).cpu().numpy()
        vf1 = f1_score(y[real_val], vp, average="macro", zero_division=0)

        if vf1 > best_f1:
            best_f1, bad = vf1, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= FUSION_PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        args = get("te")
        while len(args) < 3:
            args.append(None)
        logits, g = model(*args)
        p = logits.argmax(1).cpu().numpy()

    r = _metrics(y[te], p, time.time() - t0)
    if g is not None and g.shape[1] == 3:
        gm = g.mean(0).cpu().numpy()
        r["gate_mfcc"], r["gate_pros"], r["gate_emb"] = map(float, gm)

        # ── v3: CHẨN ĐOÁN GATE COLLAPSE ─────────────────────────────────────
        # Gate về (0,0,1) sau BAO NHIÊU epoch? Nhanh = nghi ngờ chết yểu.
        if gate_history:
            gh = np.array(gate_history)              # (n_epoch, 3)
            hand = gh[:, 0] + gh[:, 1]               # tổng trọng số nhánh thủ công
            collapsed = np.where(hand < 0.05)[0]
            r["gate_collapse_epoch"] = int(collapsed[0]) + 1 if len(collapsed) else -1
            r["gate_hand_ep1"] = float(hand[0])      # nhánh thủ công ở epoch 1
            r["gate_history"] = gh.tolist()

    free_gpu(model)
    return r


MODEL_NAMES = {
    "B0": "B0. FT-emb only (CONTROL)",
    "B1": "B1. MFCC only",
    "B2": "B2. Prosody only",
    "F1": "F1. Concat + MLP",
    "F2": "F2. AttGate 3-branch",
    "F3": "F3. Cross-Attn (time)",
}


# ============================================================================
# PHẦN 5: VISUALIZATION
# ============================================================================

def plot_results(df: pd.DataFrame) -> None:
    summ = df.groupby("Model").agg(
        acc=("Accuracy(%)", "mean"), std=("Accuracy(%)", "std")).round(2)
    order = (["F0. Model C (end-to-end)"] + list(MODEL_NAMES.values()))
    summ = summ.reindex([o for o in order if o in summ.index])

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    fig.suptitle("Cải tiến #2 (v2): Fusion trên embedding fine-tuned — LOSGO 6-fold",
                 fontsize=13, fontweight="bold")

    names = list(summ.index)
    x = np.arange(len(names))
    accs = summ["acc"].values
    errs = np.nan_to_num(summ["std"].values, nan=0.0)
    colors = ["#8FA8C8" if n.startswith("F0") else
              "#D4A574" if n.startswith("B") else "#2E7D5B" for n in names]

    axes[0].bar(x, accs, 0.6, yerr=errs, capsize=4, color=colors, edgecolor="grey")
    for i, a in enumerate(accs):
        axes[0].text(i, a + 0.7, f"{a:.1f}", ha="center", fontsize=9,
                     fontweight="bold")

    b0 = summ.loc["B0. FT-emb only (CONTROL)", "acc"] \
        if "B0. FT-emb only (CONTROL)" in summ.index else None
    if b0:
        axes[0].axhline(b0, color="red", linestyle=":", linewidth=2,
                        label=f"B0 control ({b0:.1f}%) ← mọi fusion phải vượt cái này")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Fusion có vượt CONTROL B0 không?", fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    for m in summ.index:
        sub = df[df.Model == m]
        axes[1].plot(range(1, len(sub) + 1), sub["Accuracy(%)"], "o-",
                     label=m, linewidth=1.8, markersize=4)
    axes[1].set_xlabel("Speaker group (fold)")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Ổn định qua các nhóm speaker", fontweight="bold")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out("fusion_finetuned_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Biểu đồ → {out('fusion_finetuned_comparison.png')}")


def plot_gate_history(df: pd.DataFrame) -> None:
    """
    v3: Gate trôi DẦN về (0,0,1) hay SẬP ngay epoch đầu?

    Đây là biểu đồ phân biệt hai kịch bản có ý nghĩa khoa học trái ngược:
      • Trôi dần  → model HỌC ĐƯỢC rằng nhánh thủ công vô dụng (kết luận thật)
      • Sập ngay  → nhánh chết yểu do softmax bão hoà (bệnh lý, không kết luận)
    """
    if "gate_history" not in df.columns:
        return
    sub = df[df["gate_history"].notna()]
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    for _, row in sub.iterrows():
        gh = row["gate_history"]
        if isinstance(gh, str):
            gh = json.loads(gh)
        gh = np.array(gh)
        ep = np.arange(1, len(gh) + 1)
        ax.plot(ep, gh[:, 0] + gh[:, 1], alpha=0.7, linewidth=1.8,
                label=f"{row['Fold']}")

    ax.axhline(0.05, color="red", linestyle="--",
               label="Ngưỡng 'triệt tiêu' (0.05)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Tổng trọng số gate cho nhánh THỦ CÔNG (MFCC + Prosody)")
    ax.set_title("Gate sập DẦN (model học được) hay sập NGAY (chết yểu)?",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out("fusion_gate_history.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Gate history → {out('fusion_gate_history.png')}")


def plot_gates(df: pd.DataFrame, summ: pd.DataFrame) -> None:
    """Gate weights ĐỌC CÙNG accuracy đơn nhánh (FIX #3)."""
    sub = df[df.get("gate_mfcc", pd.Series(dtype=float)).notna()] \
        if "gate_mfcc" in df.columns else pd.DataFrame()
    if sub.empty:
        return

    gm, gp, ge = sub["gate_mfcc"].mean(), sub["gate_pros"].mean(), sub["gate_emb"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("AttGate học được gì? (đọc CÙNG accuracy đơn nhánh)",
                 fontweight="bold")

    axes[0].bar(["MFCC", "Prosody", "FT-emb"], [gm, gp, ge],
                color=["#E8A0A0", "#D4A574", "#2E7D5B"], edgecolor="grey")
    axes[0].axhline(1 / 3, color="grey", linestyle="--",
                    label="Cân bằng (0.333)")
    for i, v in enumerate([gm, gp, ge]):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")
    axes[0].set_ylabel("Trọng số cổng TB")
    axes[0].set_title("Trọng số cổng", fontweight="bold")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    accs = []
    for k, lbl in [("B1. MFCC only", "MFCC"), ("B2. Prosody only", "Prosody"),
                   ("B0. FT-emb only (CONTROL)", "FT-emb")]:
        accs.append(summ.loc[k, "Acc"] if k in summ.index else 0)
    axes[1].bar(["MFCC", "Prosody", "FT-emb"], accs,
                color=["#E8A0A0", "#D4A574", "#2E7D5B"], edgecolor="grey")
    for i, v in enumerate(accs):
        axes[1].text(i, v + 0.7, f"{v:.1f}%", ha="center", fontweight="bold")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Năng lực đơn nhánh (để diễn giải gate)", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out("fusion_gate_weights.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Gate weights → {out('fusion_gate_weights.png')}")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--gate-reg", type=float, nargs="?",
                    const=GATE_REG_DEFAULT, default=0.0,
                    help=f"Entropy bonus giữ nhánh gate sống (mặc định "
                         f"{GATE_REG_DEFAULT} nếu bật). Đã hiệu chuẩn thực nghiệm: "
                         f"phải >= 0.5 mới có tác dụng, 0.05 là vô dụng. "
                         f"Nếu gate VẪN bỏ nhánh thủ công dù được giữ sống → "
                         f"bằng chứng RẤT MẠNH rằng chúng thật sự vô dụng.")
    args = ap.parse_args()

    epochs = 10 if args.quick else args.epochs
    n_grp = 2 if args.quick else 6

    print("\n" + "=" * 74)
    print("  CẢI TIẾN #2 (v2): FUSION TRÊN EMBEDDING FINE-TUNED")
    print(f"  Device: {DEVICE}  |  LOSGO {n_grp} fold  |  backbone epochs={epochs}")
    print(f"  Ngân sách ĐỒNG NHẤT: mọi head = {FUSION_EPOCHS} epochs, "
          f"patience={FUSION_PATIENCE}")
    print("  ⚠️  Embedding trích LẠI mỗi fold (chống leakage)")
    print(f"  B0 control: CÙNG kiến trúc head với F2 (loại nhiễu kiến trúc)")
    if args.gate_reg > 0:
        print(f"  Gate entropy bonus = {args.gate_reg} (giữ nhánh sống)")
    print(f"  Output → {OUT_DIR.absolute()}")
    print("=" * 74)

    waves, labels, actors = load_ravdess_16k("./RAVDESS")
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_cls = len(le.classes_)
    actors = np.array(actors)

    print("\n[INFO] Trích đặc trưng thủ công (dùng chung mọi fold)...")
    waves22 = load_ravdess_22k("./RAVDESS")
    X_mfcc = extract_mfcc_hc(waves22)
    X_pros = extract_prosody(waves22)          # FIX #2
    del waves22

    splits = losgo_splits(actors)[:n_grp]
    rows = []

    for name, tr, te, g in splits:
        print(f"\n{'─'*68}\n  {name}  |  VRAM: {gpu_mem()}\n{'─'*68}")

        ck = OUT_DIR / f"_fus_{name.split()[0]}.json"
        if ck.exists() and not args.force:
            print("  [RESUME] đã có → bỏ qua")
            rows.extend(json.loads(ck.read_text()))
            continue

        E_pool, E_seq, m_c = finetune_and_extract(
            waves, y, tr, te, actors, n_cls, epochs, name)

        Xs = {"mfcc": X_mfcc, "pros": X_pros, "emb": E_pool}
        fold_rows = [_row("F0. Model C (end-to-end)", name, m_c)]

        for kind, disp in MODEL_NAMES.items():
            r = train_head(kind, Xs, y, tr, te, actors, n_cls, seq=E_seq,
                           gate_reg=args.gate_reg)
            gate = ""
            if "gate_mfcc" in r:
                ce = r.get("gate_collapse_epoch", -1)
                ce_txt = f" sập@ep{ce}" if ce > 0 else " không sập"
                gate = (f"  [gate: mfcc={r['gate_mfcc']:.2f} "
                        f"pros={r['gate_pros']:.2f} emb={r['gate_emb']:.2f}"
                        f"|{ce_txt}]")
            print(f"  [{disp:26s}] Acc={r['Accuracy(%)']:.2f}%  "
                  f"F1={r['F1_macro(%)']:.2f}%  ({r['Time(s)']:.0f}s){gate}")
            fold_rows.append(_row(disp, name, r))

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        ck.write_text(json.dumps(fold_rows))
        rows.extend(fold_rows)
        pd.DataFrame(rows).to_csv(out("results_partial.csv"), index=False)
        del E_pool, E_seq

    # ── Tổng hợp ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(out("results_fusion_finetuned.csv"), index=False)

    summ = df.groupby("Model").agg(
        Acc=("Accuracy(%)", "mean"), Std=("Accuracy(%)", "std"),
        F1=("F1_macro(%)", "mean")).round(2)

    print("\n" + "=" * 74)
    print("  KẾT QUẢ – LOSGO 6-fold")
    print("=" * 74)
    print(summ.sort_values("Acc", ascending=False).to_string())

    print("\n  Đối chiếu LOSGO đã có:")
    for k, v in LOSGO_BASELINE.items():
        print(f"    {k:26s} {v:6.2f}%")

    # ── TRẢ LỜI CÂU HỎI (so với B0, KHÔNG so với F0) ─────────────────────────
    B0 = "B0. FT-emb only (CONTROL)"
    if B0 in summ.index:
        b0 = summ.loc[B0, "Acc"]
        fus = summ.loc[[m for m in summ.index if m.startswith(("F1", "F2", "F3"))]]
        best = fus["Acc"].idxmax()
        best_acc = fus.loc[best, "Acc"]
        delta = best_acc - b0

        print("\n" + "=" * 74)
        print("  TRẢ LỜI CÂU HỎI NGHIÊN CỨU")
        print("=" * 74)
        print(f"  CONTROL B0 (chỉ FT-emb, CÙNG kiến trúc head)   = {b0:.2f}%")
        print(f"  Fusion tốt nhất: {best:26s} = {best_acc:.2f}%")
        print(f"  Δ = {delta:+.2f}%")

        # ── v3: KIỂM TRA GATE TRƯỚC KHI KẾT LUẬN ────────────────────────────
        # Bản v2 kết luận "fusion CÓ ích" chỉ dựa vào Δ>1.5 → SAI.
        # Kết quả --quick: Δ=+2.50% NHƯNG gate=(0,0,1) — gate nói nhánh thủ công
        # đóng góp BẰNG KHÔNG. Hai điều không thể cùng đúng. Δ đó là nhiễu.
        gate_ok = True
        hand_w = None
        if "gate_emb" in df.columns and df["gate_emb"].notna().any():
            gm, gp, ge = (df["gate_mfcc"].mean(), df["gate_pros"].mean(),
                          df["gate_emb"].mean())
            hand_w = gm + gp
            print(f"\n  Gate TB: MFCC={gm:.3f}  Prosody={gp:.3f}  FT-emb={ge:.3f}")
            print(f"  → Tổng trọng số nhánh thủ công = {hand_w:.3f}")

            if hand_w < 0.05:
                gate_ok = False
                print("\n  ⚠️  GATE ĐÃ TRIỆT TIÊU HOÀN TOÀN nhánh thủ công.")
                print("      Nếu chúng đóng góp ~0 mà fusion vẫn hơn B0, chênh lệch")
                print("      đó KHÔNG đến từ fusion — chỉ là nhiễu huấn luyện.")

            # Chẩn đoán collapse sớm
            if "gate_collapse_epoch" in df.columns:
                ce = df["gate_collapse_epoch"].replace(-1, np.nan).dropna()
                e1 = df["gate_hand_ep1"].dropna()
                if len(ce):
                    print(f"\n  Gate sập ở epoch TB = {ce.mean():.1f}  "
                          f"(nhánh thủ công ở epoch 1 = {e1.mean():.3f})")
                    if ce.mean() <= 3:
                        print("      ⚠️  Sập trong 1-3 epoch đầu → NGHI NGỜ chết yểu")
                        print("      (softmax bão hoà trước khi nhánh kịp học).")
                        print(f"      Chạy lại với --gate-reg (={GATE_REG_DEFAULT}) "
                              "để giữ nhánh sống rồi so.")
                    else:
                        print("      → Sập DẦN qua nhiều epoch = model thực sự HỌC ĐƯỢC")
                        print("        rằng nhánh thủ công vô dụng. Bằng chứng đáng tin.")

        print()
        if delta > 1.5 and gate_ok:
            print("  → KẾT LUẬN: nhánh thủ công CÓ bổ sung thông tin HuBERT thiếu.")
            print("    Cải tiến #1 + #2 cộng hưởng → đây là mô hình đề xuất.")
        elif delta > 1.5 and not gate_ok:
            print("  → KẾT LUẬN: KHÔNG KẾT LUẬN ĐƯỢC fusion có ích.")
            print(f"    Fusion hơn B0 {delta:+.2f}%, NHƯNG gate cho nhánh thủ công")
            print(f"    trọng số ~{hand_w:.3f} (≈0). Nếu nhánh đó đóng góp bằng 0 mà")
            print("    kết quả vẫn khác, chênh lệch đến từ nhiễu, không từ fusion.")
            print("    → Cần chạy --gate-reg và kiểm định thống kê F2 vs B0.")
        else:
            print("  → KẾT LUẬN: nhánh thủ công KHÔNG bổ sung gì đáng kể.")
            print("    HuBERT fine-tuned đã học đủ đặc trưng cảm xúc, KỂ CẢ prosody.")
            print("    Kiến trúc ĐƠN GIẢN HƠN mà vẫn mạnh hơn.")
            print("    (Kết luận CÓ GIÁ TRỊ — đối lập với fusion trên embedding FROZEN,")
            print("     vốn cần nhánh thủ công để bù cho backbone chưa fine-tune.)")

        # Diễn giải gate cùng accuracy đơn nhánh
        for k, lbl in [("B1. MFCC only", "MFCC"), ("B2. Prosody only", "Prosody")]:
            if k in summ.index:
                print(f"    ({lbl}-only = {summ.loc[k,'Acc']:.1f}% → đọc gate CÙNG "
                      f"con số này, đừng đọc gate đơn độc)")

    if df["Fold"].nunique() >= 3:
        print("\n" + "=" * 74)
        print(f"  KIỂM ĐỊNH THỐNG KÊ (paired, n={df['Fold'].nunique()})")
        print("=" * 74)
        try:
            st = paired_tests(df)
            if not st.empty:
                print(st.to_string(index=False))
                st.to_csv(out("statistical_tests_fusion.csv"), index=False)
                print("\n  [LƯU Ý] n=6 → Wilcoxon p sàn = 0.0312. Dùng t-test + Cohen's d.")
        except ImportError:
            print("  [WARN] Cần scipy")

    # ── Kiểm định riêng: F2 vs B0 (câu hỏi trọng tâm) ────────────────────────
    if df["Fold"].nunique() >= 3 and B0 in summ.index:
        try:
            from scipy import stats
            piv = df.pivot_table(index="Fold", columns="Model", values="Accuracy(%)")
            for fm in [m for m in piv.columns if m.startswith(("F1", "F2", "F3"))]:
                a, b = piv[B0].values, piv[fm].values
                d = b - a
                t_p = stats.ttest_rel(b, a).pvalue
                cd = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else 0.0
                verdict = "CÓ ý nghĩa" if t_p < 0.05 else "KHÔNG đủ bằng chứng"
                print(f"  {fm:26s} vs B0:  Δ={d.mean():+6.2f}%  "
                      f"t_p={t_p:.4f}  d={cd:5.2f}  → {verdict}")
        except ImportError:
            pass

    plot_results(df)
    plot_gates(df, summ)
    plot_gate_history(df)

    for f in OUT_DIR.glob("_fus_*.json"):
        f.unlink()
    p = Path(out("results_partial.csv"))
    if p.exists():
        p.unlink()

    print("\n" + "=" * 74)
    print("  HOÀN THÀNH")
    print(f"  • {out('results_fusion_finetuned.csv')}")
    print(f"  • {out('fusion_finetuned_comparison.png')}")
    print(f"  • {out('fusion_gate_weights.png')}")
    print("=" * 74 + "\n")


def _row(model: str, fold: str, r: dict) -> dict:
    d = {
        "Model":        model,
        "Fold":         fold,
        "Accuracy(%)":  round(r["Accuracy(%)"], 2),
        "F1_macro(%)":  round(r["F1_macro(%)"], 2),
        "Precision(%)": round(r["Precision(%)"], 2),
        "Recall(%)":    round(r["Recall(%)"], 2),
        "Time(s)":      round(r["Time(s)"], 1),
    }
    for k in ("gate_mfcc", "gate_pros", "gate_emb"):
        if k in r:
            d[k] = round(r[k], 3)
    if "gate_collapse_epoch" in r:
        d["gate_collapse_epoch"] = r["gate_collapse_epoch"]
        d["gate_hand_ep1"] = round(r["gate_hand_ep1"], 3)
        d["gate_history"] = json.dumps(r["gate_history"])
    return d


if __name__ == "__main__":
    main()
