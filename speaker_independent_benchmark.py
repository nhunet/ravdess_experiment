"""
=============================================================================
BENCHMARK SPEAKER-INDEPENDENT – SO SÁNH 3 MÔ HÌNH DƯỚI CÙNG MỘT GIAO THỨC
Đề tài: Hệ thống gợi ý sản phẩm dựa trên phân tích giọng nói và cảm xúc
Học viên: Nguyễn Tấn Nhu | GVHD: TS. Bùi Thanh Hùng
=============================================================================

TẠI SAO CẦN FILE NÀY

  Kết quả speaker-independent của hubert_finetune.py cho thấy:

      val_acc  (actor 1-20, model ĐÃ THẤY giọng)   = 85.56%
      test_acc (actor 21-24, giọng HOÀN TOÀN MỚI)  = 70.83%
                                                     ─────────
                                          khoảng cách  -14.7%

  Khoảng cách 14.7% này CHÍNH LÀ phần model học "giọng của actor X" thay vì
  học "cảm xúc". Nghĩa là con số 89.58% (5-fold ngẫu nhiên) BỊ THỔI LÊN do
  speaker leakage: RAVDESS chỉ có 24 actor, mỗi actor thu đủ 8 cảm xúc, nên
  khi chia ngẫu nhiên thì giọng cùng một actor xuất hiện ở CẢ train lẫn test.

  Đây không phải lỗi của riêng ai — hầu hết bài báo dùng RAVDESS với random
  k-fold đều dính. Nhưng người phản biện sẽ chọc đúng vào chỗ này.

LỖI SO SÁNH TRONG hubert_finetune.py (file này sinh ra để SỬA)

  Script cũ in ra:  "So với HuBERT frozen + SVM (81.94%): Δ = -11.11%"
  → SAI. 81.94% là con số 5-FOLD (có leakage), còn 70.83% là SPEAKER-INDEP
    (không leakage). So hai thứ khác giao thức = so điểm thi có phao với
    điểm thi không phao. Vô nghĩa.

  Muốn biết fine-tune thắng hay thua THẬT thì phải đo CẢ BA model dưới
  CÙNG một split. Đó là việc của file này.

GIAO THỨC: LEAVE-ONE-SPEAKER-GROUP-OUT (LOSGO)

  24 actor → chia 6 nhóm × 4 actor:
      Group 1: actor 01-04      Group 4: actor 13-16
      Group 2: actor 05-08      Group 5: actor 17-20
      Group 3: actor 09-12      Group 6: actor 21-24

  6 fold: mỗi fold lấy 1 nhóm làm TEST, 5 nhóm còn lại làm TRAIN.
  → Mỗi mẫu được test đúng 1 lần, và có mean ± std (khắc phục điểm yếu
    của split đơn 70.83% – chỉ 1 lần đo, không biết dao động bao nhiêu).

  RAVDESS cân bằng giới tính (actor lẻ = nam, chẵn = nữ) nên mỗi nhóm 4
  actor liên tiếp luôn có 2 nam + 2 nữ → không lệch giới.

BA MÔ HÌNH ĐƯỢC ĐO

  A. HuBERT (frozen) + SVM   – baseline gốc, 81.94% ở 5-fold
  B. AttGate Fusion          – MFCC + HuBERT frozen, 87.15% ở 5-fold
  C. HuBERT fine-tuned       – cải tiến #1, 89.58% ở 5-fold

  Cả ba dùng CHUNG một bộ split → con số so sánh được với nhau.

OUTPUT
  • results_speaker_independent.csv   – bảng đầy đủ 3 model × 6 fold
  • speaker_independent_comparison.png – biểu đồ 5-fold vs LOSGO
  • speaker_leakage_analysis.png       – định lượng mức leakage

CÁCH CHẠY
  python speaker_independent_benchmark.py                # cả 3 model (~40 phút GPU)
  python speaker_independent_benchmark.py --models A B   # chỉ model nhẹ (~5 phút)
  python speaker_independent_benchmark.py --quick        # 2 fold, 10 epochs
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
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)

warnings.filterwarnings("ignore")

# ── Dùng lại nguyên xi model + data loader từ cải tiến #1 ────────────────────
from hubert_finetune import (
    HubertSER, RavdessDataset, load_ravdess_16k, set_seed, free_gpu, gpu_mem,
    _repair_pos_conv,                      # ← FIX #1: bắt buộc, dùng cho A/B
    AttentionPooling,                       # dùng cho baseline G (Wav2Vec2)
    TARGET_SR, MAX_SAMPLES, EMOTIONS, RANDOM_SEED,
    BATCH_SIZE, GRAD_ACCUM, LR_BACKBONE, LR_HEAD, WEIGHT_DECAY,
    WARMUP_RATIO, PATIENCE, LABEL_SMOOTH, VAL_RATIO,
    _make_optimizer, _evaluate,
)

OUT_DIR = Path(os.environ.get("OUT_DIR", "./outputs_speaker_indep_bench"))


def out(f: str) -> str:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return str(OUT_DIR / f)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── KIỂM TRA BẮT BUỘC: hubert_finetune.py phải có FIX #8 ─────────────────────
# Nếu thiếu, pos_conv_embed của HuBERT bị khởi tạo NGẪU NHIÊN thay vì nạp từ
# checkpoint → model C chạy trên một HuBERT hỏng → mọi con số vô nghĩa.
import hubert_finetune as _hf
if not hasattr(_hf, "_repair_pos_conv"):
    import inspect
    if "_repair_pos_conv" not in inspect.getsource(_hf):
        raise RuntimeError(
            "\n" + "=" * 70 +
            "\nhubert_finetune.py CHƯA CÓ FIX #8 (_repair_pos_conv)."
            "\n\nKhông có nó, pos_conv_embed bị khởi tạo ngẫu nhiên thay vì nạp"
            "\ntừ checkpoint → HuBERT hỏng một phần → kết quả model C vô nghĩa."
            "\n\nHãy dùng đúng bản hubert_finetune.py đã vá (bản đang chạy trên"
            "\nColab, log có in dòng '[FIX#8] Đã nạp lại pos_conv_embed').\n"
            + "=" * 70
        )

# Handcrafted feature params (khớp fusion_experiment.py)
SR_HC      = 22050
N_MFCC     = 40
N_FFT      = 2048
HOP_LENGTH = 512
DURATION   = 3.0

# Kết quả 5-fold đã có (để định lượng mức leakage)
FIVEFOLD = {
    "A. HuBERT frozen + SVM": 81.94,
    "B. AttGate Fusion":      87.15,
    "C. HuBERT fine-tuned":   89.58,
}

# Mã model (dùng ở --models) → tên đầy đủ trong bảng kết quả và tên checkpoint.
MODEL_NAMES = {
    "A": "A. HuBERT frozen + SVM",
    "B": "B. AttGate Fusion",
    "C": "C. HuBERT fine-tuned",
    # ── 4 baseline bổ sung (lấp lỗ hổng 7.1 #2 của báo cáo tháng 7): ─────────
    # Các bài SER trước đây báo cáo trên RAVDESS ở 5-fold ngẫu nhiên với các
    # kiến trúc CNN/LSTM/CRNN, và Wav2Vec2 là backbone SSL "kinh điển" cạnh
    # tranh HuBERT. Chưa có so sánh dưới cùng giao thức LOSGO trên chính
    # RAVDESS. Bốn model dưới đây train trên cùng splits, cùng val strategy
    # speaker-disjoint, cùng epochs/patience — để so công bằng với A/B/C.
    "D": "D. CNN2D on MFCC",
    "E": "E. BiLSTM on MFCC",
    "F": "F. CRNN (Conv+BiLSTM) on MFCC",
    "G": "G. Wav2Vec2 fine-tuned",
}

# Thứ tự cảm xúc để vẽ (LabelEncoder sắp xếp theo alphabet, không theo thứ tự này)
EMO_ORDER = ["neutral", "calm", "happy", "sad",
             "angry", "fearful", "disgust", "surprised"]

FUSION_EPOCHS = 60
FUSION_BATCH  = 32


# ============================================================================
# PHẦN 1: LEAVE-ONE-SPEAKER-GROUP-OUT SPLIT
# ============================================================================

def losgo_splits(actors: np.ndarray, n_groups: int = 6) -> list[tuple]:
    """
    24 actor → 6 nhóm × 4 actor. Mỗi fold: 1 nhóm test, 5 nhóm train.

    RAVDESS: actor lẻ = nam, chẵn = nữ → 4 actor liên tiếp luôn 2 nam + 2 nữ.
    """
    uniq = np.sort(np.unique(actors))
    groups = np.array_split(uniq, n_groups)

    splits = []
    for i, g in enumerate(groups):
        te = np.where(np.isin(actors, g))[0]
        tr = np.where(~np.isin(actors, g))[0]
        name = f"G{i+1} (actor {g.min():02d}-{g.max():02d})"
        splits.append((name, tr, te, g))
    return splits


# ============================================================================
# PHẦN 2: ĐẶC TRƯNG
# ============================================================================

def load_ravdess_22k(data_path: str = "./RAVDESS") -> np.ndarray:
    """
    VÁ LỖI #2: load audio Ở ĐÚNG 22050Hz TỪ FILE GỐC.

    Bản trước upsample 16k → 22050. SAI: waveform 16k có Nyquist = 8kHz, upsample
    KHÔNG tạo lại được nội dung trên 8kHz. Trong khi fusion_experiment.py gốc
    (cho 87.15%) load thẳng file .wav ở 22050 → MFCC có nội dung tới 11kHz.

    Hệ quả nếu không vá: MFCC của model B nghèo hơn MFCC trong thí nghiệm gốc
    → model B bị làm YẾU ĐI một cách giả tạo → so sánh với C mất công bằng.
    """
    path = Path(data_path)
    wav_files = sorted(path.rglob("*.wav"))
    target = int(SR_HC * DURATION)

    waves = []
    print(f"  [Audio 22050Hz] Load {len(wav_files)} files ở sample rate GỐC...")
    for i, fp in enumerate(wav_files):
        if i % 400 == 0:
            print(f"      {i}/{len(wav_files)}")
        parts = fp.stem.split("-")
        if len(parts) < 7 or int(parts[2]) not in EMOTIONS:
            continue
        y, _ = librosa.load(str(fp), sr=SR_HC, duration=DURATION, mono=True)
        if len(y) < target:
            y = np.pad(y, (0, target - len(y)), mode="constant")
        else:
            y = y[:target]
        peak = np.abs(y).max()
        if peak > 0:
            y = y / peak
        waves.append(y.astype(np.float32))
    return np.stack(waves)


def extract_mfcc_hc(waves22k: np.ndarray) -> np.ndarray:
    """MFCC + delta + delta² → 240 chiều. Nhận waveform 22050Hz GỐC (đã vá #2)."""
    print("  [MFCC] Đang trích xuất từ audio 22050Hz gốc...")
    feats = []
    for i, y in enumerate(waves22k):
        if i % 400 == 0:
            print(f"      {i}/{len(waves22k)}")
        m = librosa.feature.mfcc(y=y, sr=SR_HC, n_mfcc=N_MFCC,
                                 n_fft=N_FFT, hop_length=HOP_LENGTH)
        d1 = librosa.feature.delta(m)
        d2 = librosa.feature.delta(m, order=2)
        feats.append(np.concatenate([
            np.mean(m, axis=1),  np.std(m, axis=1),
            np.mean(d1, axis=1), np.std(d1, axis=1),
            np.mean(d2, axis=1), np.std(d2, axis=1),
        ]))
    X = np.array(feats)
    print(f"  [MFCC] shape={X.shape}")
    return X


@torch.no_grad()
def extract_hubert_frozen(waves: np.ndarray) -> np.ndarray:
    """
    HuBERT FROZEN – mean-pool last_hidden_state → 768 chiều (baseline 81.94%).

    VÁ LỖI #1: GỌI _repair_pos_conv() Ở ĐÂY.
    Bản trước tôi bắt model C phải có FIX #8, thậm chí raise lỗi nếu thiếu —
    rồi lại QUÊN vá cho A và B. Nghĩa là A/B đấu với C bằng một HuBERT có
    pos_conv_embed khởi tạo ngẫu nhiên. C thắng là đương nhiên, và chiến thắng
    đó VÔ GIÁ TRỊ. Đây là kiểu lỗi cho ra kết quả đẹp nhưng sai — nguy hiểm
    hơn lỗi làm crash.
    """
    from transformers import HubertModel
    print("  [HuBERT frozen] Đang trích xuất embedding 768-d...")

    m = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    _repair_pos_conv(m, "facebook/hubert-base-ls960")   # ← VÁ #1
    m = m.to(DEVICE).eval()

    embs = []
    for i in range(0, len(waves), 16):
        batch = torch.from_numpy(waves[i:i+16]).to(DEVICE)
        with torch.autocast(device_type=DEVICE.type,
                            enabled=(DEVICE.type == "cuda")):
            o = m(batch)
        embs.append(o.last_hidden_state.float().mean(dim=1).cpu().numpy())
        if i % 320 == 0:
            print(f"      {i}/{len(waves)}")

    X = np.concatenate(embs)
    free_gpu(m)
    print(f"  [HuBERT frozen] shape={X.shape}")
    return X


def make_val_split(tr: np.ndarray, y: np.ndarray, actors: np.ndarray,
                   speaker_disjoint: bool,
                   seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    VÁ LỖI #3: validation cũng phải SPEAKER-DISJOINT.

    Bản trước tách val bằng train_test_split ngẫu nhiên từ train fold → val
    CÙNG ACTOR với train. Early stopping vì thế chọn checkpoint "giỏi nhận
    giọng quen nhất", rồi mới đem test trên giọng lạ → chọn sai checkpoint
    cho đúng mục tiêu.

    speaker_disjoint=True : lấy 4 actor trong train fold ra làm val
                            → early-stop dựa trên giọng LẠ, khớp mục tiêu.
    speaker_disjoint=False: val cùng actor (giữ để ĐO khoảng cách leakage).

    ── VÁ LỖI MULTI-SEED (quan trọng cho sweep 3 seed) ──────────────────────
    Trước đây hàm này luôn dùng RANDOM_SEED cố định (=42) lấy từ
    hubert_finetune, KHÔNG đọc biến môi trường SEED. Hệ quả: chạy sweep
    SEED=42/43/44 thì cả 3 seed đều chọn ĐÚNG 4 diễn viên validation giống
    nhau — variance đo được vì thế hẹp hơn thực tế, vì một nguồn biến thiên
    lớn (chọn ai làm val) bị đóng băng.

    Giờ nhận tham số `seed`. Bỏ trống → giữ nguyên hành vi cũ (RANDOM_SEED),
    nên MỌI kết quả đã chạy với seed 42 vẫn tái lập được y hệt; chỉ seed
    43/44 mới nhận split khác.
    """
    seed = RANDOM_SEED if seed is None else int(seed)

    if not speaker_disjoint:
        sub_tr, sub_val = train_test_split(
            np.arange(len(tr)), test_size=VAL_RATIO,
            stratify=y[tr], random_state=seed)
        return tr[sub_tr], tr[sub_val]

    # Lấy 4 actor cuối trong train fold làm val (2 nam + 2 nữ nếu liên tiếp)
    tr_actors = np.unique(actors[tr])
    rng = np.random.RandomState(seed)
    val_actors = rng.choice(tr_actors, size=4, replace=False)

    mask_val = np.isin(actors[tr], val_actors)
    return tr[~mask_val], tr[mask_val]


# ============================================================================
# PHẦN 3: BA MÔ HÌNH
# ============================================================================

def _metrics(y_true, y_pred, elapsed) -> dict:
    """Tính đủ 4 metric + thời gian, dùng chung cho cả 3 model."""
    return {
        "Accuracy(%)":  accuracy_score(y_true, y_pred) * 100,
        "F1_macro(%)":  f1_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "Precision(%)": precision_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "Recall(%)":    recall_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "Time(s)":      elapsed,
        "y_pred":       y_pred,
        "y_true":       y_true,
    }


def run_A_frozen_svm(X_emb, y, tr, te) -> dict:
    """A. HuBERT frozen + SVM (RBF) – baseline gốc."""
    clf = Pipeline([
        ("sc", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_SEED)),
    ])
    t0 = time.time()
    clf.fit(X_emb[tr], y[tr])
    p = clf.predict(X_emb[te])
    return _metrics(y[te], p, time.time() - t0)


def _build_attgate(dim_hc: int, dim_emb: int, n_cls: int, proj: int = 256):
    """AttGate Fusion – copy nguyên kiến trúc từ fusion_experiment.py."""

    class AttentionGateFusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj_hc = nn.Sequential(
                nn.Linear(dim_hc, proj), nn.BatchNorm1d(proj),
                nn.ReLU(), nn.Dropout(0.3))
            self.proj_emb = nn.Sequential(
                nn.Linear(dim_emb, proj), nn.BatchNorm1d(proj),
                nn.ReLU(), nn.Dropout(0.3))
            self.gate = nn.Sequential(
                nn.Linear(proj * 2, 64), nn.ReLU(),
                nn.Linear(64, 2), nn.Softmax(dim=-1))
            self.classifier = nn.Sequential(
                nn.Linear(proj, 128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, n_cls))

        def forward(self, x_hc, x_emb):
            h1, h2 = self.proj_hc(x_hc), self.proj_emb(x_emb)
            g = self.gate(torch.cat([h1, h2], dim=1))
            fused = g[:, 0:1] * h1 + g[:, 1:2] * h2
            return self.classifier(fused), g

    return AttentionGateFusion()


def run_B_attgate(X_hc, X_emb, y, tr, te, n_cls, actors, spk_disjoint) -> dict:
    """
    B. AttGate Fusion – MFCC + HuBERT frozen.

    VÁ LỖI #4: THÊM validation + early stopping.
    Bản trước B chạy cứng 60 epochs, không val, không early-stop — trong khi
    C có cả hai. Ba model không cùng "ngân sách tối ưu hoá" thì so sánh vẫn
    lệch. Giờ B dùng ĐÚNG cơ chế val/early-stop như C.
    """
    set_seed(RANDOM_SEED)
    t0 = time.time()

    real_tr, real_val = make_val_split(tr, y, actors, spk_disjoint)

    s1, s2 = StandardScaler(), StandardScaler()
    hc_tr = s1.fit_transform(X_hc[real_tr]).astype(np.float32)
    hc_va = s1.transform(X_hc[real_val]).astype(np.float32)
    hc_te = s1.transform(X_hc[te]).astype(np.float32)
    em_tr = s2.fit_transform(X_emb[real_tr]).astype(np.float32)
    em_va = s2.transform(X_emb[real_val]).astype(np.float32)
    em_te = s2.transform(X_emb[te]).astype(np.float32)

    dl = DataLoader(
        TensorDataset(torch.tensor(hc_tr), torch.tensor(em_tr),
                      torch.tensor(y[real_tr], dtype=torch.long)),
        batch_size=FUSION_BATCH, shuffle=True)

    model = _build_attgate(hc_tr.shape[1], em_tr.shape[1], n_cls).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FUSION_EPOCHS)
    lossfn = nn.CrossEntropyLoss()

    def _pred(xh, xe):
        model.eval()
        with torch.no_grad():
            lg, _ = model(torch.tensor(xh).to(DEVICE), torch.tensor(xe).to(DEVICE))
        return lg.argmax(1).cpu().numpy()

    best_f1, best_state, bad, best_val_acc = 0.0, None, 0, 0.0

    for _ in range(FUSION_EPOCHS):
        model.train()
        for a, b, yy in dl:
            a, b, yy = a.to(DEVICE), b.to(DEVICE), yy.to(DEVICE)
            opt.zero_grad()
            lossfn(model(a, b)[0], yy).backward()
            opt.step()
        sch.step()

        vp = _pred(hc_va, em_va)
        vf1 = f1_score(y[real_val], vp, average="macro", zero_division=0)
        if vf1 > best_f1:
            best_f1, bad = vf1, 0
            # VÁ LỖI #5: ghi val_acc TẠI EPOCH ĐƯỢC CHỌN (không phải max toàn cục)
            best_val_acc = accuracy_score(y[real_val], vp) * 100
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)

    r = _metrics(y[te], _pred(hc_te, em_te), time.time() - t0)
    r["val_acc"] = best_val_acc
    free_gpu(model)
    return r


def run_C_finetune(waves, y, tr, te, n_cls, epochs, actors, spk_disjoint) -> dict:
    """C. HuBERT fine-tuned – cải tiến #1, đúng pipeline hubert_finetune.py."""
    set_seed(RANDOM_SEED)
    t0 = time.time()

    # FIX #3: val speaker-disjoint (hoặc không, tuỳ chế độ – để đo leakage)
    real_tr, real_val = make_val_split(tr, y, actors, spk_disjoint)

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

    best_f1, best_state, bad, best_val_acc = 0.0, None, 0, 0.0

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
            # FIX #5: val_acc TẠI epoch được chọn, không phải max toàn cục
            best_val_acc = accuracy_score(vt, vp) * 100
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
    r["val_acc"] = best_val_acc
    free_gpu(model)
    return r


# ============================================================================
# PHẦN 3B: BASELINE DEEP LEARNING (D, E, F, G)
# ----------------------------------------------------------------------------
# Lấp lỗ hổng 7.1 #2 của báo cáo tháng 7: chưa có CNN/LSTM/CRNN/Wav2Vec2 dưới
# LOSGO. Bốn model dưới đây train trên CÙNG splits, val speaker-disjoint,
# epochs, patience như model C, để so công bằng.
# ============================================================================

# Params riêng cho MFCC time-series (không đụng extract_mfcc_hc dùng cho B)
N_MFCC_SEQ  = 40    # số hệ số MFCC (chuẩn cho SER)
HOP_SEQ     = 512   # hop → T ≈ 130 frames với DURATION=3s @ 22.05 kHz
BATCH_MFCC  = 32    # MFCC nhẹ hơn waveform nhiều, tăng batch lên
EPOCHS_MFCC = 60    # tăng epochs vì mô hình nhỏ, hội tụ chậm hơn
LR_MFCC     = 1e-3  # LR chuẩn cho model random-init
WD_MFCC     = 1e-4
PATIENCE_MFCC = 10  # nới patience vì val có thể dao động


def extract_mfcc_seq(waves22k: np.ndarray) -> np.ndarray:
    """
    MFCC theo trục thời gian → (N, F, T) — cho CNN2D/LSTM/CRNN.

    Khác extract_mfcc_hc (dùng cho model B): hàm đó ĐÃ nén thời gian bằng
    mean/std → vector 240-d tĩnh, không phù hợp cho model có cấu trúc temporal.

    Chuẩn hoá per-utterance (z-score) để CNN/LSTM ổn định — không dùng
    StandardScaler global vì mỗi mẫu là một chuỗi.
    """
    print(f"  [MFCC-seq] Trích MFCC theo thời gian ({N_MFCC_SEQ} coef)...")
    feats = []
    for i, y in enumerate(waves22k):
        if i % 400 == 0:
            print(f"      {i}/{len(waves22k)}")
        m = librosa.feature.mfcc(y=y, sr=SR_HC, n_mfcc=N_MFCC_SEQ,
                                 n_fft=N_FFT, hop_length=HOP_SEQ)
        feats.append(m.astype(np.float32))
    # waveform đã pad về DURATION cố định nên T đồng đều; safe cắt về min
    T = min(f.shape[1] for f in feats)
    X = np.stack([f[:, :T] for f in feats])
    mean = X.mean(axis=(1, 2), keepdims=True)
    std = X.std(axis=(1, 2), keepdims=True) + 1e-8
    X = (X - mean) / std
    print(f"  [MFCC-seq] shape={X.shape}  (N, F, T)")
    return X


class _MFCCDataset(torch.utils.data.Dataset):
    """(N, F, T) → tensor, không augment (dùng chung cho D/E/F)."""
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y
    def __len__(self) -> int:
        return len(self.y)
    def __getitem__(self, i):
        return (torch.from_numpy(self.X[i]),
                torch.tensor(self.y[i], dtype=torch.long))


# ── D. CNN2D on MFCC spectrogram ────────────────────────────────────────────
class _CNN2D_SER(nn.Module):
    """
    Coi MFCC (F, T) như ảnh 1 kênh. Kiến trúc chuẩn Issa et al. 2020 (rút gọn)
    — 3 khối Conv+BN+ReLU+MaxPool + AdaptiveAvgPool + MLP head.
    Chọn kiến trúc phổ biến/đủ mạnh, không tối ưu quá — mục tiêu là baseline
    trung thực chứ không phải leaderboard.
    """
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Dropout2d(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 256), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )
    def forward(self, x):
        # (B, F, T) → (B, 1, F, T)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.classifier(self.features(x))


# ── E. BiLSTM on MFCC sequence ──────────────────────────────────────────────
class _BiLSTM_SER(nn.Module):
    """
    BiLSTM 2 lớp × 128 hidden trên MFCC theo trục thời gian. Mean-pool để
    giữ nguyên chi phí tính so với B0 (embedding-only) của cải tiến #2.
    """
    def __init__(self, n_classes: int, input_dim: int = N_MFCC_SEQ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 128, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )
    def forward(self, x):
        # (B, F, T) → (B, T, F) để LSTM nhận đúng thứ tự (batch, seq, feat)
        if x.dim() == 3 and x.shape[1] == N_MFCC_SEQ:
            x = x.transpose(1, 2)
        out, _ = self.lstm(x)          # (B, T, 256)
        return self.classifier(out.mean(dim=1))


# ── F. CRNN (Conv trên freq, LSTM trên time) ────────────────────────────────
class _CRNN_SER(nn.Module):
    """
    3 khối Conv2D chỉ pool theo trục freq (giữ nguyên time) → reshape → BiLSTM.
    Là kết hợp phổ biến của SER (Trigeorgis et al. 2016-style):
      • CNN học đặc trưng local trên freq axis
      • LSTM học phụ thuộc dài trên time axis
    Sau 3 lần MaxPool((2,1)) trên trục freq: F 40 → 5 → mỗi time-step có
    128×5=640 feature.
    """
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)), nn.Dropout2d(0.3),
        )
        self.lstm = nn.LSTM(128 * (N_MFCC_SEQ // 8), 128, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        h = self.conv(x)                # (B, C, F', T)
        b, c, f, t = h.shape
        h = h.permute(0, 3, 1, 2).reshape(b, t, c * f)  # (B, T, C·F')
        out, _ = self.lstm(h)
        return self.classifier(out.mean(dim=1))


def _train_mfcc_model(model: nn.Module, X: np.ndarray, y: np.ndarray,
                      tr: np.ndarray, te: np.ndarray, actors: np.ndarray,
                      spk_disjoint: bool, epochs: int = EPOCHS_MFCC) -> dict:
    """Hàm train chung cho D/E/F: MFCC input, cross-entropy, AdamW, early-stop."""
    set_seed(RANDOM_SEED)
    t0 = time.time()
    real_tr, real_val = make_val_split(tr, y, actors, spk_disjoint)

    kw = dict(pin_memory=(DEVICE.type == "cuda"),
              num_workers=2 if DEVICE.type == "cuda" else 0)
    tr_dl = DataLoader(_MFCCDataset(X[real_tr], y[real_tr]),
                       batch_size=BATCH_MFCC, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(_MFCCDataset(X[real_val], y[real_val]),
                        batch_size=BATCH_MFCC, **kw)
    te_dl = DataLoader(_MFCCDataset(X[te], y[te]),
                       batch_size=BATCH_MFCC, **kw)

    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_MFCC, weight_decay=WD_MFCC)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossfn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    best_f1, best_state, bad, best_val_acc = 0.0, None, 0, 0.0

    for _ep in range(epochs):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            with torch.autocast(device_type=DEVICE.type,
                                enabled=(DEVICE.type == "cuda")):
                loss = lossfn(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        sch.step()

        vp, vt = _evaluate(model, val_dl)
        vf1 = f1_score(vt, vp, average="macro", zero_division=0)

        if vf1 > best_f1:
            best_f1, bad = vf1, 0
            best_val_acc = accuracy_score(vt, vp) * 100
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE_MFCC:
                break

    if best_state:
        model.load_state_dict(best_state)

    p, t = _evaluate(model, te_dl)
    r = _metrics(t, p, time.time() - t0)
    r["val_acc"] = best_val_acc
    free_gpu(model)
    return r


def run_D_cnn2d(X_mfcc, y, tr, te, n_cls, actors, spk_disjoint,
                epochs: int = EPOCHS_MFCC):
    """D. CNN2D on MFCC."""
    return _train_mfcc_model(_CNN2D_SER(n_cls), X_mfcc, y, tr, te,
                             actors, spk_disjoint, epochs=epochs)


def run_E_bilstm(X_mfcc, y, tr, te, n_cls, actors, spk_disjoint,
                 epochs: int = EPOCHS_MFCC):
    """E. BiLSTM on MFCC."""
    return _train_mfcc_model(_BiLSTM_SER(n_cls), X_mfcc, y, tr, te,
                             actors, spk_disjoint, epochs=epochs)


def run_F_crnn(X_mfcc, y, tr, te, n_cls, actors, spk_disjoint,
               epochs: int = EPOCHS_MFCC):
    """F. CRNN (Conv + BiLSTM) on MFCC."""
    return _train_mfcc_model(_CRNN_SER(n_cls), X_mfcc, y, tr, te,
                             actors, spk_disjoint, epochs=epochs)


# ── G. Wav2Vec2 fine-tuned (mirror HubertSER architecture) ──────────────────
class _Wav2Vec2SER(nn.Module):
    """
    Mirror HubertSER (weighted layer-sum + attention pooling + embed_fc +
    classifier), chỉ đổi backbone → Wav2Vec2-base. Cùng head → so công bằng
    "HuBERT vs Wav2Vec2 as pretrained backbone".

    Không cần _repair_pos_conv — bug weight_norm mà FIX #8 vá chỉ ảnh hưởng
    HuBERT trong transformers 4.51; Wav2Vec2 dùng parametrization khác và
    HuggingFace ánh xạ đúng.
    """
    HIDDEN = 768
    EMBED_DIM = 256

    def __init__(self, n_classes: int,
                 model_name: str = "facebook/wav2vec2-base") -> None:
        super().__init__()
        from transformers import Wav2Vec2Model, Wav2Vec2Config
        cfg = Wav2Vec2Config.from_pretrained(model_name)
        cfg.output_hidden_states = True
        self.wav2vec = Wav2Vec2Model.from_pretrained(model_name, config=cfg)
        # Đóng băng CNN feature extractor (chuẩn khi fine-tune SSL cho SER)
        try:
            self.wav2vec.feature_extractor._freeze_parameters()
        except AttributeError:
            for p in self.wav2vec.feature_extractor.parameters():
                p.requires_grad = False

        n_layers = cfg.num_hidden_layers
        self.layer_weights = nn.Parameter(torch.zeros(n_layers + 1))
        self.pool = AttentionPooling(self.HIDDEN)
        self.embed_fc = nn.Sequential(
            nn.Linear(self.HIDDEN, self.EMBED_DIM), nn.ReLU(), nn.Dropout(0.1),
        )
        self.classifier = nn.Linear(self.EMBED_DIM, n_classes)

    def forward(self, x):
        out = self.wav2vec(x)
        # weighted layer-sum
        hs = torch.stack(out.hidden_states, dim=0)     # (L+1, B, T, H)
        w = torch.softmax(self.layer_weights, dim=0)
        h = (hs * w.view(-1, 1, 1, 1)).sum(dim=0)      # (B, T, H)
        return self.classifier(self.embed_fc(self.pool(h)))


def run_G_wav2vec2_ft(waves, y, tr, te, n_cls, epochs, actors, spk_disjoint):
    """
    G. Wav2Vec2 fine-tuned — copy pipeline của run_C_finetune, chỉ đổi model.
    Cùng optimizer (discriminative LR: backbone chậm, head nhanh), cùng
    OneCycleLR + AMP + grad clip → so công bằng với C.
    """
    set_seed(RANDOM_SEED)
    t0 = time.time()
    real_tr, real_val = make_val_split(tr, y, actors, spk_disjoint)

    kw = dict(pin_memory=(DEVICE.type == "cuda"),
              num_workers=2 if DEVICE.type == "cuda" else 0)
    tr_dl = DataLoader(RavdessDataset(waves[real_tr], y[real_tr]),
                       batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(RavdessDataset(waves[real_val], y[real_val]),
                        batch_size=BATCH_SIZE, **kw)
    te_dl = DataLoader(RavdessDataset(waves[te], y[te]),
                       batch_size=BATCH_SIZE, **kw)

    model = _Wav2Vec2SER(n_cls).to(DEVICE)

    # Discriminative LR: backbone (pretrained) học chậm; head học nhanh
    backbone, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone if name.startswith("wav2vec") else head).append(p)
    opt = torch.optim.AdamW(
        [{"params": backbone, "lr": LR_BACKBONE},
         {"params": head,     "lr": LR_HEAD}],
        weight_decay=WEIGHT_DECAY)

    lossfn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))
    steps = max(1, (len(tr_dl) // GRAD_ACCUM) * epochs)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[LR_BACKBONE, LR_HEAD],
        total_steps=steps, pct_start=WARMUP_RATIO)

    best_f1, best_state, bad, best_val_acc = 0.0, None, 0, 0.0

    for _ep in range(epochs):
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
            best_val_acc = accuracy_score(vt, vp) * 100
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
    r["val_acc"] = best_val_acc
    free_gpu(model)
    return r


# ============================================================================
# PHẦN 4: VISUALIZATION
# ============================================================================

def paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    VÁ LỖI #6: KIỂM ĐỊNH THỐNG KÊ.

    Ba model chạy trên CÙNG 6 fold → dữ liệu ghép cặp (paired) → dùng được
    paired t-test và Wilcoxon signed-rank.

    Tại sao bắt buộc: nếu C hơn B 3% nhưng p = 0.4 thì KHÔNG kết luận được gì —
    chênh lệch đó có thể chỉ là nhiễu. Người phản biện chắc chắn hỏi câu này.
    Với n=6 fold, Wilcoxon đáng tin hơn t-test (không giả định phân phối chuẩn),
    nên báo cáo cả hai.
    """
    from itertools import combinations
    from scipy import stats

    piv = df.pivot_table(index="Fold", columns="Model", values="Accuracy(%)")
    rows = []

    for m1, m2 in combinations(piv.columns, 2):
        a, b = piv[m1].dropna(), piv[m2].dropna()
        idx = a.index.intersection(b.index)
        a, b = a[idx].values, b[idx].values
        if len(a) < 3:
            continue

        diff = b - a
        t_p = stats.ttest_rel(b, a).pvalue
        try:
            w_p = stats.wilcoxon(b, a).pvalue
        except ValueError:          # tất cả diff = 0
            w_p = 1.0

        # Cohen's d (paired)
        d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0

        rows.append({
            "So sánh":      f"{m2}  vs  {m1}",
            "Δ Acc(%)":     round(diff.mean(), 2),
            "t-test p":     round(t_p, 4),
            "Wilcoxon p":   round(w_p, 4),
            "Cohen's d":    round(d, 2),
            "Kết luận":     ("CÓ ý nghĩa (p<0.05)" if w_p < 0.05
                             else "KHÔNG đủ bằng chứng"),
        })

    return pd.DataFrame(rows)


def per_emotion_analysis(preds: dict, le: LabelEncoder) -> None:
    """
    Per-emotion + confusion matrix cho từng model, gộp toàn bộ fold.

    Chỉ chạy được từ khi checkpoint lưu y_true/y_pred (xem phần vá trong
    main()). Trước đây `_row()` vứt predictions nên muốn có bảng per-emotion
    là phải train lại toàn bộ 6 fold — đúng thứ báo cáo/bài báo đang thiếu.
    """
    if not preds:
        print("\n[INFO] Chưa có checkpoint nào kèm predictions "
              "(checkpoint cũ không lưu) → bỏ qua per-emotion.\n"
              "       Chạy lại với --force để sinh dữ liệu này.")
        return

    from sklearn.metrics import classification_report

    print("\n" + "=" * 72)
    print("  PHÂN TÍCH PER-EMOTION (gộp mọi fold)")
    print("=" * 72)

    emo_rows, pair_rows = [], []
    for model_name, d in preds.items():
        yt, yp = np.array(d["true"]), np.array(d["pred"])
        acc = accuracy_score(yt, yp) * 100
        rep = classification_report(yt, yp, target_names=le.classes_,
                                    output_dict=True, zero_division=0)
        for emo in le.classes_:
            s = rep[emo]
            emo_rows.append({
                "Model": model_name, "Emotion": emo,
                "Precision(%)": round(s["precision"] * 100, 2),
                "Recall(%)":    round(s["recall"] * 100, 2),
                "F1(%)":        round(s["f1-score"] * 100, 2),
                "Support":      int(s["support"]),
            })

        cm = confusion_matrix(yt, yp).astype(float)
        cm_pct = cm / np.clip(cm.sum(axis=1, keepdims=True), 1e-9, None) * 100
        cls = list(le.classes_)

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                    xticklabels=cls, yticklabels=cls, linewidths=0.5, ax=ax,
                    cbar_kws={"label": "Tỷ lệ (%)"})
        ax.set_title(f"Confusion Matrix – {model_name}\n"
                     f"LOSGO 6-fold gộp, Acc={acc:.2f}%", fontweight="bold")
        ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế")
        plt.tight_layout()
        fname = f"confusion_{model_name.split('.')[0].strip()}.png"
        plt.savefig(out(fname), dpi=150, bbox_inches="tight")
        plt.close()

        pairs = sorted(((cls[i], cls[j], cm_pct[i, j])
                        for i in range(len(cls)) for j in range(len(cls))
                        if i != j), key=lambda x: -x[2])
        print(f"\n  [{model_name}]  Acc={acc:.2f}%  → {fname}")
        print("    3 cặp nhầm lẫn nặng nhất:")
        for a, b, v in pairs[:3]:
            print(f"      {a:10s} → bị đoán thành {b:10s}: {v:5.1f}%")

        row = {"Model": model_name}
        if "neutral" in cls and "calm" in cls:
            i_n, i_c = cls.index("neutral"), cls.index("calm")
            row["neutral→calm"] = round(float(cm_pct[i_n, i_c]), 2)
            row["calm→neutral"] = round(float(cm_pct[i_c, i_n]), 2)
            print(f"      ── Cặp kinh điển neutral↔calm: "
                  f"{row['neutral→calm']:.1f}% / {row['calm→neutral']:.1f}%")
        pair_rows.append(row)

    df_emo = pd.DataFrame(emo_rows)
    df_emo.to_csv(out("per_emotion_speaker_independent.csv"), index=False)
    pd.DataFrame(pair_rows).to_csv(out("confusion_pairs_bench.csv"), index=False)
    print(f"\n  → {out('per_emotion_speaker_independent.csv')}")


def plot_comparison(df: pd.DataFrame) -> None:
    """Biểu đồ chính: 5-fold (có leakage) vs LOSGO (không leakage)."""
    summ = df.groupby("Model").agg(
        acc=("Accuracy(%)", "mean"), std=("Accuracy(%)", "std"),
        f1=("F1_macro(%)", "mean")).reindex(FIVEFOLD.keys()).dropna(subset=["acc"])

    if summ.empty:
        print("[WARN] Không đủ dữ liệu để vẽ.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Speaker Leakage: 5-fold ngẫu nhiên vs Leave-One-Speaker-Group-Out",
                 fontsize=14, fontweight="bold")

    names = list(summ.index)
    x = np.arange(len(names))
    five = [FIVEFOLD[n] for n in names]
    losgo = summ["acc"].values
    # FIX #5b: std của 1 fold = NaN → matplotlib lỗi. Thay bằng 0.
    errs = np.nan_to_num(summ["std"].values, nan=0.0)

    axes[0].bar(x - 0.2, five, 0.4, label="5-fold ngẫu nhiên (CÓ leakage)",
                color="#E8A0A0", edgecolor="grey")
    axes[0].bar(x + 0.2, losgo, 0.4, yerr=errs, capsize=4,
                label="LOSGO (KHÔNG leakage)", color="#2E7D5B", edgecolor="grey")
    for i, (a, b) in enumerate(zip(five, losgo)):
        axes[0].text(i - 0.2, a + 0.5, f"{a:.1f}", ha="center", fontsize=9)
        axes[0].text(i + 0.2, b + 0.5, f"{b:.1f}", ha="center", fontsize=9,
                     fontweight="bold")
        axes[0].annotate(f"−{a-b:.1f}%", xy=(i, (a + b) / 2),
                         ha="center", fontsize=10, color="red", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=12, ha="right", fontsize=9)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Mức thổi phồng do speaker leakage", fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    for name in names:
        sub = df[df.Model == name]
        axes[1].plot(range(1, len(sub) + 1), sub["Accuracy(%)"],
                     "o-", label=name, linewidth=2)
    axes[1].set_xlabel("Speaker group (fold)")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Ổn định qua các nhóm speaker", fontweight="bold")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out("speaker_independent_comparison.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"[INFO] Biểu đồ → {out('speaker_independent_comparison.png')}")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["A", "B", "C"],
                    choices=list(MODEL_NAMES),
                    help="A=frozen+SVM  B=AttGate  C=HuBERT-FT  "
                         "D=CNN2D  E=BiLSTM  F=CRNN  G=Wav2Vec2-FT "
                         "(D/E/F/G bổ sung để lấp lỗ hổng 7.1 #2 của báo cáo 07)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--epochs-mfcc", type=int, default=EPOCHS_MFCC,
                    help="Epochs cho D/E/F (mặc định 60; mô hình nhỏ nên có thể "
                         "chạy nhiều epochs mà vẫn nhanh)")
    ap.add_argument("--quick", action="store_true", help="2 fold, 10 epochs")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--val-random", action="store_true",
                    help="Val CÙNG actor với train (để ĐO leakage). "
                         "Mặc định: val speaker-disjoint (FIX #3)")
    args = ap.parse_args()
    spk_disjoint = not args.val_random

    epochs = 10 if args.quick else args.epochs
    epochs_mfcc = 10 if args.quick else args.epochs_mfcc
    n_grp = 2 if args.quick else 6

    print("\n" + "=" * 72)
    print("  BENCHMARK SPEAKER-INDEPENDENT (Leave-One-Speaker-Group-Out)")
    print(f"  Device: {DEVICE}  |  Models: {', '.join(args.models)}")
    print(f"  {n_grp} nhóm × 4 actor  |  epochs (model C) = {epochs}")
    print(f"  Validation: {'SPEAKER-DISJOINT (nghiêm ngặt)' if spk_disjoint else 'cùng actor (đo leakage)'}")
    print(f"  Output → {OUT_DIR.absolute()}")
    print("=" * 72)

    waves, labels, actors = load_ravdess_16k("./RAVDESS")
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_cls = len(le.classes_)
    actors = np.array(actors)

    splits = losgo_splits(actors)[:n_grp]
    print(f"\n[INFO] {len(splits)} fold:")
    for name, tr, te, g in splits:
        print(f"    {name}: train={len(tr)}  test={len(te)}")

    # ── Trích đặc trưng 1 LẦN (dùng chung mọi fold – tiết kiệm rất nhiều) ────
    X_hc = X_emb = X_mfcc_seq = None
    needs_hubert_emb = bool({"A", "B"} & set(args.models))
    needs_mfcc_hc    = "B" in args.models
    needs_mfcc_seq   = bool({"D", "E", "F"} & set(args.models))

    if needs_hubert_emb:
        print("\n[INFO] Trích đặc trưng dùng chung...")
        X_emb = extract_hubert_frozen(waves)
    if needs_mfcc_hc or needs_mfcc_seq:
        # FIX #2: load audio ở 22050Hz GỐC, không upsample từ 16k
        waves22 = load_ravdess_22k("./RAVDESS")
        if needs_mfcc_hc:
            X_hc = extract_mfcc_hc(waves22)
        if needs_mfcc_seq:
            X_mfcc_seq = extract_mfcc_seq(waves22)
        del waves22

    # ── VÁ LỖI CHECKPOINT (2 lỗi thật, đều làm mất dữ liệu) ──────────────────
    # LỖI 1: checkpoint cũ khoá theo FOLD (_bench_G1.json), không theo MODEL.
    #        Chạy `--models A` trước rồi `--models C` sau → vòng resume thấy
    #        _bench_G1.json đã tồn tại nên BỎ QUA cả fold, model C không bao
    #        giờ được chạy, và bảng kết quả chỉ có mỗi model A.
    # LỖI 2: cuối main() có vòng xoá sạch _bench_*.json sau khi chạy xong →
    #        lần chạy sau mất toàn bộ dữ liệu cũ, buộc phải chạy lại từ đầu.
    # → Giờ khoá theo (fold, model) và GIỮ checkpoint. Đồng thời lưu
    #   y_true/y_pred để phân tích per-emotion / confusion về sau mà KHÔNG
    #   phải train lại (trước đây _row() vứt bỏ predictions).
    for name, tr, te, g in splits:
        print(f"\n{'─'*64}\n  {name}  |  VRAM: {gpu_mem()}\n{'─'*64}")
        fold_key = name.split()[0]

        # Tương thích ngược: đọc checkpoint ĐỊNH DẠNG CŨ nếu còn trên disk.
        legacy = OUT_DIR / f"_bench_{fold_key}.json"
        legacy_by_model = {}
        if legacy.exists():
            try:
                legacy_by_model = {r["Model"]: r
                                   for r in json.loads(legacy.read_text())}
            except (json.JSONDecodeError, KeyError, TypeError):
                print(f"  [WARN] Checkpoint cũ {legacy.name} hỏng → bỏ qua")

        for code in args.models:
            model_name = MODEL_NAMES[code]
            ck = OUT_DIR / f"_bench_{fold_key}_{code}.json"

            if ck.exists() and not args.force:
                saved = json.loads(ck.read_text())
                print(f"  [RESUME] {model_name} → bỏ qua "
                      f"({saved['row']['Accuracy(%)']:.2f}%)")
                continue

            if model_name in legacy_by_model and not args.force:
                # Nâng cấp sang định dạng mới. Checkpoint cũ KHÔNG có
                # predictions nên per-emotion sẽ thiếu model này cho tới khi
                # chạy lại với --force.
                row = legacy_by_model[model_name]
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                ck.write_text(json.dumps({"row": row,
                                          "y_true": None, "y_pred": None}))
                print(f"  [RESUME-CŨ] {model_name} → chuyển đổi định dạng "
                      f"({row['Accuracy(%)']:.2f}%, không có predictions)")
                continue

            if code == "A":
                r = run_A_frozen_svm(X_emb, y, tr, te)
            elif code == "B":
                r = run_B_attgate(X_hc, X_emb, y, tr, te, n_cls, actors,
                                  spk_disjoint)
            elif code == "C":
                r = run_C_finetune(waves, y, tr, te, n_cls, epochs, actors,
                                   spk_disjoint)
            elif code == "D":
                r = run_D_cnn2d(X_mfcc_seq, y, tr, te, n_cls, actors,
                                spk_disjoint, epochs=epochs_mfcc)
            elif code == "E":
                r = run_E_bilstm(X_mfcc_seq, y, tr, te, n_cls, actors,
                                 spk_disjoint, epochs=epochs_mfcc)
            elif code == "F":
                r = run_F_crnn(X_mfcc_seq, y, tr, te, n_cls, actors,
                               spk_disjoint, epochs=epochs_mfcc)
            elif code == "G":
                r = run_G_wav2vec2_ft(waves, y, tr, te, n_cls, epochs, actors,
                                      spk_disjoint)
            else:
                raise ValueError(f"Unknown model code: {code}")

            label = model_name.split(".")[0]
            print(f"  [{label}. {model_name.split('.',1)[1].strip()[:22]:<22s}] "
                  f"Acc={r['Accuracy(%)']:.2f}%  F1={r['F1_macro(%)']:.2f}%  "
                  f"({r['Time(s)']:.0f}s)")
            if code in ("C", "G"):
                gap = r.get("val_acc", 0) - r["Accuracy(%)"]
                print(f"       val_acc (cùng actor) = {r.get('val_acc',0):.2f}%  "
                      f"→ khoảng cách leakage = {gap:+.2f}%")

            OUT_DIR.mkdir(parents=True, exist_ok=True)
            ck.write_text(json.dumps({
                "row":    _row(model_name, name, r),
                "y_true": np.asarray(r["y_true"]).tolist(),
                "y_pred": np.asarray(r["y_pred"]).tolist(),
            }))

    # ── Tổng hợp: quét LẠI toàn bộ checkpoint trên disk ──────────────────────
    # Giống cơ chế của speaker_adversarial.py: bảng kết quả luôn đầy đủ mọi
    # model đã từng chạy, kể cả khi chạy tách nhiều lần / nhiều phiên Colab.
    rows, preds = [], {}
    for ck in sorted(OUT_DIR.glob("_bench_*_[A-G].json")):
        s = json.loads(ck.read_text())
        rows.append(s["row"])
        if s.get("y_true") is not None:
            p = preds.setdefault(s["row"]["Model"], {"true": [], "pred": []})
            p["true"].extend(s["y_true"])
            p["pred"].extend(s["y_pred"])

    df = pd.DataFrame(rows)
    if df.empty:
        print("\n[WARN] Chưa có checkpoint nào — dừng.")
        return
    df = df.sort_values(["Model", "Fold"]).reset_index(drop=True)
    df.to_csv(out("results_speaker_independent.csv"), index=False)
    print(f"\n[INFO] Gộp {df['Model'].nunique()} model × "
          f"{df['Fold'].nunique()} fold từ {len(rows)} checkpoint bền vững.")

    print("\n" + "=" * 72)
    print("  KẾT QUẢ – LEAVE-ONE-SPEAKER-GROUP-OUT")
    print("=" * 72)

    summ = df.groupby("Model").agg(
        Acc=("Accuracy(%)", "mean"), Std=("Accuracy(%)", "std"),
        F1=("F1_macro(%)", "mean")).round(2)
    print(summ.to_string())

    print("\n" + "=" * 72)
    print("  BẢNG CHO BÁO CÁO – ẢNH HƯỞNG CỦA SPEAKER LEAKAGE")
    print("=" * 72)
    print(f"  {'Model':<26} {'5-fold':>9} {'LOSGO':>16} {'Chênh':>9}")
    print("  " + "─" * 64)
    for m in summ.index:
        if m in FIVEFOLD:
            f5 = FIVEFOLD[m]
            lo = summ.loc[m, "Acc"]
            sd = summ.loc[m, "Std"]
            print(f"  {m:<26} {f5:>8.2f}% {lo:>10.2f}±{sd:<4.2f}% {lo-f5:>+8.2f}%")

    print("\n  → Cột 'Chênh' chính là mức THỔI PHỒNG do speaker leakage.")
    print("  → Cột LOSGO mới là năng lực THẬT trên giọng nói chưa từng nghe.")

    if len(summ) > 1:
        best = summ["Acc"].idxmax()
        print(f"\n  Model tốt nhất (speaker-independent): {best} "
              f"({summ.loc[best,'Acc']:.2f}%)")

        # ── FIX #6: KIỂM ĐỊNH THỐNG KÊ (paired – cùng 6 fold) ────────────────
        print("\n" + "=" * 72)
        print("  KIỂM ĐỊNH THỐNG KÊ (paired, n=%d fold)" % df["Fold"].nunique())
        print("=" * 72)
        try:
            st = paired_tests(df)
            if not st.empty:
                print(st.to_string(index=False))
                st.to_csv(out("statistical_tests.csv"), index=False)
                print("\n  → Chênh lệch chỉ ĐÁNG TIN khi p < 0.05.")
                print("  → Cohen's d: 0.2=nhỏ  0.5=vừa  0.8=lớn")
                n_fold = df["Fold"].nunique()
                if n_fold <= 6:
                    print(f"\n  [LƯU Ý QUAN TRỌNG] Với n={n_fold} fold, Wilcoxon có "
                          f"p TỐI THIỂU = {2**(-n_fold)*2:.4f}")
                    print("  → KHÔNG BAO GIỜ đạt p<0.05 dù model thắng cả 6/6 fold.")
                    print("  → Đây là giới hạn toán học của kiểm định phi tham số với")
                    print("    mẫu nhỏ, KHÔNG phải bằng chứng rằng chênh lệch là nhiễu.")
                    print("  → Trong báo cáo: dựa vào paired t-test + Cohen's d, và")
                    print("    ghi chú rõ giới hạn này của Wilcoxon.")
            else:
                print("  Không đủ fold để kiểm định (cần >= 3).")
        except ImportError:
            print("  [WARN] Cần scipy: pip install scipy")

    plot_comparison(df)
    per_emotion_analysis(preds, le)

    # KHÔNG xoá _bench_*.json nữa. Chúng là nguồn dữ liệu bền vững: giữ lại thì
    # chạy bổ sung model khác ở phiên Colab sau vẫn ra bảng đầy đủ, và
    # predictions bên trong là thứ sinh ra bảng per-emotion mà không phải
    # train lại. Muốn chạy lại từ đầu → dùng --force.
    # Dọn checkpoint ĐỊNH DẠNG CŨ (tên không có hậu tố _A/_B/_C) — dữ liệu
    # trong đó đã được chuyển sang định dạng mới ở vòng trên.
    for legacy in OUT_DIR.glob("_bench_*.json"):
        if not legacy.stem.endswith(tuple(f"_{c}" for c in MODEL_NAMES)):
            legacy.unlink()
    p = Path(out("results_partial.csv"))
    if p.exists():
        p.unlink()

    print("\n" + "=" * 72)
    print("  HOÀN THÀNH – Output:")
    print(f"  • {out('results_speaker_independent.csv')}")
    print(f"  • {out('speaker_independent_comparison.png')}")
    if preds:
        print(f"  • {out('per_emotion_speaker_independent.csv')}")
        print(f"  • confusion_A/B/C.png")
    print(f"  • {len(list(OUT_DIR.glob('_bench_*_[A-G].json')))} checkpoint "
          f"(giữ lại để chạy bổ sung / phân tích sau)")
    print("=" * 72 + "\n")


def _row(model: str, fold: str, r: dict) -> dict:
    return {
        "Model":        model,
        "Fold":         fold,
        "Accuracy(%)":  round(r["Accuracy(%)"], 2),
        "F1_macro(%)":  round(r["F1_macro(%)"], 2),
        "Precision(%)": round(r["Precision(%)"], 2),
        "Recall(%)":    round(r["Recall(%)"], 2),
        "Time(s)":      round(r["Time(s)"], 1),
        "val_acc(%)":   round(r.get("val_acc", 0), 2),
    }


if __name__ == "__main__":
    main()
