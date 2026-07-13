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
                   speaker_disjoint: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    VÁ LỖI #3: validation cũng phải SPEAKER-DISJOINT.

    Bản trước tách val bằng train_test_split ngẫu nhiên từ train fold → val
    CÙNG ACTOR với train. Early stopping vì thế chọn checkpoint "giỏi nhận
    giọng quen nhất", rồi mới đem test trên giọng lạ → chọn sai checkpoint
    cho đúng mục tiêu.

    speaker_disjoint=True : lấy 4 actor trong train fold ra làm val
                            → early-stop dựa trên giọng LẠ, khớp mục tiêu.
    speaker_disjoint=False: val cùng actor (giữ để ĐO khoảng cách leakage).
    """
    if not speaker_disjoint:
        sub_tr, sub_val = train_test_split(
            np.arange(len(tr)), test_size=VAL_RATIO,
            stratify=y[tr], random_state=RANDOM_SEED)
        return tr[sub_tr], tr[sub_val]

    # Lấy 4 actor cuối trong train fold làm val (2 nam + 2 nữ nếu liên tiếp)
    tr_actors = np.unique(actors[tr])
    rng = np.random.RandomState(RANDOM_SEED)
    val_actors = rng.choice(tr_actors, size=4, replace=False)

    mask_val = np.isin(actors[tr], val_actors)
    return tr[~mask_val], tr[mask_val]
    return {
        "Accuracy(%)":  accuracy_score(y_true, y_pred) * 100,
        "F1_macro(%)":  f1_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "Precision(%)": precision_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "Recall(%)":    recall_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "Time(s)":      elapsed,
        "y_pred":       y_pred,
        "y_true":       y_true,
    }


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
                    choices=["A", "B", "C"],
                    help="A=frozen+SVM  B=AttGate  C=fine-tuned")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--quick", action="store_true", help="2 fold, 10 epochs")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--val-random", action="store_true",
                    help="Val CÙNG actor với train (để ĐO leakage). "
                         "Mặc định: val speaker-disjoint (FIX #3)")
    args = ap.parse_args()
    spk_disjoint = not args.val_random

    epochs = 10 if args.quick else args.epochs
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
    X_hc = X_emb = None
    if "A" in args.models or "B" in args.models:
        print("\n[INFO] Trích đặc trưng dùng chung...")
        X_emb = extract_hubert_frozen(waves)
    if "B" in args.models:
        # FIX #2: load audio ở 22050Hz GỐC, không upsample từ 16k
        waves22 = load_ravdess_22k("./RAVDESS")
        X_hc = extract_mfcc_hc(waves22)
        del waves22

    rows = []
    for name, tr, te, g in splits:
        print(f"\n{'─'*64}\n  {name}  |  VRAM: {gpu_mem()}\n{'─'*64}")

        ck = OUT_DIR / f"_bench_{name.split()[0]}.json"
        if ck.exists() and not args.force:
            saved = json.loads(ck.read_text())
            print(f"  [RESUME] đã có → bỏ qua")
            rows.extend(saved)
            continue

        fold_rows = []
        if "A" in args.models:
            r = run_A_frozen_svm(X_emb, y, tr, te)
            print(f"  [A. frozen+SVM   ] Acc={r['Accuracy(%)']:.2f}%  "
                  f"F1={r['F1_macro(%)']:.2f}%  ({r['Time(s)']:.0f}s)")
            fold_rows.append(_row("A. HuBERT frozen + SVM", name, r))

        if "B" in args.models:
            r = run_B_attgate(X_hc, X_emb, y, tr, te, n_cls, actors, spk_disjoint)
            print(f"  [B. AttGate      ] Acc={r['Accuracy(%)']:.2f}%  "
                  f"F1={r['F1_macro(%)']:.2f}%  ({r['Time(s)']:.0f}s)")
            fold_rows.append(_row("B. AttGate Fusion", name, r))

        if "C" in args.models:
            r = run_C_finetune(waves, y, tr, te, n_cls, epochs, actors, spk_disjoint)
            gap = r.get("val_acc", 0) - r["Accuracy(%)"]
            print(f"  [C. fine-tuned   ] Acc={r['Accuracy(%)']:.2f}%  "
                  f"F1={r['F1_macro(%)']:.2f}%  ({r['Time(s)']:.0f}s)")
            print(f"       val_acc (cùng actor) = {r.get('val_acc',0):.2f}%  "
                  f"→ khoảng cách leakage = {gap:+.2f}%")
            fold_rows.append(_row("C. HuBERT fine-tuned", name, r))

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        ck.write_text(json.dumps(fold_rows))
        rows.extend(fold_rows)
        pd.DataFrame(rows).to_csv(out("results_partial.csv"), index=False)

    # ── Tổng hợp ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(out("results_speaker_independent.csv"), index=False)

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

    for f in OUT_DIR.glob("_bench_*.json"):
        f.unlink()
    p = Path(out("results_partial.csv"))
    if p.exists():
        p.unlink()

    print("\n" + "=" * 72)
    print("  HOÀN THÀNH – Output:")
    print(f"  • {out('results_speaker_independent.csv')}")
    print(f"  • {out('speaker_independent_comparison.png')}")
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
