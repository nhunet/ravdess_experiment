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

CHECKPOINT BỀN VỮNG (không cần merge thủ công)
  Mỗi (config × fold) được lưu vĩnh viễn dưới dạng _adv_<config>_<G>.json
  trong OUT_DIR. Chạy từng cấu hình riêng biệt LUÔN cho kết quả tổng hợp
  đầy đủ vì script quét lại toàn bộ checkpoint mỗi lần chạy. Ví dụ:

    # ngày 1
    python speaker_adversarial.py --configs base aug
    # ngày 2 – results_adversarial.csv sẽ chứa cả 3 config, không phải chỉ grl
    python speaker_adversarial.py --configs grl
    # ngày 3 – results_adversarial.csv sẽ chứa đủ 4 config
    python speaker_adversarial.py --configs aug+grl

  --force: chỉ xoá checkpoint của các config được chỉ định trong lần chạy đó,
           KHÔNG đụng các config khác đã có trên disk.

MULTI-SEED SWEEP (bắt buộc cho paper)
  n=6 fold × 1 seed → statistical power thấp, không phát hiện được Δ ≤ 3%.
  Chạy 3 seed để có 3 × 6 = 18 giá trị/config → power đủ cho Δ ≥ 1%.

  Dùng SEED env var (script tự set numpy + torch seed từ đây):
    SEED=42 python speaker_adversarial.py                # seed mặc định (đã chạy)
    SEED=43 OUT_DIR=./out_seed43 python speaker_adversarial.py
    SEED=44 OUT_DIR=./out_seed44 python speaker_adversarial.py

  QUAN TRỌNG: mỗi seed dùng OUT_DIR RIÊNG — không lẫn checkpoint. Sau đó dùng
  script gộp riêng để tính stats trên toàn bộ 72 điểm dữ liệu (3 seed × 4
  config × 6 fold).

VÁ LỖI KỸ THUẬT (bản này, quan trọng cho paper)
  BUG 1 (num_workers augmentation không sinh random khác nhau):
    Bản trước dùng self.rng khởi tạo một lần trong __init__. Với num_workers=2,
    2 worker fork dataset → cùng seed → cùng chuỗi random → augmentation giả
    tính đa dạng. Bản này chuyển sang worker-safe RNG (mỗi __getitem__ tạo
    RandomState mới từ worker_seed + index).
  BUG 2 (peak normalization vô hiệu hoá gain augmentation):
    Bản trước rescale x /= peak nếu peak > 1 → gain +6dB (peak x2) bị chia
    lại → gain gần như không có tác dụng. Bản này dùng np.clip(-1, 1) —
    giữ hiệu ứng gain nhưng không tràn số.
  BUG 3 (adv_head_first không log):
    Chỉ có adv_head_last → không biết head có bao giờ học được không. Bản
    này log first/max/last + cảnh báo nếu max < chance+5 (head không học nổi
    → GRL không có tín hiệu đảo).
  NÂNG CẤP: speaker probe giờ có LR + MLP + permutation test (n_repeat=5,
  n_perm=50) → probe finding defensible về methodology.
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
    _evaluate, RANDOM_SEED as _DEFAULT_SEED, BATCH_SIZE, GRAD_ACCUM,
    LR_BACKBONE, LR_HEAD, WEIGHT_DECAY, WARMUP_RATIO, LABEL_SMOOTH,
    TARGET_SR, MAX_SAMPLES, EMBED_DIM,
)
from speaker_independent_benchmark import (
    losgo_splits, make_val_split, _metrics, paired_tests,
)

OUT_DIR = Path(os.environ.get("OUT_DIR", "./outputs_adversarial"))

# ── Multi-seed support cho paper ────────────────────────────────────────────
# Set SEED env var để override default seed từ hubert_finetune. Mỗi seed
# nên có OUT_DIR riêng để checkpoint không lẫn nhau. Xem docstring đầu file.
RANDOM_SEED = int(os.environ.get("SEED", _DEFAULT_SEED))
if RANDOM_SEED != _DEFAULT_SEED:
    print(f"[SEED] Override từ env: SEED={RANDOM_SEED} "
          f"(default: {_DEFAULT_SEED})")


def out(f: str) -> str:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return str(OUT_DIR / f)


def _infer_seed(path: Path, default: int) -> int:
    """
    Suy ra seed cho các checkpoint CŨ (ghi trước khi có trường "Seed").

    Quy ước của run_seed_sweep.sh: mỗi seed có thư mục riêng dạng
    .../outputs_speaker_adversarial/seed_43/_adv_base_G1.json
    → lấy số từ tên thư mục. Không khớp thì dùng `default` (seed hiện tại).

    Dùng re.search chứ KHÔNG phải fullmatch: docstring đầu file hướng dẫn
    cả dạng OUT_DIR=./out_seed43, mà fullmatch thì trượt tên đó.
    Lấy thư mục GẦN file nhất trở ra để tên cha không ghi đè tên con.
    """
    import re
    for part in reversed(path.resolve().parts):
        m = re.search(r"seed[_-]?(\d+)", part, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return default


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

    ⚠️ WORKER-SAFE RNG (BUG FIX quan trọng):
      Bản trước dùng self.rng = np.random.RandomState(RANDOM_SEED) khởi tạo
      MỘT LẦN trong __init__. Với num_workers=2 (bạn đang dùng trong Colab),
      DataLoader FORK dataset cho mỗi worker → mỗi worker có bản sao rng với
      CÙNG seed → cả 2 worker sinh CÙNG chuỗi số ngẫu nhiên → augmentation
      lặp lại đúng những mẫu như nhau, không đa dạng như tưởng.

      Bản này tạo rng MỚI mỗi __getitem__ với seed từ (worker_id, index, os pid,
      thời gian cao độ chính xác) → mỗi call độc lập, mỗi worker khác nhau.
    """

    def __init__(self, waves: np.ndarray, y: np.ndarray,
                 spk: np.ndarray | None = None, augment: bool = False):
        self.waves = waves
        self.y = y
        self.spk = spk
        self.augment = augment
        # KHÔNG khởi tạo self.rng ở đây — làm trong __getitem__ để worker-safe.

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

    def _apply(self, x: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        # 1. Speed perturbation (Ko et al. 2015) – quan trọng nhất
        if rng.rand() < 0.5:
            x = self._speed_perturb(x, rng.choice([0.9, 1.1]))

        # 2. Time shift ±10% – chống bám vị trí tuyệt đối
        if rng.rand() < 0.5:
            s = int(rng.uniform(-0.1, 0.1) * MAX_SAMPLES)
            x = np.roll(x, s)

        # 3. Gain ±6 dB – mô phỏng micro/khoảng cách khác nhau
        # ⚠️ BUG FIX: bản trước NORMALIZE peak về 1.0 sau tất cả augmentation
        # → gain +6dB (biên độ x2, peak > 1.0) bị chia lại → gain augmentation
        # gần như bị vô hiệu hoá cho các mẫu có peak ≈ 1 (mà load_ravdess_16k
        # đã normalize sẵn về peak = 1). Chỉ gain âm mới thực sự có hiệu lực.
        # → Sửa: CLIP mẫu về [-1, 1] khi cần (thay vì rescale toàn bộ),
        # để gain thực sự thay đổi mức nghe được mà không huỷ tín hiệu.
        if rng.rand() < 0.5:
            x = x * float(10 ** (rng.uniform(-6, 6) / 20))

        # 4. Gaussian noise SNR 15-30 dB – môi trường thu khác nhau
        if rng.rand() < 0.3:
            snr = rng.uniform(15, 30)
            p_sig = np.mean(x ** 2)
            if p_sig > 0:
                p_noise = p_sig / (10 ** (snr / 10))
                x = x + rng.randn(len(x)).astype(np.float32) * np.sqrt(p_noise)

        # Clip thay vì rescale — giữ được hiệu ứng gain nhưng không tràn số
        return np.clip(x, -1.0, 1.0).astype(np.float32)

    def __getitem__(self, i):
        x = self.waves[i]
        if self.augment:
            # Worker-safe: seed rng mỗi call từ nguồn entropy độc lập
            # (worker_id đã được PyTorch set trong worker_init_fn bên dưới)
            info = torch.utils.data.get_worker_info()
            worker_seed = info.seed if info is not None else 0
            rng = np.random.RandomState((worker_seed + i) % (2**32 - 1))
            x = self._apply(x.copy(), rng)
        out_ = [torch.from_numpy(x), torch.tensor(self.y[i], dtype=torch.long)]
        if self.spk is not None:
            out_.append(torch.tensor(self.spk[i], dtype=torch.long))
        return tuple(out_)


def _worker_init_fn(worker_id: int) -> None:
    """
    PyTorch DataLoader worker_init: đảm bảo mỗi worker có seed khác nhau.
    Kết hợp với AugmentedDataset.__getitem__ dùng worker_info.seed, mỗi worker
    và mỗi index sinh chuỗi ngẫu nhiên độc lập → augmentation thật sự đa dạng.
    """
    # base_seed đến từ generator của DataLoader; unique per worker.
    seed = (torch.initial_seed() + worker_id) % (2**32 - 1)
    np.random.seed(seed)


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


def speaker_probe(model, waves, actors, idx,
                  n_repeat: int = 5, n_perm: int = 50) -> dict:
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

    NÂNG CẤP METHODOLOGY (defensible cho paper):
      Bản trước dùng CV 3-fold + LR probe duy nhất → 2 điểm attackable:

      (a) LR probe chỉ đo tách tuyến tính. Nếu identity encode phi tuyến,
          LR sẽ under-estimate → THÊM MLP probe để cross-check.

      (b) Không có phân phối null → không biết probe accuracy 12% "thật sự
          cao" so với ngẫu nhiên bao nhiêu → THÊM PERMUTATION TEST: shuffle
          nhãn speaker n_perm lần, đo probe → có phân phối null empirical.

      (c) CV 3-fold với ~40 mẫu/speaker cho std lớn → THÊM n_repeat lần
          hold-out (train/test 70/30 stratified) với seed khác nhau → có std
          probe accuracy → kiểm định paired vs base defensible hơn.

    Trả về dict:
      lr_mean, lr_std       : probe tuyến tính (Logistic Regression), mean±std
      mlp_mean, mlp_std     : probe phi tuyến (MLP 2-layer)
      chance                : 100/n_speakers
      perm_p_lr, perm_p_mlp : p-value permutation test (nhỏ = probe > ngẫu nhiên
                              có ý nghĩa thống kê)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split as _tts
    from sklearn.preprocessing import StandardScaler

    E = _extract_emb(model, waves, idx)
    spk = actors[idx]
    n_spk = len(np.unique(spk))
    chance = 100.0 / n_spk
    if n_spk < 2:
        return dict(lr_mean=float("nan"), lr_std=float("nan"),
                    mlp_mean=float("nan"), mlp_std=float("nan"),
                    chance=chance, perm_p_lr=float("nan"),
                    perm_p_mlp=float("nan"))

    sc = StandardScaler()
    E_std = sc.fit_transform(E)

    def _probe_once(clf_ctor, X, y, seed):
        Xtr, Xte, ytr, yte = _tts(X, y, test_size=0.3,
                                  stratify=y, random_state=seed)
        c = clf_ctor()
        c.fit(Xtr, ytr)
        return c.score(Xte, yte) * 100

    def _lr():
        # sklearn 1.7+ bỏ 'multi_class' (auto), 1.8+ bỏ 'n_jobs' (không dùng).
        # Truyền chỉ max_iter → forward-compatible với sklearn 1.5-1.10+
        return LogisticRegression(max_iter=2000)
    def _mlp():
        # MLP 2 hidden layer — bắt được identity phi tuyến nếu có
        return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                             early_stopping=True, random_state=0)

    # Repeated hold-out cho probe accuracy có std
    lr_scores = np.array([_probe_once(_lr, E_std, spk, s) for s in range(n_repeat)])
    mlp_scores = np.array([_probe_once(_mlp, E_std, spk, s) for s in range(n_repeat)])

    # Permutation test — shuffle spk labels để có phân phối null
    # (n_perm hơi ít vì tốn thời gian, nhưng đủ để có p ~0.02 resolution)
    rng = np.random.RandomState(0)
    null_lr = []
    null_mlp = []
    for k in range(n_perm):
        spk_shuf = rng.permutation(spk)
        # 1 hold-out mỗi permutation là đủ (n_perm là số lần lặp chính)
        null_lr.append(_probe_once(_lr, E_std, spk_shuf, k))
        null_mlp.append(_probe_once(_mlp, E_std, spk_shuf, k))
    null_lr = np.array(null_lr)
    null_mlp = np.array(null_mlp)

    # p-value: xác suất probe accuracy trên nhãn shuffle ≥ probe thực
    # (+1/(n+1) là smoothing chuẩn để tránh p=0)
    lr_obs, mlp_obs = lr_scores.mean(), mlp_scores.mean()
    p_lr = (np.sum(null_lr >= lr_obs) + 1) / (n_perm + 1)
    p_mlp = (np.sum(null_mlp >= mlp_obs) + 1) / (n_perm + 1)

    return dict(
        lr_mean=float(lr_obs), lr_std=float(lr_scores.std(ddof=1)),
        mlp_mean=float(mlp_obs), mlp_std=float(mlp_scores.std(ddof=1)),
        chance=float(chance),
        perm_p_lr=float(p_lr), perm_p_mlp=float(p_mlp),
        null_lr_mean=float(null_lr.mean()), null_lr_std=float(null_lr.std()),
        null_mlp_mean=float(null_mlp.mean()), null_mlp_std=float(null_mlp.std()),
    )


# ============================================================================
# PHẦN 3: HUẤN LUYỆN 1 FOLD, 1 CẤU HÌNH
# ============================================================================

def train_config(waves, y, actors, tr, te, n_cls, cfg: dict,
                 epochs: int, fold: str, cfg_name: str) -> dict:
    set_seed(RANDOM_SEED)
    t0 = time.time()
    use_aug, use_grl = cfg["aug"], cfg["grl"]

    # VÁ LỖI MULTI-SEED: truyền seed vào để mỗi seed chọn bộ 4 diễn viên
    # validation KHÁC nhau. Trước đây make_val_split dùng cứng RANDOM_SEED=42
    # của hubert_finetune nên 3 seed dùng chung một val split → variance bị
    # đánh giá thấp. Với SEED=42 kết quả không đổi (RandomState(42) như cũ).
    real_tr, real_val = make_val_split(tr, y, actors, speaker_disjoint=True,
                                       seed=RANDOM_SEED)

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
    # worker_init_fn quan trọng cho tr_dl khi use_aug=True: đảm bảo 2 worker sinh
    # augmentation độc lập, không lặp lại (xem AugmentedDataset docstring).
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                       drop_last=True, worker_init_fn=_worker_init_fn, **kw)
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
    # In cả first và last để đánh giá: nếu first ≈ chance ≈ last thì speaker head
    # CHƯA BAO GIỜ học được tách speaker, tức GRL không có tín hiệu để đảo →
    # cả cơ chế không hoạt động. Đây là insight quan trọng để viết Discussion:
    # phân biệt "GRL không hoạt động nói chung" vs "dataset RAVDESS quá nhỏ
    # để speaker head học được đủ trước khi λ tăng lên".
    if use_grl and spk_acc_hist:
        r["adv_head_first"] = float(spk_acc_hist[0])
        r["adv_head_last"] = float(spk_acc_hist[-1])
        r["adv_head_max"] = float(max(spk_acc_hist))
        chance_spk = 100.0 / n_spk if n_spk else 0.0
        print(f"    [{fold}|{cfg_name}] adv-head accuracy: "
              f"first={spk_acc_hist[0]:.1f}%  "
              f"max={max(spk_acc_hist):.1f}%  "
              f"last={spk_acc_hist[-1]:.1f}%  "
              f"(chance={chance_spk:.1f}% với N={n_spk} speakers)")

    # ── ĐO TRUNG THỰC: probe tuyến tính + phi tuyến trên embedding đóng băng ──
    # Đây mới là bằng chứng thật. Head đối kháng tụt accuracy KHÔNG chứng minh
    # gì cả — nó bị GRL đánh bại nên đương nhiên phải tụt.
    #
    # Ghi cả LR probe (tuyến tính) và MLP probe (phi tuyến) + permutation p-value:
    # - LR probe cao → identity tách tuyến tính được (leakage rõ ràng)
    # - MLP probe > LR probe → identity encode phi tuyến, LR under-estimate
    # - perm_p thấp → probe accuracy KHÁC BIỆT có ý nghĩa với ngẫu nhiên
    try:
        pr = speaker_probe(model, waves, actors, real_tr,
                           n_repeat=5, n_perm=50)
        r["spk_probe_lr(%)"]      = pr["lr_mean"]
        r["spk_probe_lr_std(%)"]  = pr["lr_std"]
        r["spk_probe_mlp(%)"]     = pr["mlp_mean"]
        r["spk_probe_mlp_std(%)"] = pr["mlp_std"]
        r["spk_chance(%)"]        = pr["chance"]
        r["perm_p_lr"]            = pr["perm_p_lr"]
        r["perm_p_mlp"]           = pr["perm_p_mlp"]
        r["null_lr_mean(%)"]      = pr["null_lr_mean"]
        r["null_mlp_mean(%)"]     = pr["null_mlp_mean"]

        # Giữ tên cũ "spk_probe(%)" (= LR probe mean) cho backward compat
        # với stats code + notebook đã có
        r["spk_probe(%)"] = pr["lr_mean"]

        adv = ""
        if use_grl and spk_acc_hist:
            adv = f"  [head đối kháng: {spk_acc_hist[-1]:.1f}%]"
        print(f"    [{fold}|{cfg_name}] speaker-PROBE:")
        print(f"      LR:  {pr['lr_mean']:5.1f}±{pr['lr_std']:.1f}%  "
              f"(null: {pr['null_lr_mean']:.1f}%,  perm-p={pr['perm_p_lr']:.4f})")
        print(f"      MLP: {pr['mlp_mean']:5.1f}±{pr['mlp_std']:.1f}%  "
              f"(null: {pr['null_mlp_mean']:.1f}%, perm-p={pr['perm_p_mlp']:.4f})")
        print(f"      chance={pr['chance']:.1f}%{adv}")
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
# KIỂM ĐỊNH THỐNG KÊ ĐA-METRIC (local — không phụ thuộc paired_tests import)
# ============================================================================

def _paired_tests_local(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """
    Paired t-test + Wilcoxon + Cohen's d cho mọi cặp cấu hình, trên một metric.

    Local implementation — không phụ thuộc `paired_tests` từ speaker_independent_benchmark,
    vì hàm đó chỉ hỗ trợ Accuracy. Ở đây cần chạy cho cả Accuracy, F1 VÀ speaker-probe
    (speaker-probe là finding chính của cải tiến #3, cần kiểm định thống kê riêng).

    ── VÁ LỖI ĐA SEED ───────────────────────────────────────────────────────
    Bản trước dùng df.pivot(index="Fold", ...). Khi dữ liệu chứa nhiều seed,
    mỗi Fold xuất hiện nhiều lần → pandas ném ValueError "Index contains
    duplicate entries, cannot reshape" và toàn bộ phần thống kê chết.
    Giờ ghép cặp theo (Seed, Fold) khi có cột Seed → n = n_seed × n_fold.
    """
    from scipy import stats as _stats

    # Đơn vị ghép cặp: một lần train = (Seed, Fold). Không có cột Seed
    # (checkpoint cũ) thì rơi về Fold như trước.
    idx_cols = ["Seed", "Fold"] if "Seed" in df.columns else ["Fold"]
    piv = df.pivot_table(index=idx_cols, columns="Config",
                         values=metric_col, aggfunc="mean")
    piv = piv.reindex(columns=[c for c in CONFIGS if c in piv.columns])
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
            try:
                _, p_t = _stats.ttest_rel(b, a)
            except Exception:
                p_t = float("nan")
            try:
                _, p_w = _stats.wilcoxon(b, a)
            except Exception:
                p_w = float("nan")
            sd = float(diff.std(ddof=1))
            d_val = float(diff.mean()) / sd if sd > 0 else 0.0
            rows.append({
                "So sánh":     f"{a_name} vs {b_name}",
                "Δ mean":      round(float(diff.mean()), 2),
                "t-test p":    round(float(p_t), 4) if not np.isnan(p_t) else np.nan,
                "Wilcoxon p":  round(float(p_w), 4) if not np.isnan(p_w) else np.nan,
                "Cohen_d":     round(d_val, 2),
                "n_fold":      int(len(diff)),
                "Kết luận":    ("có ý nghĩa (p<0.05)" if not np.isnan(p_t) and p_t < 0.05
                                else "KHÔNG đủ bằng chứng"),
            })
    return pd.DataFrame(rows)


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

    # ── Nếu --force: chỉ xoá checkpoint của các config được yêu cầu train lại ──
    # (KHÔNG đụng đến checkpoint của các config KHÁC — giữ nguyên dữ liệu cũ.)
    if args.force:
        for cfg_name in args.configs:
            for f in OUT_DIR.glob(f"_adv_{cfg_name}_*.json"):
                f.unlink()
                print(f"  [FORCE] Đã xoá checkpoint cũ: {f.name}")

    for name, tr, te, g in splits:
        print(f"\n{'─'*68}\n  {name}  |  VRAM: {gpu_mem()}\n{'─'*68}")

        for cfg_name in args.configs:
            ck = OUT_DIR / f"_adv_{cfg_name}_{name.split()[0]}.json"
            if ck.exists():
                s = json.loads(ck.read_text())
                print(f"  [RESUME] {cfg_name} @ {name} → bỏ qua "
                      f"({s['row']['Accuracy(%)']:.2f}%)")
                continue

            r = train_config(waves, y, actors, tr, te, n_cls,
                             CONFIGS[cfg_name], epochs, name, cfg_name)

            row = {
                "Config":       cfg_name,
                "Fold":         name,
                # Ghi seed vào từng checkpoint để merge_seeds.py ghép cặp đúng
                # (Seed, Fold) khi gộp nhiều seed. Checkpoint CŨ không có
                # trường này — được backfill khi quét, xem _infer_seed().
                "Seed":         RANDOM_SEED,
                "Accuracy(%)":  round(r["Accuracy(%)"], 2),
                "F1_macro(%)":  round(r["F1_macro(%)"], 2),
                "Precision(%)": round(r["Precision(%)"], 2),
                "Recall(%)":    round(r["Recall(%)"], 2),
                "Time(s)":      round(r["Time(s)"], 1),
            }
            # Metrics mới từ probe nâng cấp (LR + MLP + permutation test) và
            # từ adversarial head (giờ có cả first/max/last để phát hiện
            # "head không bao giờ học được")
            for k in ("spk_probe(%)", "spk_probe_lr(%)", "spk_probe_lr_std(%)",
                      "spk_probe_mlp(%)", "spk_probe_mlp_std(%)",
                      "spk_chance(%)", "perm_p_lr", "perm_p_mlp",
                      "null_lr_mean(%)", "null_mlp_mean(%)",
                      "adv_head_first", "adv_head_last", "adv_head_max"):
                if k in r:
                    row[k] = round(r[k], 4) if "perm_p" in k else round(r[k], 2)

            print(f"  [{cfg_name:8s}] Acc={r['Accuracy(%)']:.2f}%  "
                  f"F1={r['F1_macro(%)']:.2f}%  ({r['Time(s)']:.0f}s)")

            OUT_DIR.mkdir(parents=True, exist_ok=True)
            ck.write_text(json.dumps({
                "row": row,
                "y_true": r["y_true"].tolist(),
                "y_pred": r["y_pred"].tolist(),
            }))

    # ── TỔNG HỢP: quét TẤT CẢ checkpoint _adv_*.json (mọi config từng chạy) ──
    # Đây là chỗ then chốt: không dùng biến `rows` local nữa. Mỗi lần chạy,
    # script đọc lại toàn bộ dữ liệu bền vững trên disk, nên bảng tổng hợp và
    # thống kê LUÔN đầy đủ 4 cấu hình dù bạn chạy từng cấu hình riêng biệt.
    rows = []
    preds: dict[str, dict[str, list]] = {}
    all_ckpts = sorted(OUT_DIR.glob("_adv_*.json"))
    for ck in all_ckpts:
        s = json.loads(ck.read_text())
        cfg = s["row"]["Config"]
        # Backfill cho checkpoint ghi trước khi có trường "Seed" (ví dụ lần
        # chạy seed 42 đang diễn ra) — không phải chạy lại gì cả.
        s["row"].setdefault("Seed", _infer_seed(ck, RANDOM_SEED))
        rows.append(s["row"])
        preds.setdefault(cfg, {"true": [], "pred": []})
        preds[cfg]["true"].extend(s["y_true"])
        preds[cfg]["pred"].extend(s["y_pred"])

    df = pd.DataFrame(rows)
    if df.empty:
        print("\n[WARN] Chưa có checkpoint nào — bỏ qua bước tổng hợp.")
        return

    df = df.sort_values(["Config", "Fold"]).reset_index(drop=True)
    df.to_csv(out("results_adversarial.csv"), index=False)
    print(f"\n[INFO] Đã gộp {df['Config'].nunique()} cấu hình × "
          f"{df.groupby('Config').size().min()}-{df.groupby('Config').size().max()} fold "
          f"từ {len(all_ckpts)} checkpoint bền vững.")

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
        print("  Đo bằng PROBE MỚI trên embedding ĐÓNG BĂNG (LR + MLP + permutation).")
        print("  KHÔNG dùng accuracy của head đối kháng — head đó bị GRL đánh bại")
        print("  nên đương nhiên tụt, điều đó không chứng minh gì cả.")
        print("  Interpretation:")
        print("    • LR probe cao → identity tách được tuyến tính (leakage rõ)")
        print("    • MLP > LR → identity encode phi tuyến, LR under-estimate")
        print("    • perm-p thấp → probe khác biệt CÓ Ý NGHĨA vs shuffle nhãn\n")

        # Có thể có checkpoint cũ chưa có LR/MLP tách biệt — dùng .get() an toàn
        cols_needed = ["spk_probe_lr(%)", "spk_probe_mlp(%)",
                       "perm_p_lr", "perm_p_mlp", "spk_chance(%)"]
        has_new = all(c in df.columns for c in cols_needed)

        if has_new:
            pr = df.groupby("Config")[cols_needed].mean().round(3)
            base_probe = pr.loc["base", "spk_probe_lr(%)"] if "base" in pr.index else None
            chance = pr["spk_chance(%)"].mean()

            print(f"  {'Cấu hình':10s} {'LR probe':>12s} {'MLP probe':>12s} "
                  f"{'perm-p (LR)':>13s} {'perm-p (MLP)':>13s} {'Δ LR vs base':>13s}")
            print("  " + "─" * 78)
            for c in pr.index:
                v_lr = pr.loc[c, "spk_probe_lr(%)"]
                v_mlp = pr.loc[c, "spk_probe_mlp(%)"]
                p_lr = pr.loc[c, "perm_p_lr"]
                p_mlp = pr.loc[c, "perm_p_mlp"]
                d = (f"{v_lr - base_probe:+.2f}%"
                     if base_probe is not None and c != "base" else "—")
                print(f"  {c:10s} {v_lr:11.2f}% {v_mlp:11.2f}% "
                      f"{p_lr:13.4f} {p_mlp:13.4f} {d:>13s}")
            print(f"  {'(chance)':10s} {chance:11.2f}%")

            # Insight tự động
            if base_probe is not None and "grl" in pr.index:
                mlp_grl = pr.loc["grl", "spk_probe_mlp(%)"]
                lr_grl = pr.loc["grl", "spk_probe_lr(%)"]
                if mlp_grl > lr_grl + 5:
                    print(f"\n  → MLP probe của grl ({mlp_grl:.1f}%) > LR probe "
                          f"({lr_grl:.1f}%) → identity encode PHI TUYẾN")
                    print(f"    LR probe under-estimate leakage thật sự.")
        else:
            # Fallback cho checkpoint cũ chưa có LR/MLP tách biệt
            pr = df.groupby("Config")[["spk_probe(%)", "spk_chance(%)"]].mean().round(2)
            base_probe = pr.loc["base", "spk_probe(%)"] if "base" in pr.index else None
            chance = pr["spk_chance(%)"].mean()

            print(f"  {'Cấu hình':10s} {'speaker-probe':>15s} {'Δ vs base':>11s}")
            print("  " + "─" * 40)
            for c in pr.index:
                v = pr.loc[c, "spk_probe(%)"]
                d = f"{v - base_probe:+.2f}%" if base_probe is not None and c != "base" else "—"
                print(f"  {c:10s} {v:14.2f}% {d:>11s}")
            print(f"  {'(chance)':10s} {chance:14.2f}%")

        # ── Adversarial head: first vs max vs last ──────────────────────────
        # QUAN TRỌNG cho Discussion: nếu head KHÔNG BAO GIỜ học được tách speaker
        # (first ≈ max ≈ chance), GRL không có tín hiệu để đảo → cả cơ chế
        # không hoạt động, khác với "GRL không work nói chung".
        if "adv_head_first" in df.columns and df["adv_head_first"].notna().any():
            print("\n  ── Adversarial head accuracy (chỉ nhánh có GRL) ──")
            ah = df.dropna(subset=["adv_head_first"]).groupby("Config").agg(
                first=("adv_head_first", "mean"),
                max=("adv_head_max", "mean") if "adv_head_max" in df.columns else ("adv_head_last", "mean"),
                last=("adv_head_last", "mean"),
            ).round(2)
            print(f"  {'Cấu hình':10s} {'first':>10s} {'max':>10s} {'last':>10s}")
            print("  " + "─" * 42)
            for c in ah.index:
                print(f"  {c:10s} {ah.loc[c, 'first']:9.2f}% "
                      f"{ah.loc[c, 'max']:9.2f}% {ah.loc[c, 'last']:9.2f}%")

            # Diagnostic
            max_head = ah["max"].max()
            spk_chance_val = 100.0 / 16  # ~6.25% với 16 speakers
            if max_head < spk_chance_val + 5:
                print(f"\n  → CẢNH BÁO: speaker head KHÔNG BAO GIỜ đạt > chance "
                      f"({spk_chance_val:.1f}%+5).")
                print("    Head không học được tách speaker → GRL không có tín hiệu")
                print("    để đảo → cơ chế GRL KHÔNG HOẠT ĐỘNG trên dataset này.")
                print("    (Đây là insight cần đưa vào Discussion — khác với")
                print("     'GRL nói chung không work'.)")

    # ── Per-emotion + confusion ──────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("  PHÂN TÍCH PER-EMOTION")
    print("=" * 74)

    emo_rows, conf_rows = [], []
    all_cfgs_on_disk = [c for c in CONFIGS if c in preds]
    for cfg in all_cfgs_on_disk:
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
    # Chạy trên TOÀN BỘ config có trên disk, cho CẢ 3 metric: Accuracy, F1, speaker-probe.
    # Speaker-probe là finding chính của cải tiến #3 — bắt buộc kiểm định riêng.
    n_cfgs_on_disk = df["Config"].nunique()
    # Đơn vị ghép cặp là (Seed, Fold), không phải Fold — nếu thư mục này chứa
    # nhiều seed thì n lớn hơn số fold.
    _pair_cols = ["Seed", "Fold"] if "Seed" in df.columns else ["Fold"]
    n_pairs = len(df[_pair_cols].drop_duplicates())
    n_seeds = df["Seed"].nunique() if "Seed" in df.columns else 1
    if n_pairs >= 3 and n_cfgs_on_disk > 1:
        print("\n" + "=" * 74)
        print(f"  KIỂM ĐỊNH THỐNG KÊ (paired, n={n_pairs} = "
              f"{n_seeds} seed × {df['Fold'].nunique()} fold, "
              f"{n_cfgs_on_disk} cấu hình)")
        print("=" * 74)
        try:
            metrics_to_test = [
                ("Accuracy(%)",     "Accuracy"),
                ("F1_macro(%)",     "F1-macro"),
                ("spk_probe_lr(%)", "Speaker-probe (LR, tuyến tính)"),
                ("spk_probe_mlp(%)","Speaker-probe (MLP, phi tuyến)"),
                # Giữ cột cũ để backward compat với checkpoint đã có
                ("spk_probe(%)",    "Speaker-probe (legacy = LR)"),
            ]
            # Loại trùng: nếu cả spk_probe_lr(%) và spk_probe(%) đều tồn tại,
            # chúng bằng nhau — chỉ chạy 1 để không double-report
            if all(c in df.columns for c in ("spk_probe(%)", "spk_probe_lr(%)")):
                metrics_to_test = [m for m in metrics_to_test
                                   if m[0] != "spk_probe(%)"]
            all_tests = []
            for col, label in metrics_to_test:
                if col not in df.columns or df[col].isna().all():
                    continue
                st = _paired_tests_local(df, col)
                if st.empty:
                    continue
                st.insert(0, "Metric", label)
                all_tests.append(st)

                print(f"\n  ── {label} ──")
                print(st.drop(columns=["Metric"]).to_string(index=False))

                sig = st[st["t-test p"] < 0.05]
                if not sig.empty:
                    print(f"  → {len(sig)} cặp có ý nghĩa (p<0.05) trên {label}:")
                    for _, r in sig.iterrows():
                        d_col = r["Cohen_d"]
                        print(f"     • {r['So sánh']}: Δ={r['Δ mean']:+.2f}, "
                              f"p={r['t-test p']:.4f}, d={d_col:+.2f}")

            if all_tests:
                st_all = pd.concat(all_tests, ignore_index=True)
                st_all.to_csv(out("statistical_tests_adversarial.csv"), index=False)

            n_tests = sum(len(s) for s in all_tests)
            # p sàn của Wilcoxon = 2 / 2^n (two-sided). Với n=6 → 0.0312 (không
            # bao giờ < 0.05); với n=18 (3 seed) → ~7.6e-6, hết bị chặn.
            w_floor = 2.0 ** (-n_pairs) * 2
            print(f"\n  [LƯU Ý] n={n_pairs} → Wilcoxon p sàn lý thuyết = {w_floor:.2e}.")
            if w_floor > 0.05:
                print(f"           p sàn > 0.05 → Wilcoxon KHÔNG THỂ đạt ý nghĩa dù "
                      f"thắng {n_pairs}/{n_pairs}.")
                print(f"           Đây là giới hạn toán học của mẫu nhỏ, không phải "
                      f"bằng chứng chênh lệch là nhiễu.")
            print(f"           Ưu tiên đọc t-test + Cohen's d; Wilcoxon để tham khảo chéo.")
            if n_tests > 1:
                bonf = 0.05 / n_tests
                print(f"           Multiple comparisons: {n_tests} test tổng → "
                      f"Bonferroni threshold = 0.05/{n_tests} ≈ {bonf:.4f}")
        except ImportError:
            print("  [WARN] Cần scipy")

    plot_ablation(summ)

    # KHÔNG xoá _adv_*.json — đây là nguồn dữ liệu bền vững để tổng hợp các lần
    # chạy riêng biệt. Nếu muốn train lại từ đầu, dùng --force (chỉ xoá checkpoint
    # của các config bạn truyền qua --configs).

    print("\n" + "=" * 74)
    print("  HOÀN THÀNH – Output:")
    print(f"  • {out('results_adversarial.csv')}          (ablation, "
          f"{n_cfgs_on_disk} cấu hình)")
    print(f"  • {out('per_emotion_results.csv')}       (per-emotion)")
    print(f"  • {out('confusion_pairs.csv')}          (cặp nhầm lẫn)")
    print(f"  • {out('ablation_study.png')}")
    print(f"  • {out('per_emotion_f1.png')}")
    print(f"  • confusion_<config>.png")
    print(f"  • {len(all_ckpts)} file _adv_*.json (checkpoint bền vững – GIỮ LẠI)")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()
