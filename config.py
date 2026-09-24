"""Central configuration for all xHuBERT experiments."""

import os
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
RAVDESS_ROOT = os.environ.get("RAVDESS_ROOT", "./RAVDESS")
SAVE_DIR = os.environ.get("XHUBERT_SAVE_DIR", "./results")
CKPT_DIR = os.path.join(SAVE_DIR, "checkpoints")

# ─── Dataset ─────────────────────────────────────────────────────────────────
NUM_ACTORS = 24
NUM_EMOTIONS = 8
EMOTION_LABELS = {
    1: "neutral", 2: "calm", 3: "happy", 4: "sad",
    5: "angry", 6: "fearful", 7: "disgust", 8: "surprised",
}
EMOTION_NAMES = list(EMOTION_LABELS.values())

# ─── Audio ───────────────────────────────────────────────────────────────────
SR_HANDCRAFTED = 22_050
SR_HUBERT = 16_000
DURATION_SEC = 3.0
MAX_SAMPLES_HANDCRAFTED = int(SR_HANDCRAFTED * DURATION_SEC)  # 66150
MAX_SAMPLES_HUBERT = int(SR_HUBERT * DURATION_SEC)            # 48000

# ─── Feature extraction ─────────────────────────────────────────────────────
N_MFCC = 40
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
N_LPCC = 14

# ─── ML classifiers ─────────────────────────────────────────────────────────
SVM_PARAMS = {"C": 10, "gamma": "scale", "kernel": "rbf", "probability": True}
RF_PARAMS = {"n_estimators": 200, "n_jobs": -1, "random_state": 42}
KNN_PARAMS = {"n_neighbors": 5, "metric": "euclidean"}

# ─── xHuBERT fine-tuning (Table 1 of paper) ─────────────────────────────────
PRETRAINED_MODEL = "facebook/hubert-base-ls960"
BACKBONE_LR = 3e-5
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
MAX_EPOCHS = 30
PATIENCE = 6
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
SPECAUGMENT_PROB = 0.03
SPECAUGMENT_LENGTH = 10
LABEL_SMOOTHING = 0.05
DROPOUT = 0.1
EMBEDDING_DIM = 256
FREEZE_CNN = True
FREEZE_TRANSFORMER_LAYERS = 0
GRADIENT_CLIP = 1.0
WARMUP_FRACTION = 0.1

# ─── Fusion ──────────────────────────────────────────────────────────────────
FUSION_EPOCHS = 80
FUSION_PATIENCE = 12
FUSION_LR = 1e-3
GATE_ENTROPY_LAMBDA = 0.5

# ─── GRL / Speaker-adversarial ───────────────────────────────────────────────
AUG_SPEED_FACTORS = [0.9, 1.1]
AUG_GAIN_DB = 6.0
AUG_SHIFT_FRACTION = 0.1
AUG_NOISE_SNR_RANGE = (15, 30)
AUG_PROBS = {"speed": 0.5, "gain": 0.5, "shift": 0.5, "noise": 0.3}

# ─── Reproducibility ────────────────────────────────────────────────────────
SEEDS = [42, 43, 44]
DEFAULT_SEED = 42

# ─── Output ──────────────────────────────────────────────────────────────────
FIGURE_DPI = 150

# ─── Colab environment versions (from pip freeze) ───────────────────────────
TORCH_VERSION = "2.11.0+cu128"
TRANSFORMERS_VERSION = "4.51.3"
