"""Utilities: pos_conv_embed fix, seeding, timing, checkpoint management."""

import os
import gc
import json
import time
import random
import numpy as np
import torch
from pathlib import Path

import config


# ─── Reproducibility ────────────────────────────────────────────────────────

def set_seed(seed: int = config.DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── pos_conv_embed Fix ─────────────────────────────────────────────────────

def fix_pos_conv_weight_norm(model) -> None:
    """
    PyTorch >= 2.1 changed weight_norm API. HuggingFace transformers (up to
    v4.51) doesn't remap checkpoint keys for pos_conv_embed.conv, causing
    random initialization instead of pretrained weights.
    """
    from huggingface_hub import hf_hub_download

    pos = model.encoder.pos_conv_embed.conv
    has_parametrizations = hasattr(pos, "parametrizations")

    ckpt_path = hf_hub_download(
        config.PRETRAINED_MODEL, "pytorch_model.bin"
    )
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    g_key = "encoder.pos_conv_embed.conv.weight_g"
    v_key = "encoder.pos_conv_embed.conv.weight_v"

    if g_key not in sd:
        print("[FIX] Checkpoint has no weight_g/weight_v — may be fixed in newer version")
        return

    if has_parametrizations:
        with torch.no_grad():
            pos.parametrizations.weight.original0.copy_(sd[g_key])
            pos.parametrizations.weight.original1.copy_(sd[v_key])
        print("[FIX] Restored pos_conv_embed weights (parametrizations API)")
    else:
        with torch.no_grad():
            pos.weight_g.copy_(sd[g_key])
            pos.weight_v.copy_(sd[v_key])
        print("[FIX] Restored pos_conv_embed weights (legacy API)")


def load_hubert_model(output_hidden_states: bool = True):
    """Load HuBERT with pos_conv fix applied."""
    from transformers import HubertModel, HubertConfig

    cfg = HubertConfig.from_pretrained(config.PRETRAINED_MODEL)
    cfg.output_hidden_states = output_hidden_states
    model = HubertModel.from_pretrained(config.PRETRAINED_MODEL, config=cfg)
    fix_pos_conv_weight_norm(model)
    return model


# ─── Timing ──────────────────────────────────────────────────────────────────

class Timer:
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), elapsed % 60
        print(f"[TIME] {self.name}: {h}h {m}m {s:.1f}s")


# ─── Checkpoint / Results ────────────────────────────────────────────────────

def ensure_dirs():
    for d in [config.SAVE_DIR, config.CSV_DIR, config.FIG_DIR,
              config.CKPT_DIR, config.EMB_DIR, config.LOG_DIR]:
        os.makedirs(d, exist_ok=True)


def save_fold_result(result: dict, exp_name: str, fold: int, seed: int = 42):
    ensure_dirs()
    path = os.path.join(config.LOG_DIR, f"{exp_name}_fold{fold}_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=_json_default)
    print(f"[INFO] Saved {path}")


def load_fold_result(exp_name: str, fold: int, seed: int = 42):
    path = os.path.join(config.LOG_DIR, f"{exp_name}_fold{fold}_seed{seed}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def fold_completed(exp_name: str, fold: int, seed: int = 42) -> bool:
    return load_fold_result(exp_name, fold, seed) is not None


def save_checkpoint(state_dict: dict, exp_name: str, fold: int, seed: int = 42):
    ensure_dirs()
    path = os.path.join(config.CKPT_DIR, f"{exp_name}_fold{fold}_seed{seed}.pt")
    torch.save(state_dict, path)
    print(f"[INFO] Checkpoint saved: {path}")


def load_checkpoint(exp_name: str, fold: int, seed: int = 42):
    path = os.path.join(config.CKPT_DIR, f"{exp_name}_fold{fold}_seed{seed}.pt")
    if os.path.exists(path):
        return torch.load(path, map_location="cpu", weights_only=True)
    return None


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    all_labels = list(range(config.NUM_EMOTIONS))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)) * 100,
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=all_labels, zero_division=0)) * 100,
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", labels=all_labels, zero_division=0)) * 100,
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", labels=all_labels, zero_division=0)) * 100,
    }


def compute_per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import f1_score
    all_labels = list(range(config.NUM_EMOTIONS))
    f1s = f1_score(y_true, y_pred, average=None, labels=all_labels, zero_division=0)
    return {config.EMOTION_NAMES[i]: float(f1s[i]) * 100 for i in range(len(f1s))}
