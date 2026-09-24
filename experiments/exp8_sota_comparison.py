"""Exp8: SOTA comparison — reported numbers + re-implementation where feasible."""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
from data import RavdessDataset
from protocols import get_stratified_cv_splits, get_losgo_splits, verify_losgo_splits
from models.dl_1d import CNN1D, LSTMClassifier
from models.dl_2d import CNN2D
from features import extract_mfcc, extract_logmel, build_feature_matrix, FEATURE_EXTRACTORS
from utils import (compute_metrics, ensure_dirs, Timer, set_seed, cleanup_gpu)

import librosa


# ─── Literature-reported results (Table for paper) ──────────────────────────

LITERATURE_RESULTS = pd.DataFrame([
    {"Method": "Wei et al. 2025 (CNN+Transformer)", "Protocol": "Random split",
     "Accuracy": 80.0, "Note": "Cross-attention fusion"},
    {"Method": "Bhanbhro et al. 2025 (CNN-LSTM)", "Protocol": "Random split",
     "Accuracy": None, "Note": "Attention-enhanced CNN-LSTM"},
    {"Method": "Waleed & Shaker 2025 (CNN)", "Protocol": "Random split",
     "Accuracy": None, "Note": "Multi-dataset CNN on MELD+RAVDESS"},
    {"Method": "Vijaya Bharathi & Nitnaware 2026 (Hopfield NN)", "Protocol": "Random split",
     "Accuracy": 90.0, "Note": "Lightweight Hopfield NN, >90% claimed"},
])


# ─── Re-implementable baselines (CNN1D, LSTM, CNN2D on MFCC/LogMel) ────────

def train_dl_model(model, train_loader, val_loader, device,
                   max_epochs=30, patience=6, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_f1, best_state, pat = 0.0, None, 0

    for epoch in range(max_epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                logits = model(X)
                preds.extend(logits.argmax(-1).cpu().numpy())
                labels.extend(y.numpy())
        m = compute_metrics(np.array(labels), np.array(preds))

        if m["f1_macro"] > best_f1:
            best_f1 = m["f1_macro"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    return model


def run_exp8(dataset: RavdessDataset = None, force: bool = False,
             quick: bool = False):
    set_seed(config.DEFAULT_SEED)
    ensure_dirs()

    out_csv = os.path.join(config.SAVE_DIR, "results_exp8_sota.csv")
    if os.path.exists(out_csv) and not force:
        print(f"[INFO] Exp8 already done: {out_csv}")
        return pd.read_csv(out_csv)

    if dataset is None:
        dataset = RavdessDataset(sr=config.SR_HANDCRAFTED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Extract MFCC features
    print("[INFO] Extracting MFCC features for DL baselines ...")
    X_mfcc, y_mfcc = build_feature_matrix(
        dataset.waveforms, dataset.labels,
        FEATURE_EXTRACTORS["MFCC"][0], FEATURE_EXTRACTORS["MFCC"][1],
        sr=config.SR_HANDCRAFTED,
    )

    # Extract LogMel spectrograms (2D) for CNN2D
    print("[INFO] Extracting LogMel spectrograms for CNN2D ...")
    spectrograms = []
    spec_labels = []
    for i, y in enumerate(dataset.waveforms):
        try:
            mel = librosa.feature.melspectrogram(
                y=y, sr=config.SR_HANDCRAFTED,
                n_mels=config.N_MELS, n_fft=config.N_FFT,
                hop_length=config.HOP_LENGTH,
            )
            log_mel = librosa.power_to_db(mel, ref=np.max)
            spectrograms.append(log_mel)
            spec_labels.append(dataset.labels[i])
        except Exception:
            pass
    # Pad spectrograms to same time length
    max_time = max(s.shape[1] for s in spectrograms)
    X_spec = np.zeros((len(spectrograms), 1, config.N_MELS, max_time), dtype=np.float32)
    for i, s in enumerate(spectrograms):
        X_spec[i, 0, :, :s.shape[1]] = s
    y_spec = np.array(spec_labels)

    all_rows = []

    # DL baselines under both protocols
    DL_MODELS = {
        "CNN1D+MFCC": (CNN1D, {"input_dim": 240}, X_mfcc, y_mfcc, "1d"),
        "LSTM+MFCC": (LSTMClassifier, {"input_dim": 240}, X_mfcc, y_mfcc, "1d"),
        "CNN2D+LogMel": (CNN2D, {}, X_spec, y_spec, "2d"),
    }

    for protocol in ["5-fold CV", "LOSGO"]:
        if protocol == "5-fold CV":
            splits = get_stratified_cv_splits(y_mfcc)
        else:
            splits = get_losgo_splits(dataset.actors)

        max_folds = 1 if quick else len(splits)

        for model_name, (model_cls, kwargs, X, y, input_type) in DL_MODELS.items():
            for split in splits[:max_folds]:
                fold_name = split["fold_name"]

                if protocol == "LOSGO":
                    train_idx = split["train_idx"]
                    val_idx = split["val_idx"]
                    test_idx = split["test_idx"]
                else:
                    train_idx = split["train_idx"]
                    val_size = max(1, len(train_idx) // 5)
                    np.random.shuffle(train_idx)
                    val_idx = train_idx[:val_size]
                    train_idx = train_idx[val_size:]
                    test_idx = split["test_idx"]

                # Clamp indices
                max_idx = len(y) - 1
                train_idx = train_idx[train_idx <= max_idx]
                val_idx = val_idx[val_idx <= max_idx]
                test_idx = test_idx[test_idx <= max_idx]

                X_tr = torch.tensor(X[train_idx], dtype=torch.float32)
                y_tr = torch.tensor(y[train_idx], dtype=torch.long)
                X_val = torch.tensor(X[val_idx], dtype=torch.float32)
                y_val = torch.tensor(y[val_idx], dtype=torch.long)
                X_te = torch.tensor(X[test_idx], dtype=torch.float32)
                y_te = torch.tensor(y[test_idx], dtype=torch.long)

                train_dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)
                val_dl = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
                test_dl = DataLoader(TensorDataset(X_te, y_te), batch_size=32)

                model = model_cls(**kwargs).to(device)
                max_ep = 2 if quick else 30
                model = train_dl_model(model, train_dl, val_dl, device,
                                       max_epochs=max_ep, patience=6)

                model.eval()
                preds, labels = [], []
                with torch.no_grad():
                    for Xb, yb in test_dl:
                        Xb = Xb.to(device)
                        logits = model(Xb)
                        preds.extend(logits.argmax(-1).cpu().numpy())
                        labels.extend(yb.numpy())

                metrics = compute_metrics(np.array(labels), np.array(preds))
                row = {
                    "Experiment": "Exp8", "Model": model_name,
                    "Protocol": protocol, "Fold": fold_name,
                    "Seed": config.DEFAULT_SEED, **metrics,
                    "Source": "re-implemented",
                }
                all_rows.append(row)
                print(f"  {model_name} {protocol} {fold_name}: Acc={metrics['accuracy']:.1f}%")

                del model
                cleanup_gpu()

    df = pd.DataFrame(all_rows)

    # Append literature results
    for _, lit in LITERATURE_RESULTS.iterrows():
        df = pd.concat([df, pd.DataFrame([{
            "Experiment": "Exp8", "Model": lit["Method"],
            "Protocol": lit["Protocol"], "Fold": "reported",
            "Seed": None, "accuracy": lit["Accuracy"],
            "Source": "literature",
        }])], ignore_index=True)

    df.to_csv(out_csv, index=False)
    print(f"\n[INFO] Saved {out_csv}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run_exp8(force=args.force, quick=args.quick)
