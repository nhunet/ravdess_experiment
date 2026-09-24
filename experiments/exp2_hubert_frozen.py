"""Exp2: HuBERT frozen baseline — extract embeddings + SVM/RF."""

import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import config
from data import RavdessDataset
from protocols import get_stratified_cv_splits, get_losgo_splits, verify_losgo_splits
from models.ml_classifiers import get_svm_pipeline, get_rf_pipeline
from utils import (load_hubert_model, compute_metrics, ensure_dirs,
                    Timer, set_seed, cleanup_gpu)


def extract_frozen_embeddings(dataset: RavdessDataset, device: str = "cuda"):
    """Extract last_hidden_state mean-pooled embeddings from frozen HuBERT."""
    model = load_hubert_model(output_hidden_states=False)
    model.eval()
    model.to(device)

    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Extracting embeddings"):
            waveform = dataset.get_waveform(i)
            x = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(device)
            outputs = model(x)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
            embeddings.append(emb)

    del model
    cleanup_gpu()
    return np.array(embeddings)


def run_exp2(dataset: RavdessDataset = None, force: bool = False):
    set_seed(config.DEFAULT_SEED)
    ensure_dirs()

    out_csv = os.path.join(config.SAVE_DIR, "results_exp2_hubert_frozen.csv")
    if os.path.exists(out_csv) and not force:
        print(f"[INFO] Exp2 already done: {out_csv}")
        return pd.read_csv(out_csv)

    if dataset is None:
        dataset = RavdessDataset(sr=config.SR_HUBERT)
    dataset.print_summary()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Extract embeddings once
    emb_path = os.path.join(config.SAVE_DIR, "hubert_frozen_embeddings.npy")
    if os.path.exists(emb_path) and not force:
        print(f"[INFO] Loading cached embeddings from {emb_path}")
        X = np.load(emb_path)
    else:
        with Timer("HuBERT frozen embedding extraction"):
            X = extract_frozen_embeddings(dataset, device)
        np.save(emb_path, X)
        print(f"[INFO] Saved embeddings: {emb_path} ({X.shape})")

    y = dataset.labels
    all_rows = []

    # Protocol 1: 5-fold CV
    splits_5f = get_stratified_cv_splits(y)
    for clf_name, clf_factory in [("SVM", get_svm_pipeline), ("RF", get_rf_pipeline)]:
        for split in splits_5f:
            clf = clf_factory(seed=config.DEFAULT_SEED)
            clf.fit(X[split["train_idx"]], y[split["train_idx"]])
            y_pred = clf.predict(X[split["test_idx"]])
            metrics = compute_metrics(y[split["test_idx"]], y_pred)
            all_rows.append({
                "Experiment": "Exp2", "Model": f"HuBERT-frozen+{clf_name}",
                "Protocol": "5-fold CV", "Fold": split["fold_name"],
                "Seed": config.DEFAULT_SEED, **metrics,
            })
            print(f"  HuBERT-frozen+{clf_name} {split['fold_name']}: "
                  f"Acc={metrics['accuracy']:.1f}%")

    # Protocol 2: LOSGO
    splits_losgo = get_losgo_splits(dataset.actors)
    verify_losgo_splits(splits_losgo, dataset.actors)
    for clf_name, clf_factory in [("SVM", get_svm_pipeline), ("RF", get_rf_pipeline)]:
        for split in splits_losgo:
            train_idx = np.concatenate([split["train_idx"], split["val_idx"]])
            clf = clf_factory(seed=config.DEFAULT_SEED)
            clf.fit(X[train_idx], y[train_idx])
            y_pred = clf.predict(X[split["test_idx"]])
            metrics = compute_metrics(y[split["test_idx"]], y_pred)
            all_rows.append({
                "Experiment": "Exp2", "Model": f"HuBERT-frozen+{clf_name}",
                "Protocol": "LOSGO", "Fold": split["fold_name"],
                "Seed": config.DEFAULT_SEED, **metrics,
            })
            print(f"  HuBERT-frozen+{clf_name} LOSGO {split['fold_name']}: "
                  f"Acc={metrics['accuracy']:.1f}%")

    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)
    print(f"\n[INFO] Saved {out_csv}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_exp2(force=args.force)
