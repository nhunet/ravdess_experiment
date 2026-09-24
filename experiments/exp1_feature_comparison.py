"""Exp1: Handcrafted features × ML classifiers (5-fold Stratified CV)."""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from data import RavdessDataset
from features import FEATURE_EXTRACTORS, build_feature_matrix
from protocols import get_stratified_cv_splits
from models.ml_classifiers import ML_CLASSIFIERS
from utils import compute_metrics, ensure_dirs, Timer, set_seed


def run_exp1(dataset: RavdessDataset = None, force: bool = False):
    set_seed(config.DEFAULT_SEED)
    ensure_dirs()

    out_csv = os.path.join(config.SAVE_DIR, "results_exp1_feature_comparison.csv")
    if os.path.exists(out_csv) and not force:
        print(f"[INFO] Exp1 already done: {out_csv}")
        return pd.read_csv(out_csv)

    if dataset is None:
        dataset = RavdessDataset(sr=config.SR_HANDCRAFTED)
    dataset.print_summary()

    splits = get_stratified_cv_splits(dataset.labels)
    all_rows = []

    for feat_name, (extractor_fn, expected_dim) in FEATURE_EXTRACTORS.items():
        print(f"\n[INFO] Extracting {feat_name} ({expected_dim}d) ...")
        with Timer(f"Feature extraction: {feat_name}"):
            X, y = build_feature_matrix(
                dataset.waveforms, dataset.labels,
                extractor_fn, expected_dim, sr=config.SR_HANDCRAFTED,
            )
        print(f"[INFO] {feat_name}: {X.shape[0]} samples × {X.shape[1]} dims")

        for clf_name, clf_factory in ML_CLASSIFIERS.items():
            for split in splits:
                fold_name = split["fold_name"]
                train_idx = split["train_idx"]
                test_idx = split["test_idx"]

                # Filter indices that survived feature extraction
                max_idx = len(y) - 1
                train_idx = train_idx[train_idx <= max_idx]
                test_idx = test_idx[test_idx <= max_idx]

                X_train, y_train = X[train_idx], y[train_idx]
                X_test, y_test = X[test_idx], y[test_idx]

                clf = clf_factory() if clf_name == "KNN" else clf_factory(seed=config.DEFAULT_SEED)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                metrics = compute_metrics(y_test, y_pred)

                row = {
                    "Experiment": "Exp1",
                    "Feature": feat_name,
                    "Model": clf_name,
                    "Protocol": "5-fold CV",
                    "Fold": fold_name,
                    "Seed": config.DEFAULT_SEED,
                    **metrics,
                }
                all_rows.append(row)
                print(f"  {feat_name}+{clf_name} {fold_name}: "
                      f"Acc={metrics['accuracy']:.1f}%, F1={metrics['f1_macro']:.1f}%")

    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)
    print(f"\n[INFO] Saved {out_csv}")

    # Pivot summary
    pivot = df.groupby(["Feature", "Model"]).agg(
        Accuracy_mean=("accuracy", "mean"),
        Accuracy_std=("accuracy", "std"),
        F1_mean=("f1_macro", "mean"),
        F1_std=("f1_macro", "std"),
    ).round(2).reset_index()
    pivot_csv = os.path.join(config.SAVE_DIR, "results_exp1_pivot.csv")
    pivot.to_csv(pivot_csv, index=False)
    print(f"[INFO] Saved {pivot_csv}")

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_exp1(force=args.force)
