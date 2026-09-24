"""Evaluation protocols: 5-fold Stratified CV and LOSGO 6-group."""

import numpy as np
from sklearn.model_selection import StratifiedKFold

import config


# ─── LOSGO Groups (deterministic, gender-balanced) ──────────────────────────

LOSGO_GROUPS = {
    1: [1, 2, 3, 4],
    2: [5, 6, 7, 8],
    3: [9, 10, 11, 12],
    4: [13, 14, 15, 16],
    5: [17, 18, 19, 20],
    6: [21, 22, 23, 24],
}


def get_losgo_splits(actors: np.ndarray) -> list[dict]:
    """
    Return 6 LOSGO folds. Val = group immediately after test (wraps around).
    Deterministic — no seed needed.
    """
    groups = list(LOSGO_GROUPS.keys())
    splits = []
    for i, test_g in enumerate(groups):
        val_g = groups[(i + 1) % 6]
        train_gs = [g for g in groups if g != test_g and g != val_g]

        test_actors = LOSGO_GROUPS[test_g]
        val_actors = LOSGO_GROUPS[val_g]
        train_actors = [a for g in train_gs for a in LOSGO_GROUPS[g]]

        test_idx = np.where(np.isin(actors, test_actors))[0]
        val_idx = np.where(np.isin(actors, val_actors))[0]
        train_idx = np.where(np.isin(actors, train_actors))[0]

        splits.append({
            "fold_name": f"Fold {i + 1}",
            "fold_idx": i,
            "test_idx": test_idx,
            "val_idx": val_idx,
            "train_idx": train_idx,
            "test_actors": test_actors,
            "val_actors": val_actors,
            "train_actors": train_actors,
        })
    return splits


def get_stratified_cv_splits(labels: np.ndarray, n_splits: int = 5,
                              seed: int = config.DEFAULT_SEED) -> list[dict]:
    """Return n_splits folds from StratifiedKFold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for i, (train_idx, test_idx) in enumerate(skf.split(labels, labels)):
        splits.append({
            "fold_name": f"Fold {i + 1}",
            "fold_idx": i,
            "train_idx": train_idx,
            "test_idx": test_idx,
        })
    return splits


def verify_losgo_splits(splits: list[dict], actors: np.ndarray):
    """Sanity check LOSGO splits for no leakage and balance."""
    for s in splits:
        test_a = set(actors[s["test_idx"]])
        val_a = set(actors[s["val_idx"]])
        train_a = set(actors[s["train_idx"]])
        assert test_a & val_a == set(), f"{s['fold_name']}: test∩val overlap"
        assert test_a & train_a == set(), f"{s['fold_name']}: test∩train overlap"
        assert val_a & train_a == set(), f"{s['fold_name']}: val∩train overlap"
        assert len(test_a) == 4, f"{s['fold_name']}: test should have 4 actors"
        assert len(val_a) == 4, f"{s['fold_name']}: val should have 4 actors"
        assert len(train_a) == 16, f"{s['fold_name']}: train should have 16 actors"
    print("[INFO] LOSGO splits verified: no overlap, correct sizes")
