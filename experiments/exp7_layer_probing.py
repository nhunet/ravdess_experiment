"""Exp7: Per-layer frozen probing — logistic regression on each HuBERT layer."""

import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import config
from data import RavdessDataset
from protocols import get_losgo_splits, verify_losgo_splits
from utils import (load_hubert_model, compute_metrics, ensure_dirs,
                    Timer, set_seed, cleanup_gpu)


def extract_all_layers(dataset: RavdessDataset, device: str = "cuda"):
    """Extract mean-pooled embeddings from all 13 HuBERT layers."""
    model = load_hubert_model(output_hidden_states=True)
    model.eval()
    model.to(device)

    # layer_embeddings[l] = (N, 768)
    n_layers = 13
    layer_embs = [[] for _ in range(n_layers)]

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Extracting all layers"):
            waveform = dataset.get_waveform(i)
            x = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(device)
            outputs = model(x)
            for l, hs in enumerate(outputs.hidden_states):
                emb = hs.mean(dim=1).squeeze(0).cpu().numpy()
                layer_embs[l].append(emb)

    del model
    cleanup_gpu()

    return [np.array(le) for le in layer_embs]


def run_exp7(dataset: RavdessDataset = None, force: bool = False):
    set_seed(config.DEFAULT_SEED)
    ensure_dirs()

    out_csv = os.path.join(config.CSV_DIR, "results_exp7_layer_probing.csv")
    if os.path.exists(out_csv) and not force:
        print(f"[INFO] Exp7 already done: {out_csv}")
        return pd.read_csv(out_csv)

    if dataset is None:
        dataset = RavdessDataset(sr=config.SR_HUBERT)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Extract all layer embeddings
    cache_path = os.path.join(config.EMB_DIR, "hubert_all_layers.npz")
    if os.path.exists(cache_path) and not force:
        print(f"[INFO] Loading cached layer embeddings from {cache_path}")
        data = np.load(cache_path)
        layer_embs = [data[f"layer_{l}"] for l in range(13)]
    else:
        with Timer("All-layer extraction"):
            layer_embs = extract_all_layers(dataset, device)
        np.savez(cache_path, **{f"layer_{l}": emb for l, emb in enumerate(layer_embs)})
        print(f"[INFO] Saved {cache_path}")

    y = dataset.labels
    splits = get_losgo_splits(dataset.actors)
    verify_losgo_splits(splits, dataset.actors)

    all_rows = []

    for layer_idx in range(13):
        X = layer_embs[layer_idx]
        layer_name = f"Layer {layer_idx}" if layer_idx > 0 else "CNN output"

        for split in splits:
            train_idx = np.concatenate([split["train_idx"], split["val_idx"]])
            test_idx = split["test_idx"]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])

            clf = LogisticRegression(
                max_iter=1000, multi_class="multinomial",
                solver="lbfgs", random_state=config.DEFAULT_SEED,
            )
            clf.fit(X_train, y[train_idx])
            y_pred = clf.predict(X_test)
            metrics = compute_metrics(y[test_idx], y_pred)

            row = {
                "Experiment": "Exp7",
                "Layer": layer_idx,
                "Layer_name": layer_name,
                "Protocol": "LOSGO",
                "Fold": split["fold_name"],
                "Seed": config.DEFAULT_SEED,
                **metrics,
            }
            all_rows.append(row)

        layer_accs = [r["accuracy"] for r in all_rows if r["Layer"] == layer_idx]
        print(f"  {layer_name}: Acc={np.mean(layer_accs):.1f}±{np.std(layer_accs):.1f}%")

    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)
    print(f"\n[INFO] Saved {out_csv}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_exp7(force=args.force)
