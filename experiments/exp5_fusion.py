"""Exp5: Fusion redundancy — reuse Exp3 embeddings + handcrafted features."""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import config
from data import RavdessDataset
from features import extract_mfcc, extract_prosody
from protocols import get_losgo_splits, verify_losgo_splits
from models.fusion import (ConcatMLP, AttGateFusion, CrossAttentionFusion,
                            EmbeddingOnlyBaseline, SingleFeatureBaseline)
from utils import (compute_metrics, ensure_dirs, Timer, set_seed,
                    save_fold_result, fold_completed, load_fold_result,
                    cleanup_gpu)


def extract_handcrafted_for_indices(dataset_hc: RavdessDataset,
                                     indices: np.ndarray):
    """Extract MFCC (240d) and Prosody (34d) for given indices."""
    mfcc_list, prosody_list = [], []
    for idx in indices:
        y = dataset_hc.get_waveform(idx)
        mfcc_list.append(extract_mfcc(y, config.SR_HANDCRAFTED))
        prosody_list.append(extract_prosody(y, config.SR_HANDCRAFTED))
    return np.array(mfcc_list), np.array(prosody_list)


def train_fusion_model(model, train_data, val_data, device,
                       model_type, gate_forced=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.FUSION_LR)
    criterion = nn.CrossEntropyLoss()
    best_f1, best_state, patience = 0.0, None, 0

    for epoch in range(config.FUSION_EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_data:
            optimizer.zero_grad()

            if model_type in ("concat", "attgate", "attgate_forced"):
                ft, mfcc, pros, y = [b.to(device) for b in batch]
                if model_type == "concat":
                    logits = model(ft, mfcc, pros)
                    loss = criterion(logits, y)
                else:
                    logits, gate_w = model(ft, mfcc, pros)
                    loss = criterion(logits, y)
                    if gate_forced:
                        entropy = model.entropy_loss(gate_w)
                        loss = loss - config.GATE_ENTROPY_LAMBDA * entropy
            elif model_type == "crossattn":
                ft, hc, y = [b.to(device) for b in batch]
                logits = model(ft, hc)
                loss = criterion(logits, y)
            else:  # baselines
                x, y = batch[0].to(device), batch[1].to(device)
                logits = model(x)
                loss = criterion(logits, y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        gate_weights_all = []
        with torch.no_grad():
            for batch in val_data:
                if model_type in ("concat", "attgate", "attgate_forced"):
                    ft, mfcc, pros, y = [b.to(device) for b in batch]
                    if model_type == "concat":
                        logits = model(ft, mfcc, pros)
                    else:
                        logits, gw = model(ft, mfcc, pros)
                        gate_weights_all.append(gw.cpu().numpy())
                elif model_type == "crossattn":
                    ft, hc, y = [b.to(device) for b in batch]
                    logits = model(ft, hc)
                else:
                    x, y = batch[0].to(device), batch[1].to(device)
                    logits = model(x)
                val_preds.extend(logits.argmax(-1).cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        m = compute_metrics(np.array(val_labels), np.array(val_preds))
        if m["f1_macro"] > best_f1:
            best_f1 = m["f1_macro"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if patience >= config.FUSION_PATIENCE:
            break

    if best_state:
        model.load_state_dict(best_state)

    avg_gate = None
    if gate_weights_all:
        avg_gate = np.concatenate(gate_weights_all, axis=0).mean(axis=0)

    return model, avg_gate


def run_exp5(dataset_hc: RavdessDataset = None, force: bool = False):
    set_seed(config.DEFAULT_SEED)
    ensure_dirs()

    out_csv = os.path.join(config.SAVE_DIR, "results_exp5_fusion.csv")
    if os.path.exists(out_csv) and not force:
        print(f"[INFO] Exp5 already done: {out_csv}")
        return pd.read_csv(out_csv)

    if dataset_hc is None:
        dataset_hc = RavdessDataset(sr=config.SR_HANDCRAFTED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = get_losgo_splits(dataset_hc.actors)
    verify_losgo_splits(splits, dataset_hc.actors)

    seed = config.DEFAULT_SEED
    all_rows = []

    FUSION_MODELS = {
        "B0-control": ("baseline_emb", EmbeddingOnlyBaseline, {}),
        "F1-ConcatMLP": ("concat", ConcatMLP, {}),
        "F2-AttGate": ("attgate", AttGateFusion, {"gate_forced": False}),
        "F2-AttGate-forced": ("attgate_forced", AttGateFusion, {"gate_forced": True}),
        "F3-CrossAttn": ("crossattn", CrossAttentionFusion, {}),
        "B1-MFCC-only": ("baseline_feat", SingleFeatureBaseline, {"input_dim": 240}),
        "B2-Prosody-only": ("baseline_feat", SingleFeatureBaseline, {"input_dim": 34}),
    }

    for split in splits:
        fold_idx = split["fold_idx"]
        fold_name = split["fold_name"]

        # Load precomputed embeddings from Exp3
        emb_trainval_path = os.path.join(
            config.SAVE_DIR, f"embeddings_trainval_fold{fold_idx}_seed{seed}.npy"
        )
        emb_test_path = os.path.join(
            config.SAVE_DIR, f"embeddings_test_fold{fold_idx}_seed{seed}.npy"
        )
        lab_trainval_path = os.path.join(
            config.SAVE_DIR, f"labels_trainval_fold{fold_idx}_seed{seed}.npy"
        )

        if not os.path.exists(emb_trainval_path):
            print(f"[WARN] Missing embeddings for {fold_name} — run Exp3 first")
            continue

        emb_trainval = np.load(emb_trainval_path)
        emb_test = np.load(emb_test_path)
        lab_trainval = np.load(lab_trainval_path)
        lab_test = dataset_hc.labels[split["test_idx"]]

        # Extract handcrafted for train+val and test
        trainval_idx = np.concatenate([split["train_idx"], split["val_idx"]])
        print(f"\n[INFO] {fold_name}: extracting handcrafted features ...")
        mfcc_trainval, pros_trainval = extract_handcrafted_for_indices(
            dataset_hc, trainval_idx,
        )
        mfcc_test, pros_test = extract_handcrafted_for_indices(
            dataset_hc, split["test_idx"],
        )

        # Split trainval into train/val for fusion training
        n_val = len(split["val_idx"])
        n_train = len(trainval_idx) - n_val

        for model_name, (model_type, model_cls, extra) in FUSION_MODELS.items():
            exp_key = f"exp5_{model_name}"
            if fold_completed(exp_key, fold_idx, seed) and not force:
                print(f"[INFO] {model_name} {fold_name} done, skipping")
                result = load_fold_result(exp_key, fold_idx, seed)
                all_rows.append(result)
                continue

            print(f"  Training {model_name} on {fold_name} ...")

            # Build tensors
            emb_t = torch.tensor(emb_trainval, dtype=torch.float32)
            mfcc_t = torch.tensor(mfcc_trainval, dtype=torch.float32)
            pros_t = torch.tensor(pros_trainval, dtype=torch.float32)
            lab_t = torch.tensor(lab_trainval, dtype=torch.long)

            emb_te = torch.tensor(emb_test, dtype=torch.float32)
            mfcc_te = torch.tensor(mfcc_test, dtype=torch.float32)
            pros_te = torch.tensor(pros_test, dtype=torch.float32)
            lab_te = torch.tensor(lab_test, dtype=torch.long)

            # Train/val split within trainval
            train_sl = slice(0, n_train)
            val_sl = slice(n_train, None)

            bs = 32

            if model_type in ("concat", "attgate", "attgate_forced"):
                if "gate_forced" in extra:
                    model = model_cls(gate_forced=extra["gate_forced"])
                else:
                    model = model_cls()

                train_ds = TensorDataset(emb_t[train_sl], mfcc_t[train_sl],
                                          pros_t[train_sl], lab_t[train_sl])
                val_ds = TensorDataset(emb_t[val_sl], mfcc_t[val_sl],
                                        pros_t[val_sl], lab_t[val_sl])
                test_ds = TensorDataset(emb_te, mfcc_te, pros_te, lab_te)
            elif model_type == "crossattn":
                model = model_cls()
                hc_train = torch.cat([mfcc_t[train_sl], pros_t[train_sl]], dim=-1)
                hc_val = torch.cat([mfcc_t[val_sl], pros_t[val_sl]], dim=-1)
                hc_test = torch.cat([mfcc_te, pros_te], dim=-1)

                train_ds = TensorDataset(emb_t[train_sl], hc_train, lab_t[train_sl])
                val_ds = TensorDataset(emb_t[val_sl], hc_val, lab_t[val_sl])
                test_ds = TensorDataset(emb_te, hc_test, lab_te)
            elif model_type == "baseline_emb":
                model = model_cls()
                train_ds = TensorDataset(emb_t[train_sl], lab_t[train_sl])
                val_ds = TensorDataset(emb_t[val_sl], lab_t[val_sl])
                test_ds = TensorDataset(emb_te, lab_te)
            else:  # baseline_feat
                inp_dim = extra["input_dim"]
                model = model_cls(input_dim=inp_dim)
                if inp_dim == 240:
                    train_ds = TensorDataset(mfcc_t[train_sl], lab_t[train_sl])
                    val_ds = TensorDataset(mfcc_t[val_sl], lab_t[val_sl])
                    test_ds = TensorDataset(mfcc_te, lab_te)
                else:
                    train_ds = TensorDataset(pros_t[train_sl], lab_t[train_sl])
                    val_ds = TensorDataset(pros_t[val_sl], lab_t[val_sl])
                    test_ds = TensorDataset(pros_te, lab_te)

            model.to(device)
            train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
            val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False)
            test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False)

            model, avg_gate = train_fusion_model(
                model, train_dl, val_dl, device, model_type,
                gate_forced=extra.get("gate_forced", False),
            )

            # Test evaluation
            model.eval()
            test_preds, test_labels = [], []
            with torch.no_grad():
                for batch in test_dl:
                    if model_type in ("concat", "attgate", "attgate_forced"):
                        ft, mfcc, pros, y = [b.to(device) for b in batch]
                        if model_type == "concat":
                            logits = model(ft, mfcc, pros)
                        else:
                            logits, _ = model(ft, mfcc, pros)
                    elif model_type == "crossattn":
                        ft, hc, y = [b.to(device) for b in batch]
                        logits = model(ft, hc)
                    else:
                        x, y = batch[0].to(device), batch[1].to(device)
                        logits = model(x)
                    test_preds.extend(logits.argmax(-1).cpu().numpy())
                    test_labels.extend(y.cpu().numpy())

            metrics = compute_metrics(np.array(test_labels), np.array(test_preds))

            result = {
                "Experiment": "Exp5", "Model": model_name,
                "Protocol": "LOSGO", "Fold": fold_name,
                "Seed": seed, **metrics,
            }
            if avg_gate is not None:
                result["gate_weights"] = avg_gate.tolist()
            save_fold_result(result, exp_key, fold_idx, seed)
            all_rows.append(result)

            print(f"    {model_name}: Acc={metrics['accuracy']:.1f}%  "
                  f"F1={metrics['f1_macro']:.1f}%"
                  + (f"  gates={avg_gate}" if avg_gate is not None else ""))

            del model
            cleanup_gpu()

    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)
    print(f"\n[INFO] Saved {out_csv}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_exp5(force=args.force)
