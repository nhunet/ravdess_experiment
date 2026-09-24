"""Exp3: xHuBERT fine-tuning (main result). 5-fold CV + LOSGO × multi-seed."""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

import config
from data import RavdessDataset
from protocols import get_stratified_cv_splits, get_losgo_splits, verify_losgo_splits
from models.xhubert import xHuBERT
from utils import (compute_metrics, compute_per_class_f1, ensure_dirs,
                    Timer, set_seed, save_fold_result, load_fold_result,
                    fold_completed, save_checkpoint, cleanup_gpu)


def make_loader(waveforms: list[np.ndarray], labels: np.ndarray,
                indices: np.ndarray, batch_size: int, shuffle: bool = True):
    X = torch.tensor(np.array([waveforms[i] for i in indices]), dtype=torch.float32)
    y = torch.tensor(labels[indices], dtype=torch.long)
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True)


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    assert not model.training, "Model must be in eval() mode"
    all_preds, all_labels, total_loss = [], [], 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        if criterion:
            total_loss += criterion(logits, y).item() * len(y)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    if criterion:
        metrics["loss"] = total_loss / len(all_labels)
    return metrics, np.array(all_preds), np.array(all_labels)


@torch.no_grad()
def extract_embeddings(model, loader, device):
    assert not model.training
    all_emb, all_labels = [], []
    for X, y in loader:
        X = X.to(device)
        _, emb = model(X, return_embedding=True)
        all_emb.append(emb.cpu().numpy())
        all_labels.extend(y.numpy())
    return np.concatenate(all_emb, axis=0), np.array(all_labels)


def train_one_fold(model, train_loader, val_loader, device, seed, fold_name):
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        model.get_param_groups(),
        betas=config.BETAS, weight_decay=config.WEIGHT_DECAY,
    )
    total_steps = len(train_loader) * config.MAX_EPOCHS // config.GRAD_ACCUM_STEPS
    scheduler = OneCycleLR(
        optimizer, max_lr=[g["lr"] for g in optimizer.param_groups],
        total_steps=total_steps, pct_start=config.WARMUP_FRACTION,
        anneal_strategy="linear",
    )

    best_f1, best_state, patience_counter = 0.0, None, 0

    for epoch in range(config.MAX_EPOCHS):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y) / config.GRAD_ACCUM_STEPS
            loss.backward()

            if (step + 1) % config.GRAD_ACCUM_STEPS == 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.item() * config.GRAD_ACCUM_STEPS

        # Validation
        model.eval()
        val_metrics, _, _ = evaluate(model, val_loader, device, criterion)
        train_loss = running_loss / len(train_loader)

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or patience_counter == 0:
            print(f"  [{fold_name}] Epoch {epoch+1}/{config.MAX_EPOCHS} "
                  f"train_loss={train_loss:.4f} val_F1={val_metrics['f1_macro']:.1f}% "
                  f"best_F1={best_f1:.1f}% patience={patience_counter}/{config.PATIENCE}")

        if patience_counter >= config.PATIENCE:
            print(f"  [{fold_name}] Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_state


def run_exp3(dataset: RavdessDataset = None, seeds: list[int] = None,
             protocols: list[str] = None, force: bool = False,
             quick: bool = False):
    ensure_dirs()
    seeds = seeds or config.SEEDS
    protocols = protocols or ["5-fold CV", "LOSGO"]

    if dataset is None:
        dataset = RavdessDataset(sr=config.SR_HUBERT)
    dataset.print_summary()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    all_rows = []

    for protocol in protocols:
        if protocol == "5-fold CV":
            run_seeds = [config.DEFAULT_SEED]
        else:
            run_seeds = seeds

        for seed in run_seeds:
            set_seed(seed)

            if protocol == "5-fold CV":
                splits = get_stratified_cv_splits(dataset.labels)
                # For 5-fold, no separate val — use a portion of train
            else:
                splits = get_losgo_splits(dataset.actors)
                verify_losgo_splits(splits, dataset.actors)

            max_folds = 1 if quick else len(splits)

            for split in splits[:max_folds]:
                fold_idx = split["fold_idx"]
                fold_name = split["fold_name"]
                exp_key = f"exp3_{protocol.replace(' ', '_').replace('-', '')}"

                if fold_completed(exp_key, fold_idx, seed) and not force:
                    print(f"[INFO] {fold_name} seed={seed} already done, skipping")
                    result = load_fold_result(exp_key, fold_idx, seed)
                    all_rows.append(result)
                    continue

                print(f"\n{'='*60}")
                print(f"[INFO] {protocol} {fold_name} seed={seed}")
                print(f"{'='*60}")

                max_epochs_run = 2 if quick else config.MAX_EPOCHS

                # Build data loaders
                if protocol == "LOSGO":
                    train_loader = make_loader(
                        dataset.waveforms, dataset.labels,
                        split["train_idx"], config.BATCH_SIZE,
                    )
                    val_loader = make_loader(
                        dataset.waveforms, dataset.labels,
                        split["val_idx"], config.BATCH_SIZE, shuffle=False,
                    )
                    test_loader = make_loader(
                        dataset.waveforms, dataset.labels,
                        split["test_idx"], config.BATCH_SIZE, shuffle=False,
                    )
                else:
                    # 5-fold: split train into train/val (80/20)
                    train_idx = split["train_idx"].copy()
                    np.random.shuffle(train_idx)
                    val_size = max(1, len(train_idx) // 5)
                    val_idx_cv = train_idx[:val_size]
                    train_idx_cv = train_idx[val_size:]
                    train_loader = make_loader(
                        dataset.waveforms, dataset.labels,
                        train_idx_cv, config.BATCH_SIZE,
                    )
                    val_loader = make_loader(
                        dataset.waveforms, dataset.labels,
                        val_idx_cv, config.BATCH_SIZE, shuffle=False,
                    )
                    test_loader = make_loader(
                        dataset.waveforms, dataset.labels,
                        split["test_idx"], config.BATCH_SIZE, shuffle=False,
                    )

                # Build model
                model = xHuBERT(use_sla=True, use_attention_pool=True)
                model.to(device)

                if quick:
                    orig_max = config.MAX_EPOCHS
                    config.MAX_EPOCHS = max_epochs_run
                    model, best_state = train_one_fold(
                        model, train_loader, val_loader, device, seed, fold_name,
                    )
                    config.MAX_EPOCHS = orig_max
                else:
                    with Timer(f"{fold_name} training"):
                        model, best_state = train_one_fold(
                            model, train_loader, val_loader, device, seed, fold_name,
                        )

                # Evaluate on test
                model.eval()
                test_metrics, y_pred, y_true = evaluate(
                    model, test_loader, device,
                )
                per_class = compute_per_class_f1(y_true, y_pred)

                # Layer weights
                layer_w = model.get_layer_weights()
                if layer_w is not None:
                    lw_path = os.path.join(
                        config.EMB_DIR,
                        f"layer_weights_fold{fold_idx}_seed{seed}.npy",
                    )
                    np.save(lw_path, layer_w.numpy())

                # Extract & save embeddings (for Exp5)
                if protocol == "LOSGO":
                    # Embeddings for train+val (Exp5 trains fusion on these)
                    trainval_idx = np.concatenate([split["train_idx"], split["val_idx"]])
                    trainval_loader = make_loader(
                        dataset.waveforms, dataset.labels,
                        trainval_idx, config.BATCH_SIZE, shuffle=False,
                    )
                    emb_trainval, lab_trainval = extract_embeddings(
                        model, trainval_loader, device,
                    )
                    np.save(os.path.join(
                        config.EMB_DIR,
                        f"embeddings_trainval_fold{fold_idx}_seed{seed}.npy",
                    ), emb_trainval)
                    np.save(os.path.join(
                        config.EMB_DIR,
                        f"labels_trainval_fold{fold_idx}_seed{seed}.npy",
                    ), lab_trainval)

                    # Embeddings for test
                    emb_test, lab_test = extract_embeddings(
                        model, test_loader, device,
                    )
                    np.save(os.path.join(
                        config.EMB_DIR,
                        f"embeddings_test_fold{fold_idx}_seed{seed}.npy",
                    ), emb_test)

                # Save checkpoint
                if best_state is not None:
                    save_checkpoint(best_state, "exp3", fold_idx, seed)

                result = {
                    "Experiment": "Exp3",
                    "Model": "xHuBERT-full",
                    "Protocol": protocol,
                    "Fold": fold_name,
                    "Seed": seed,
                    **test_metrics,
                    "per_class_f1": per_class,
                }
                save_fold_result(result, exp_key, fold_idx, seed)
                all_rows.append(result)

                print(f"  [{fold_name}] Test Acc={test_metrics['accuracy']:.2f}% "
                      f"F1={test_metrics['f1_macro']:.2f}%")

                del model, best_state
                cleanup_gpu()

    # Save combined results
    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(config.CSV_DIR, "results_exp3_xhubert.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[INFO] Saved {out_csv}")

    # Summary
    for protocol in protocols:
        sub = df[df["Protocol"] == protocol]
        if len(sub) > 0:
            print(f"\n{protocol}: Acc={sub['accuracy'].mean():.2f}±{sub['accuracy'].std():.2f}  "
                  f"F1={sub['f1_macro'].mean():.2f}±{sub['f1_macro'].std():.2f}")

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--protocols", nargs="+", default=None)
    args = parser.parse_args()
    run_exp3(seeds=args.seeds, protocols=args.protocols,
             force=args.force, quick=args.quick)
