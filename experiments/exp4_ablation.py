"""Exp4: Ablation — vanilla → +SLA → +AP → xHuBERT-full under LOSGO."""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR

import config
from data import RavdessDataset
from protocols import get_losgo_splits, verify_losgo_splits
from models.xhubert import xHuBERT
from experiments.exp3_xhubert_finetune import (
    make_loader, evaluate, train_one_fold,
)
from utils import (compute_metrics, compute_per_class_f1, ensure_dirs,
                    Timer, set_seed, save_fold_result, load_fold_result,
                    fold_completed, save_checkpoint, cleanup_gpu)


ABLATION_CONFIGS = {
    "HuBERT-vanilla-FT": {"use_sla": False, "use_attention_pool": False},
    "xHuBERT-SLA":       {"use_sla": True,  "use_attention_pool": False},
    "xHuBERT-AP":        {"use_sla": False, "use_attention_pool": True},
    "xHuBERT-full":      {"use_sla": True,  "use_attention_pool": True},
}


def run_exp4(dataset: RavdessDataset = None, seeds: list[int] = None,
             configs: list[str] = None, force: bool = False,
             quick: bool = False):
    ensure_dirs()
    seeds = seeds or config.SEEDS
    configs = configs or list(ABLATION_CONFIGS.keys())

    if dataset is None:
        dataset = RavdessDataset(sr=config.SR_HUBERT)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = get_losgo_splits(dataset.actors)
    verify_losgo_splits(splits, dataset.actors)

    all_rows = []

    for cfg_name in configs:
        cfg_kwargs = ABLATION_CONFIGS[cfg_name]

        for seed in seeds:
            set_seed(seed)
            max_folds = 1 if quick else len(splits)

            for split in splits[:max_folds]:
                fold_idx = split["fold_idx"]
                fold_name = split["fold_name"]
                exp_key = f"exp4_{cfg_name}"

                if fold_completed(exp_key, fold_idx, seed) and not force:
                    print(f"[INFO] {cfg_name} {fold_name} seed={seed} done, skipping")
                    result = load_fold_result(exp_key, fold_idx, seed)
                    all_rows.append(result)
                    continue

                print(f"\n[INFO] {cfg_name} | {fold_name} | seed={seed}")

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

                model = xHuBERT(n_classes=config.NUM_EMOTIONS, **cfg_kwargs)
                model.to(device)

                if quick:
                    orig = config.MAX_EPOCHS
                    config.MAX_EPOCHS = 2
                    model, best_state = train_one_fold(
                        model, train_loader, val_loader, device, seed, fold_name,
                    )
                    config.MAX_EPOCHS = orig
                else:
                    with Timer(f"{cfg_name} {fold_name}"):
                        model, best_state = train_one_fold(
                            model, train_loader, val_loader, device, seed, fold_name,
                        )

                model.eval()
                test_metrics, y_pred, y_true = evaluate(model, test_loader, device)
                per_class = compute_per_class_f1(y_true, y_pred)
                param_counts = model.count_params()

                if best_state:
                    save_checkpoint(best_state, exp_key, fold_idx, seed)

                result = {
                    "Experiment": "Exp4",
                    "Model": cfg_name,
                    "Protocol": "LOSGO",
                    "Fold": fold_name,
                    "Seed": seed,
                    **test_metrics,
                    "per_class_f1": per_class,
                    "param_counts": param_counts,
                }
                save_fold_result(result, exp_key, fold_idx, seed)
                all_rows.append(result)

                print(f"  Test Acc={test_metrics['accuracy']:.2f}% "
                      f"F1={test_metrics['f1_macro']:.2f}%")

                del model, best_state
                cleanup_gpu()

    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(config.CSV_DIR, "results_exp4_ablation.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[INFO] Saved {out_csv}")

    # Summary per config
    for cfg_name in configs:
        sub = df[df["Model"] == cfg_name]
        if len(sub) > 0:
            print(f"  {cfg_name}: Acc={sub['accuracy'].mean():.2f}±{sub['accuracy'].std():.2f}")

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--configs", nargs="+", default=None)
    args = parser.parse_args()
    run_exp4(seeds=args.seeds, configs=args.configs,
             force=args.force, quick=args.quick)
