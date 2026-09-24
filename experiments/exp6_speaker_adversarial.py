"""Exp6: Speaker-adversarial training — GRL + augmentation 2×2 ablation."""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

import config
from data import RavdessDataset
from protocols import get_losgo_splits, verify_losgo_splits
from models.xhubert import xHuBERT
from experiments.exp3_xhubert_finetune import make_loader, train_one_fold, evaluate
from utils import (compute_metrics, ensure_dirs, Timer, set_seed,
                    save_fold_result, fold_completed, load_fold_result,
                    cleanup_gpu)


# ─── Gradient Reversal Layer ────────────────────────────────────────────────

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GRLSpeakerClassifier(nn.Module):
    def __init__(self, embed_dim: int = 256, n_speakers: int = 16):
        super().__init__()
        self.clf = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_speakers),
        )

    def forward(self, x, alpha=1.0):
        x = GradReverse.apply(x, alpha)
        return self.clf(x)


# ─── Augmentation ───────────────────────────────────────────────────────────

def augment_waveform(y: np.ndarray, sr: int, seed: int = None) -> np.ndarray:
    """Apply random augmentation chain."""
    rng = np.random.default_rng(seed)

    # Speed perturbation
    if rng.random() < config.AUG_PROBS["speed"]:
        factor = rng.choice(config.AUG_SPEED_FACTORS)
        import librosa
        y = librosa.effects.time_stretch(y, rate=factor)

    target_len = int(sr * config.DURATION_SEC)

    # Gain
    if rng.random() < config.AUG_PROBS["gain"]:
        gain_db = rng.uniform(-config.AUG_GAIN_DB, config.AUG_GAIN_DB)
        y = y * (10 ** (gain_db / 20.0))

    # Time shift
    if rng.random() < config.AUG_PROBS["shift"]:
        shift = int(rng.uniform(-config.AUG_SHIFT_FRACTION, config.AUG_SHIFT_FRACTION) * len(y))
        y = np.roll(y, shift)

    # Additive noise
    if rng.random() < config.AUG_PROBS["noise"]:
        snr = rng.uniform(*config.AUG_NOISE_SNR_RANGE)
        signal_power = np.mean(y ** 2)
        noise_power = signal_power / (10 ** (snr / 10))
        noise = rng.normal(0, np.sqrt(noise_power), len(y))
        y = y + noise

    # Pad/trim to target length
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    return y.astype(np.float32)


# ─── xHuBERT with GRL ───────────────────────────────────────────────────────

class xHuBERTWithGRL(nn.Module):
    def __init__(self, base_model: xHuBERT, n_train_speakers: int = 16):
        super().__init__()
        self.base = base_model
        self.speaker_clf = GRLSpeakerClassifier(
            config.EMBEDDING_DIM, n_train_speakers,
        )

    def forward(self, input_values, alpha=1.0, return_embedding=False):
        if return_embedding:
            logits, emb = self.base(input_values, return_embedding=True)
            speaker_logits = self.speaker_clf(emb, alpha)
            return logits, speaker_logits, emb
        else:
            logits, emb = self.base(input_values, return_embedding=True)
            speaker_logits = self.speaker_clf(emb, alpha)
            return logits, speaker_logits

    def get_param_groups(self):
        groups = self.base.get_param_groups()
        groups[1]["params"].extend(list(self.speaker_clf.parameters()))
        return groups


def grl_lambda_schedule(p: float) -> float:
    """GRL lambda schedule: λ(p) = 2/(1+e^{-10p}) − 1."""
    return 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0


# ─── Speaker probe ──────────────────────────────────────────────────────────

def speaker_probe(embeddings: np.ndarray, speaker_ids: np.ndarray,
                   n_permutations: int = 50):
    """Train logistic regression + MLP to predict speaker from embedding."""
    le = LabelEncoder()
    y = le.fit_transform(speaker_ids)

    results = {}
    for name, clf in [
        ("LogReg", LogisticRegression(max_iter=1000, multi_class="multinomial")),
        ("MLP", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)),
    ]:
        clf.fit(embeddings, y)
        acc = clf.score(embeddings, y) * 100
        results[f"{name}_train_acc"] = acc

        # Permutation test
        perm_accs = []
        for _ in range(n_permutations):
            y_perm = np.random.permutation(y)
            clf_perm = LogisticRegression(max_iter=1000) if name == "LogReg" else \
                       MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)
            clf_perm.fit(embeddings, y_perm)
            perm_accs.append(clf_perm.score(embeddings, y_perm) * 100)
        results[f"{name}_perm_mean"] = np.mean(perm_accs)
        results[f"{name}_perm_std"] = np.std(perm_accs)
        results[f"{name}_p_value"] = np.mean(np.array(perm_accs) >= acc)

    return results


# ─── Main ────────────────────────────────────────────────────────────────────

ADVERSARIAL_CONFIGS = {
    "base":    {"augment": False, "grl": False},
    "aug":     {"augment": True,  "grl": False},
    "grl":     {"augment": False, "grl": True},
    "aug+grl": {"augment": True,  "grl": True},
}


def run_exp6(dataset: RavdessDataset = None, force: bool = False,
             quick: bool = False):
    set_seed(config.DEFAULT_SEED)
    ensure_dirs()

    out_csv = os.path.join(config.CSV_DIR, "results_exp6_adversarial.csv")
    if os.path.exists(out_csv) and not force:
        print(f"[INFO] Exp6 already done: {out_csv}")
        return pd.read_csv(out_csv)

    if dataset is None:
        dataset = RavdessDataset(sr=config.SR_HUBERT)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = get_losgo_splits(dataset.actors)
    verify_losgo_splits(splits, dataset.actors)

    all_rows = []
    seed = config.DEFAULT_SEED

    for cfg_name, cfg in ADVERSARIAL_CONFIGS.items():
        max_folds = 1 if quick else len(splits)

        for split in splits[:max_folds]:
            fold_idx = split["fold_idx"]
            fold_name = split["fold_name"]
            exp_key = f"exp6_{cfg_name}"

            if fold_completed(exp_key, fold_idx, seed) and not force:
                print(f"[INFO] {cfg_name} {fold_name} done, skipping")
                result = load_fold_result(exp_key, fold_idx, seed)
                all_rows.append(result)
                continue

            print(f"\n[INFO] {cfg_name} | {fold_name}")

            # Prepare waveforms (with optional augmentation)
            waveforms = dataset.waveforms
            labels = dataset.labels

            if cfg["augment"]:
                # Create augmented copies of training data
                aug_waveforms = list(waveforms)
                aug_labels = list(labels)
                aug_actors = list(dataset.actors)
                for idx in split["train_idx"]:
                    aug_y = augment_waveform(waveforms[idx], config.SR_HUBERT,
                                             seed=seed + idx)
                    aug_waveforms.append(aug_y)
                    aug_labels.append(labels[idx])
                    aug_actors.append(dataset.actors[idx])
                aug_labels = np.array(aug_labels)
                aug_actors = np.array(aug_actors)
                n_orig = len(waveforms)
                aug_train_idx = np.concatenate([
                    split["train_idx"],
                    np.arange(n_orig, len(aug_waveforms)),
                ])
                train_loader = make_loader(
                    aug_waveforms, aug_labels, aug_train_idx, config.BATCH_SIZE,
                )
            else:
                train_loader = make_loader(
                    waveforms, labels, split["train_idx"], config.BATCH_SIZE,
                )

            val_loader = make_loader(
                waveforms, labels, split["val_idx"], config.BATCH_SIZE, shuffle=False,
            )
            test_loader = make_loader(
                waveforms, labels, split["test_idx"], config.BATCH_SIZE, shuffle=False,
            )

            # Build model
            base_model = xHuBERT(use_sla=True, use_attention_pool=True)

            if cfg["grl"]:
                n_train_speakers = len(set(dataset.actors[split["train_idx"]]))
                model = xHuBERTWithGRL(base_model, n_train_speakers)
            else:
                model = base_model

            model.to(device)

            if not cfg["grl"]:
                # Standard training
                if quick:
                    orig = config.MAX_EPOCHS
                    config.MAX_EPOCHS = 2
                model, best_state = train_one_fold(
                    model, train_loader, val_loader, device, seed, fold_name,
                )
                if quick:
                    config.MAX_EPOCHS = orig
            else:
                # GRL training (custom loop)
                emotion_criterion = nn.CrossEntropyLoss(
                    label_smoothing=config.LABEL_SMOOTHING,
                )
                speaker_criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.AdamW(
                    model.get_param_groups(),
                    betas=config.BETAS, weight_decay=config.WEIGHT_DECAY,
                )

                # Speaker label encoder for train actors
                train_actor_list = sorted(set(dataset.actors[split["train_idx"]]))
                actor_to_idx = {a: i for i, a in enumerate(train_actor_list)}

                # Rebuild train_loader with speaker labels baked into the
                # TensorDataset so labels stay aligned after shuffle.
                if cfg["augment"]:
                    grl_wf, grl_act = aug_waveforms, aug_actors
                    grl_lab, grl_idx = aug_labels, aug_train_idx
                else:
                    grl_wf, grl_act = waveforms, dataset.actors
                    grl_lab, grl_idx = labels, split["train_idx"]

                X_grl = torch.tensor(
                    np.array([grl_wf[i] for i in grl_idx]),
                    dtype=torch.float32,
                )
                y_emo_grl = torch.tensor(grl_lab[grl_idx], dtype=torch.long)
                y_spk_grl = torch.tensor(
                    [actor_to_idx.get(grl_act[i], 0) for i in grl_idx],
                    dtype=torch.long,
                )
                grl_ds = TensorDataset(X_grl, y_emo_grl, y_spk_grl)
                train_loader = DataLoader(
                    grl_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                    num_workers=0, pin_memory=True,
                )

                max_epochs = 2 if quick else config.MAX_EPOCHS
                best_f1, best_state, patience = 0.0, None, 0
                total_steps = max_epochs * len(train_loader)

                for epoch in range(max_epochs):
                    model.train()
                    for step, (X, y_emo, y_spk) in enumerate(train_loader):
                        p = (epoch * len(train_loader) + step) / total_steps
                        alpha = grl_lambda_schedule(p)

                        X = X.to(device)
                        y_emo = y_emo.to(device)
                        y_spk = y_spk.to(device)
                        emo_logits, spk_logits = model(X, alpha=alpha)
                        loss_emo = emotion_criterion(emo_logits, y_emo)
                        loss_spk = speaker_criterion(spk_logits, y_spk)
                        loss = loss_emo + loss_spk

                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
                        optimizer.step()

                    # Validation
                    model.eval()
                    val_preds, val_labels_list = [], []
                    with torch.no_grad():
                        for X, y in val_loader:
                            X = X.to(device)
                            if isinstance(model, xHuBERTWithGRL):
                                logits, _ = model(X, alpha=0.0)
                            else:
                                logits = model(X)
                            val_preds.extend(logits.argmax(-1).cpu().numpy())
                            val_labels_list.extend(y.numpy())
                    m = compute_metrics(np.array(val_labels_list), np.array(val_preds))
                    if m["f1_macro"] > best_f1:
                        best_f1 = m["f1_macro"]
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        patience = 0
                    else:
                        patience += 1
                    if patience >= config.PATIENCE:
                        break

                if best_state:
                    model.load_state_dict(best_state)

            # Evaluate
            model.eval()
            test_preds, test_labels_list = [], []
            all_emb = []
            with torch.no_grad():
                for X, y in test_loader:
                    X = X.to(device)
                    if isinstance(model, xHuBERTWithGRL):
                        logits, _, emb = model(X, alpha=0.0, return_embedding=True)
                    else:
                        logits, emb = model(X, return_embedding=True)
                    test_preds.extend(logits.argmax(-1).cpu().numpy())
                    test_labels_list.extend(y.numpy())
                    if emb is not None:
                        all_emb.append(emb.cpu().numpy())

            metrics = compute_metrics(np.array(test_labels_list), np.array(test_preds))

            # Speaker probe on test embeddings
            probe_results = {}
            if all_emb:
                emb_concat = np.concatenate(all_emb, axis=0)
                test_actors = dataset.actors[split["test_idx"]]
                probe_results = speaker_probe(emb_concat, test_actors)

            result = {
                "Experiment": "Exp6", "Model": cfg_name,
                "Protocol": "LOSGO", "Fold": fold_name,
                "Seed": seed, **metrics, **probe_results,
            }
            save_fold_result(result, exp_key, fold_idx, seed)
            all_rows.append(result)

            print(f"  {cfg_name} {fold_name}: Acc={metrics['accuracy']:.1f}%")

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
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run_exp6(force=args.force, quick=args.quick)
