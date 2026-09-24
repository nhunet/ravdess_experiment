"""All plotting functions for xHuBERT experiments."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import config


def set_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 11, "figure.dpi": config.FIGURE_DPI})


def save_fig(fig, name: str):
    path = os.path.join(config.FIG_DIR, name)
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {path}")


# ─── Exp1: Feature comparison ───────────────────────────────────────────────

def plot_exp1_heatmap(df: pd.DataFrame):
    set_style()
    pivot = df.groupby(["Feature", "Model"])["accuracy"].mean().unstack()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
    ax.set_title("Exp1: Feature × Classifier Accuracy (%)")
    ax.set_ylabel("Feature")
    ax.set_xlabel("Classifier")
    save_fig(fig, "fig_exp1_heatmap.png")


def plot_exp1_ranking(df: pd.DataFrame):
    set_style()
    means = df.groupby("Feature")["accuracy"].mean().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    means.plot(kind="barh", ax=ax, color=sns.color_palette("viridis", len(means)))
    ax.set_xlabel("Mean Accuracy (%)")
    ax.set_title("Exp1: Feature Ranking (averaged over classifiers)")
    save_fig(fig, "fig_exp1_ranking.png")


# ─── Exp3: xHuBERT results ──────────────────────────────────────────────────

def plot_exp3_comparison(df_exp3: pd.DataFrame, df_exp2: pd.DataFrame = None):
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    data = []
    for proto in ["5-fold CV", "LOSGO"]:
        sub = df_exp3[df_exp3["Protocol"] == proto]
        if len(sub) > 0:
            data.append({"Model": "xHuBERT-full", "Protocol": proto,
                         "Accuracy": sub["accuracy"].mean(),
                         "Std": sub["accuracy"].std()})
        if df_exp2 is not None:
            for clf in ["SVM", "RF"]:
                sub2 = df_exp2[(df_exp2["Protocol"] == proto) &
                               (df_exp2["Model"].str.contains(clf))]
                if len(sub2) > 0:
                    data.append({"Model": f"HuBERT-frozen+{clf}", "Protocol": proto,
                                 "Accuracy": sub2["accuracy"].mean(),
                                 "Std": sub2["accuracy"].std()})

    if data:
        plot_df = pd.DataFrame(data)
        x = np.arange(len(plot_df))
        bars = ax.bar(x, plot_df["Accuracy"], yerr=plot_df["Std"],
                      capsize=4, color=sns.color_palette("Set2", len(plot_df)))
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['Model']}\n({r['Protocol']})" for _, r in plot_df.iterrows()],
                           rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Exp3: xHuBERT vs Baselines")
        for bar, val in zip(bars, plot_df["Accuracy"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    save_fig(fig, "fig_exp3_comparison.png")


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix",
                          filename="confusion.png"):
    set_style()
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(cm, display_labels=config.EMOTION_NAMES)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    save_fig(fig, filename)


# ─── Exp4: Ablation ─────────────────────────────────────────────────────────

def plot_exp4_ablation(df: pd.DataFrame):
    set_style()
    order = ["HuBERT-vanilla-FT", "xHuBERT-SLA", "xHuBERT-AP", "xHuBERT-full"]
    means = df.groupby("Model")["accuracy"].agg(["mean", "std"]).reindex(order)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(means))
    bars = ax.bar(x, means["mean"], yerr=means["std"], capsize=5,
                  color=["#e74c3c", "#f39c12", "#3498db", "#2ecc71"])
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=15)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Exp4: Ablation Study (LOSGO)")
    for bar, val in zip(bars, means["mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10)
    save_fig(fig, "fig_exp4_ablation.png")


# ─── Exp5: Fusion gate weights ──────────────────────────────────────────────

def plot_exp5_gate_weights(gate_weights_per_fold: list[np.ndarray]):
    set_style()
    if not gate_weights_per_fold:
        return
    mean_gw = np.mean(gate_weights_per_fold, axis=0)
    branches = ["FT-emb", "MFCC", "Prosody"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(branches, mean_gw, color=["#2ecc71", "#3498db", "#e74c3c"])
    ax.set_ylabel("Gate Weight")
    ax.set_title("Exp5: AttGate Branch Weights (mean over folds)")
    ax.set_ylim(0, 1)
    for i, v in enumerate(mean_gw):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
    save_fig(fig, "fig_exp5_gate_weights.png")


# ─── Exp7: Layer probing ────────────────────────────────────────────────────

def plot_exp7_layer_curve(df: pd.DataFrame, alpha_weights: np.ndarray = None):
    set_style()
    means = df.groupby("Layer")["accuracy"].mean()
    stds = df.groupby("Layer")["accuracy"].std()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(13)
    ax1.errorbar(x, means.values, yerr=stds.values, marker="o", capsize=4,
                 color="#2c3e50", label="Probing accuracy")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Accuracy (%)", color="#2c3e50")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["CNN"] + [str(i) for i in range(1, 13)])

    if alpha_weights is not None:
        ax2 = ax1.twinx()
        ax2.bar(x, alpha_weights, alpha=0.3, color="#e74c3c", label="SLA α weights")
        ax2.set_ylabel("SLA Weight (softmax)", color="#e74c3c")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    ax1.set_title("Exp7: Layer Probing Accuracy + SLA Weights")
    save_fig(fig, "fig_exp7_layer_curve.png")


# ─── Exp6: Speaker adversarial ──────────────────────────────────────────────

def plot_exp6_comparison(df: pd.DataFrame):
    set_style()
    order = ["base", "aug", "grl", "aug+grl"]
    means = df.groupby("Model")["accuracy"].agg(["mean", "std"]).reindex(order)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(means))
    bars = ax.bar(x, means["mean"], yerr=means["std"], capsize=5,
                  color=sns.color_palette("Set2", 4))
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Exp6: Speaker-Adversarial 2×2 Ablation (LOSGO)")
    for bar, val in zip(bars, means["mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom")
    save_fig(fig, "fig_exp6_adversarial.png")
