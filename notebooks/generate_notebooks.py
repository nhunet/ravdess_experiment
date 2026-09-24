"""Generate all Colab notebooks for xHuBERT experiments.

Run once:  python notebooks/generate_notebooks.py
Outputs:   notebooks/NB1_*.ipynb ... notebooks/NB7_*.ipynb
"""
import json
import os

REPO_URL = "https://github.com/nhunet/ravdess_experiment.git"
BRANCH = "feature/xhubert-rewrite"

# ─── Helpers ────────────────────────────────────────────────────────────────

def lines(src: str) -> list[str]:
    parts = src.split("\n")
    return [l + "\n" for l in parts[:-1]] + [parts[-1]]


def md_cell(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(src)}


def code_cell(src: str) -> dict:
    return {
        "cell_type": "code", "metadata": {},
        "execution_count": None, "outputs": [],
        "source": lines(src),
    }


def create_notebook(cells: list[dict], filename: str, gpu: bool = True):
    nb = {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4" if gpu else "", "toc_visible": True},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU" if gpu else "",
        },
        "cells": cells,
    }
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"[OK] Created {path} ({len(cells)} cells)")


# ─── Shared boilerplate ────────────────────────────────────────────────────

def boilerplate_cells(title: str, description: str, gpu_required: bool = True):
    """Return the 8 setup cells shared by every notebook."""
    gpu_note = '**T4 GPU**' if gpu_required else 'CPU (GPU optional)'

    cells = [
        # Cell 0: Title
        md_cell(f"""# {title}
**De tai**: He thong goi y san pham dua tren phan tich giong noi va cam xuc
**Hoc vien**: Nguyen Tan Nhu | **GVHD**: TS. Bui Thanh Hung (IUH)

{description}

**Runtime**: {gpu_note}"""),

        # Cell 1: GPU check
        code_cell("""import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
else:
    print("WARNING: GPU not enabled! Runtime -> Change runtime type -> T4 GPU")"""),

        # Cell 2: Mount Drive + set env vars
        code_cell(f"""# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import os

WORK = "/content/drive/MyDrive/xhubert_results"
os.makedirs(WORK, exist_ok=True)

os.environ["XHUBERT_SAVE_DIR"] = WORK
os.environ["RAVDESS_ROOT"] = os.path.join(WORK, "RAVDESS")

print(f"Working directory: {{WORK}}")"""),

        # Cell 3: Install dependencies
        code_cell("""!pip install -q --upgrade transformers==4.51.3 librosa huggingface-hub safetensors tqdm seaborn

import torch, transformers, librosa
print(f"torch:        {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"librosa:      {librosa.__version__}")"""),

        # Cell 4: Clone repo
        code_cell(f"""import os, subprocess

REPO_DIR = "/content/ravdess_experiment"

if os.path.exists(REPO_DIR):
    print("Repo exists, pulling latest ...")
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)
else:
    print("Cloning repo ...")
    subprocess.run([
        "git", "clone", "-b", "{BRANCH}",
        "{REPO_URL}", REPO_DIR
    ], check=True)

os.chdir(REPO_DIR)
print(f"Working in: {{os.getcwd()}}")

# Verify required files
required = [
    "config.py", "data.py", "features.py", "protocols.py",
    "stats.py", "utils.py",
    "models/__init__.py", "models/ml_classifiers.py",
    "models/xhubert.py", "models/hubert_vanilla.py",
    "models/fusion.py", "models/dl_1d.py", "models/dl_2d.py",
    "experiments/__init__.py",
]
missing = [f for f in required if not os.path.exists(f)]
if missing:
    raise FileNotFoundError(f"Missing files: {{missing}}")
print("All required files OK")"""),

        # Cell 5: Download RAVDESS
        code_cell("""import os

DRIVE_RAVDESS = os.environ["RAVDESS_ROOT"]

if not os.path.exists(DRIVE_RAVDESS):
    print("Not in Drive -> Download from Zenodo...")
    !wget -q --show-progress https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip
    !unzip -q Audio_Speech_Actors_01-24.zip -d /content/RAVDESS_tmp
    !mkdir -p "$DRIVE_RAVDESS"
    !cp -r /content/RAVDESS_tmp/* "$DRIVE_RAVDESS/"
    !rm -rf /content/RAVDESS_tmp Audio_Speech_Actors_01-24.zip
    print("Saved to Drive.")
else:
    import glob
    n = len(glob.glob(os.path.join(DRIVE_RAVDESS, "**/*.wav"), recursive=True))
    print(f"Already exist ({n} wav files), don't download.")"""),

        # Cell 6: Import config + verify paths
        code_cell("""import config
from utils import ensure_dirs

ensure_dirs()
print(f"SAVE_DIR:  {config.SAVE_DIR}")
print(f"CSV_DIR:   {config.CSV_DIR}")
print(f"FIG_DIR:   {config.FIG_DIR}")
print(f"CKPT_DIR:  {config.CKPT_DIR}")
print(f"EMB_DIR:   {config.EMB_DIR}")
print(f"LOG_DIR:   {config.LOG_DIR}")
print(f"RAVDESS:   {config.RAVDESS_ROOT}")"""),

        # Cell 7: Keep-alive tip
        md_cell("""### Keep Colab Alive
Paste this into your **browser Console** (F12 -> Console) to prevent idle timeout:
```javascript
function ClickConnect() {
    console.log("Keeping alive...");
    document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(ClickConnect, 60000)
```"""),
    ]
    return cells


# ═══════════════════════════════════════════════════════════════════════════
# NB1: Feature Comparison
# ═══════════════════════════════════════════════════════════════════════════

def make_nb1():
    cells = boilerplate_cells(
        "xHuBERT Experiment 1: Feature Comparison",
        """## Pipeline
```
Step 1: Load RAVDESS (22kHz for handcrafted features)
Step 2: Extract 7 feature types (MFCC, LogMel, LPCC, Chroma, Statistical, Prosody, MFCC+Prosody)
Step 3: Train SVM/RF/KNN on each feature type (5-fold Stratified CV)
Step 4: Visualization & export
```""",
        gpu_required=False,
    )

    cells += [
        md_cell("## Load Dataset"),
        code_cell("""from data import RavdessDataset
import config

dataset = RavdessDataset(sr=config.SR_HANDCRAFTED)
dataset.print_summary()"""),

        md_cell("## Run Experiment 1: Feature Comparison"),
        code_cell("""from experiments.exp1_feature_comparison import run_exp1

df_exp1 = run_exp1(dataset=dataset, force=False)
print(f"\\nTotal rows: {len(df_exp1)}")
df_exp1.head(10)"""),

        md_cell("## Visualization"),
        code_cell("""from visualization.plots import plot_exp1_heatmap, plot_exp1_ranking

plot_exp1_heatmap(df_exp1)
plot_exp1_ranking(df_exp1)
print("Figures saved!")"""),

        md_cell("## Summary"),
        code_cell("""import pandas as pd
pivot = df_exp1.groupby(["Feature", "Model"]).agg(
    Accuracy_mean=("accuracy", "mean"),
    Accuracy_std=("accuracy", "std"),
    F1_mean=("f1_macro", "mean"),
    F1_std=("f1_macro", "std"),
).round(2)
print(pivot.to_string())"""),
    ]
    create_notebook(cells, "NB1_Feature_Comparison.ipynb", gpu=False)


# ═══════════════════════════════════════════════════════════════════════════
# NB2: HuBERT Frozen + Layer Probing (Exp2 + Exp7)
# ═══════════════════════════════════════════════════════════════════════════

def make_nb2():
    cells = boilerplate_cells(
        "xHuBERT Experiments 2+7: HuBERT Frozen Baseline + Layer Probing",
        """## Pipeline
```
Step 1: Exp2 -- HuBERT frozen + SVM/RF (5-fold + LOSGO)
Step 2: Exp7 -- Per-layer probing with logistic regression (LOSGO)
```""",
    )

    cells += [
        md_cell("## Load Dataset (16kHz for HuBERT)"),
        code_cell("""from data import RavdessDataset
import config

dataset = RavdessDataset(sr=config.SR_HUBERT)
dataset.print_summary()"""),

        md_cell("## Exp2: HuBERT Frozen Baseline"),
        code_cell("""from experiments.exp2_hubert_frozen import run_exp2

df_exp2 = run_exp2(dataset=dataset, force=False)
print(df_exp2.groupby(["Model", "Protocol"])["accuracy"].agg(["mean", "std"]).round(2))"""),

        md_cell("## Exp7: Layer Probing"),
        code_cell("""from experiments.exp7_layer_probing import run_exp7

df_exp7 = run_exp7(dataset=dataset, force=False)
print(df_exp7.groupby("Layer")["accuracy"].agg(["mean", "std"]).round(2))"""),

        md_cell("## Visualization"),
        code_cell("""from visualization.plots import plot_exp7_layer_curve
import numpy as np, os, config

alpha_path = os.path.join(config.EMB_DIR, "layer_weights_fold0_seed42.npy")
alpha = np.load(alpha_path) if os.path.exists(alpha_path) else None

plot_exp7_layer_curve(df_exp7, alpha_weights=alpha)
print("Layer probing figure saved!")"""),
    ]
    create_notebook(cells, "NB2_HuBERT_Frozen_LayerProbe.ipynb")


# ═══════════════════════════════════════════════════════════════════════════
# NB3: xHuBERT Fine-tune (Exp3)
# ═══════════════════════════════════════════════════════════════════════════

def make_nb3():
    cells = boilerplate_cells(
        "xHuBERT Experiment 3: xHuBERT Fine-tuning (Main Result)",
        """## Pipeline
```
Step 1: xHuBERT fine-tune: 5-fold CV (seed 42) + LOSGO (seeds 42,43,44)
Step 2: Extract embeddings for Exp5
Step 3: Visualization
```

**Estimated time**: ~7h (seed 42) + ~12h (seeds 43,44)""",
    )

    cells += [
        md_cell("## Load Dataset"),
        code_cell("""from data import RavdessDataset
import config

dataset = RavdessDataset(sr=config.SR_HUBERT)
dataset.print_summary()"""),

        md_cell("## Quick Test (1 fold, 2 epochs)"),
        code_cell("""from experiments.exp3_xhubert_finetune import run_exp3

df_quick = run_exp3(dataset=dataset, seeds=[42], protocols=["LOSGO"],
                    force=True, quick=True)
print("Quick test passed!" if len(df_quick) > 0 else "FAILED!")"""),

        md_cell("## Full Run: 5-fold CV (seed 42)"),
        code_cell("""df_5fold = run_exp3(dataset=dataset, seeds=[42], protocols=["5-fold CV"], force=False)
print(df_5fold.groupby("Protocol")["accuracy"].agg(["mean", "std"]).round(2))"""),

        md_cell("## Full Run: LOSGO (seed 42)"),
        code_cell("""df_losgo_42 = run_exp3(dataset=dataset, seeds=[42], protocols=["LOSGO"], force=False)
print(df_losgo_42.groupby("Protocol")["accuracy"].agg(["mean", "std"]).round(2))"""),

        md_cell("## LOSGO seeds 43, 44 (run in Session 2 if timeout)"),
        code_cell("""df_losgo_extra = run_exp3(dataset=dataset, seeds=[43, 44],
                          protocols=["LOSGO"], force=False)
print(df_losgo_extra.groupby(["Protocol", "Seed"])["accuracy"].agg(["mean", "std"]).round(2))"""),

        md_cell("## Visualization"),
        code_cell("""from visualization.plots import plot_exp3_comparison, plot_confusion_matrix
import pandas as pd, os, config

exp3_csv = os.path.join(config.CSV_DIR, "results_exp3_xhubert.csv")
df_exp3 = pd.read_csv(exp3_csv) if os.path.exists(exp3_csv) else df_5fold

exp2_csv = os.path.join(config.CSV_DIR, "results_exp2_hubert_frozen.csv")
df_exp2 = pd.read_csv(exp2_csv) if os.path.exists(exp2_csv) else None

plot_exp3_comparison(df_exp3, df_exp2)
print("Figures saved!")"""),

        md_cell("## Summary"),
        code_cell("""import pandas as pd, os, config

exp3_csv = os.path.join(config.CSV_DIR, "results_exp3_xhubert.csv")
if os.path.exists(exp3_csv):
    df = pd.read_csv(exp3_csv)
    print("=== xHuBERT Results ===")
    print(df.groupby(["Protocol", "Seed"])[["accuracy", "f1_macro"]].agg(["mean", "std"]).round(2))
else:
    print("Run experiments first!")"""),
    ]
    create_notebook(cells, "NB3_xHuBERT_Finetune.ipynb")


# ═══════════════════════════════════════════════════════════════════════════
# NB4: Ablation (Exp4)
# ═══════════════════════════════════════════════════════════════════════════

def make_nb4():
    cells = boilerplate_cells(
        "xHuBERT Experiment 4: Ablation Study",
        """## Configurations
| Config | SLA | AP | Description |
|--------|-----|----|-------------|
| HuBERT-vanilla-FT | No | No | Fine-tune + last_hidden_state + mean-pool |
| xHuBERT-SLA | Yes | No | +Selective Layer Aggregation |
| xHuBERT-AP | No | Yes | +Attention Pooling |
| xHuBERT-full | Yes | Yes | Proposed method |

**Protocol**: LOSGO x 3 seeds x 6 folds = 72 runs (~72h total)""",
    )

    cells += [
        md_cell("## Load Dataset"),
        code_cell("""from data import RavdessDataset
import config

dataset = RavdessDataset(sr=config.SR_HUBERT)
dataset.print_summary()"""),

        md_cell("## Quick Test"),
        code_cell("""from experiments.exp4_ablation import run_exp4

df_quick = run_exp4(dataset=dataset, seeds=[42], configs=["HuBERT-vanilla-FT"],
                    force=True, quick=True)
print("Quick test passed!" if len(df_quick) > 0 else "FAILED!")"""),

        md_cell("## Session 1: seed 42 (all 4 configs)"),
        code_cell("""df_s1 = run_exp4(dataset=dataset, seeds=[42], force=False)
print(df_s1.groupby("Model")["accuracy"].agg(["mean", "std"]).round(2))"""),

        md_cell("## Session 2: seeds 43, 44"),
        code_cell("""df_s2 = run_exp4(dataset=dataset, seeds=[43, 44], force=False)
print(df_s2.groupby(["Model", "Seed"])["accuracy"].agg(["mean", "std"]).round(2))"""),

        md_cell("## Statistical Tests"),
        code_cell("""import pandas as pd, numpy as np, os, config
from stats import paired_ttest, format_result

csv_path = os.path.join(config.CSV_DIR, "results_exp4_ablation.csv")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    avg = df.groupby(["Model", "Fold"])["accuracy"].mean().reset_index()

    vanilla = avg[avg["Model"] == "HuBERT-vanilla-FT"]["accuracy"].values
    full = avg[avg["Model"] == "xHuBERT-full"]["accuracy"].values

    if len(vanilla) == len(full) and len(vanilla) > 0:
        print(format_result("xHuBERT-full", "HuBERT-vanilla-FT", "Accuracy", full, vanilla))
else:
    print("Run ablation first!")"""),

        md_cell("## Visualization"),
        code_cell("""from visualization.plots import plot_exp4_ablation
import pandas as pd, os, config

csv_path = os.path.join(config.CSV_DIR, "results_exp4_ablation.csv")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    plot_exp4_ablation(df)
    print("Figure saved!")
else:
    print("Run ablation first!")"""),
    ]
    create_notebook(cells, "NB4_Ablation.ipynb")


# ═══════════════════════════════════════════════════════════════════════════
# NB5: Fusion (Exp5)
# ═══════════════════════════════════════════════════════════════════════════

def make_nb5():
    cells = boilerplate_cells(
        "xHuBERT Experiment 5: Fusion Redundancy",
        """**Pre-requisite**: Run Exp3 first (needs embeddings + checkpoints)

## Models
| Model | Description |
|-------|-------------|
| B0-control | FT-emb only (matched architecture) |
| F1-ConcatMLP | Concat(FT-emb, MFCC, Prosody) -> MLP |
| F2-AttGate | 3-branch attention gate |
| F2-AttGate-forced | AttGate + entropy regularization |
| F3-CrossAttn | Cross-attention fusion |
| B1-MFCC-only | MFCC only baseline |
| B2-Prosody-only | Prosody only baseline |""",
    )

    cells += [
        md_cell("## Verify Exp3 Outputs"),
        code_cell("""import os, glob, config
emb_files = glob.glob(os.path.join(config.EMB_DIR, "embeddings_trainval_fold*_seed42.npy"))
print(f"Found {len(emb_files)} embedding files from Exp3")
if len(emb_files) < 6:
    print("WARNING: Need 6 folds of embeddings. Run Exp3 LOSGO seed=42 first!")
else:
    print("All 6 folds available!")"""),

        md_cell("## Run Fusion Experiment"),
        code_cell("""from data import RavdessDataset
from experiments.exp5_fusion import run_exp5
import config

dataset_hc = RavdessDataset(sr=config.SR_HANDCRAFTED)
df_exp5 = run_exp5(dataset_hc=dataset_hc, force=False)
print(df_exp5.groupby("Model")["accuracy"].agg(["mean", "std"]).round(2))"""),

        md_cell("## TOST Equivalence Test"),
        code_cell("""import pandas as pd, numpy as np, os, config
from stats import tost_test

csv_path = os.path.join(config.CSV_DIR, "results_exp5_fusion.csv")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    b0 = df[df["Model"] == "B0-control"].groupby("Fold")["accuracy"].mean().values

    for fusion_name in ["F1-ConcatMLP", "F2-AttGate", "F3-CrossAttn"]:
        fusion = df[df["Model"] == fusion_name].groupby("Fold")["accuracy"].mean().values
        if len(fusion) == len(b0) and len(b0) > 0:
            result = tost_test(fusion, b0, delta=2.0)
            print(f"{fusion_name} vs B0: TOST p={result['p_tost']:.4f}, "
                  f"equivalent={result['equivalent']}")
else:
    print("Run Exp5 first!")"""),

        md_cell("## Gate Weights"),
        code_cell("""from visualization.plots import plot_exp5_gate_weights
import json, os, numpy as np, config

gate_weights = []
for fold in range(6):
    path = os.path.join(config.LOG_DIR, f"exp5_F2-AttGate_fold{fold}_seed42.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        if "gate_weights" in data:
            gate_weights.append(np.array(data["gate_weights"]))

if gate_weights:
    plot_exp5_gate_weights(gate_weights)
    print("Gate weights figure saved!")
else:
    print("No gate weight data found")"""),
    ]
    create_notebook(cells, "NB5_Fusion.ipynb")


# ═══════════════════════════════════════════════════════════════════════════
# NB6: Adversarial (Exp6)
# ═══════════════════════════════════════════════════════════════════════════

def make_nb6():
    cells = boilerplate_cells(
        "xHuBERT Experiment 6: Speaker-Adversarial Training",
        """## 2x2 Ablation
| Config | Augmentation | GRL |
|--------|-------------|-----|
| base | No | No |
| aug | Yes | No |
| grl | No | Yes |
| aug+grl | Yes | Yes |

**Estimated time**: ~10h""",
    )

    cells += [
        md_cell("## Load Dataset"),
        code_cell("""from data import RavdessDataset
import config

dataset = RavdessDataset(sr=config.SR_HUBERT)
dataset.print_summary()"""),

        md_cell("## Quick Test"),
        code_cell("""from experiments.exp6_speaker_adversarial import run_exp6

df_quick = run_exp6(dataset=dataset, force=True, quick=True)
print("Quick test passed!" if len(df_quick) > 0 else "FAILED!")"""),

        md_cell("## Full Run"),
        code_cell("""df_exp6 = run_exp6(dataset=dataset, force=False)
print(df_exp6.groupby("Model")["accuracy"].agg(["mean", "std"]).round(2))"""),

        md_cell("## Visualization"),
        code_cell("""from visualization.plots import plot_exp6_comparison
import pandas as pd, os, config

csv_path = os.path.join(config.CSV_DIR, "results_exp6_adversarial.csv")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    plot_exp6_comparison(df)
    print("Figure saved!")

    probe_cols = [c for c in df.columns if "probe" in c.lower() or "LogReg" in c or "MLP" in c]
    if probe_cols:
        print("\\n=== Speaker Probe Results ===")
        print(df[["Model", "Fold"] + probe_cols].to_string())"""),
    ]
    create_notebook(cells, "NB6_Adversarial.ipynb")


# ═══════════════════════════════════════════════════════════════════════════
# NB7: SOTA Comparison (Exp8)
# ═══════════════════════════════════════════════════════════════════════════

def make_nb7():
    cells = boilerplate_cells(
        "xHuBERT Experiment 8: SOTA Comparison",
        """**Note**: This is Notebook 7 but runs Experiment 8 (NB2 covers both Exp2 and Exp7).

## Methods
- Re-implemented: CNN1D+MFCC, LSTM+MFCC, CNN2D+LogMel
- Literature: Wei et al. 2025, Bhanbhro et al. 2025, Waleed & Shaker 2025, etc.""",
    )

    cells += [
        md_cell("## Load Dataset"),
        code_cell("""from data import RavdessDataset
import config

dataset = RavdessDataset(sr=config.SR_HANDCRAFTED)
dataset.print_summary()"""),

        md_cell("## Quick Test"),
        code_cell("""from experiments.exp8_sota_comparison import run_exp8

df_quick = run_exp8(dataset=dataset, force=True, quick=True)
print("Quick test passed!" if len(df_quick) > 0 else "FAILED!")"""),

        md_cell("## Full Run"),
        code_cell("""df_exp8 = run_exp8(dataset=dataset, force=False)

reimpl = df_exp8[df_exp8["Source"] == "re-implemented"]
print("=== Re-implemented Baselines ===")
print(reimpl.groupby(["Model", "Protocol"])["accuracy"].agg(["mean", "std"]).round(2))

lit = df_exp8[df_exp8["Source"] == "literature"]
print("\\n=== Literature Results ===")
print(lit[["Model", "Protocol", "accuracy"]].to_string())"""),

        md_cell("## Combined Comparison Table"),
        code_cell("""import pandas as pd, os, config

exp3_csv = os.path.join(config.CSV_DIR, "results_exp3_xhubert.csv")
if os.path.exists(exp3_csv):
    df3 = pd.read_csv(exp3_csv)
    print("=== Full Comparison ===")
    for proto in ["5-fold CV", "LOSGO"]:
        sub = df3[df3["Protocol"] == proto]
        if len(sub) > 0:
            print(f"  xHuBERT-full ({proto}): "
                  f"{sub['accuracy'].mean():.2f} +/- {sub['accuracy'].std():.2f}")

    if len(reimpl) > 0:
        for proto in ["5-fold CV", "LOSGO"]:
            sub = reimpl[reimpl["Protocol"] == proto]
            for model in sub["Model"].unique():
                ms = sub[sub["Model"] == model]
                print(f"  {model} ({proto}): "
                      f"{ms['accuracy'].mean():.2f} +/- {ms['accuracy'].std():.2f}")
else:
    print("Run Exp3 first for full comparison!")"""),
    ]
    create_notebook(cells, "NB7_Exp8_SOTA_Comparison.ipynb")


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    make_nb1()
    make_nb2()
    make_nb3()
    make_nb4()
    make_nb5()
    make_nb6()
    make_nb7()
    print("\nAll notebooks generated!")
