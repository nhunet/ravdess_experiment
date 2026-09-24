"""Orchestrator: run all experiments in sequence."""

import argparse
import config
from utils import ensure_dirs, Timer


def main():
    parser = argparse.ArgumentParser(description="xHuBERT Experiments Orchestrator")
    parser.add_argument("--exp", type=int, nargs="+", default=None,
                        help="Run specific experiments (e.g., --exp 1 3)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    exps = args.exp or [1, 2, 3, 4, 5, 6, 7, 8]

    with Timer("Total pipeline"):
        if 1 in exps:
            print("\n" + "=" * 70)
            print("EXP1: Feature Comparison")
            print("=" * 70)
            from experiments.exp1_feature_comparison import run_exp1
            run_exp1(force=args.force)

        if 2 in exps:
            print("\n" + "=" * 70)
            print("EXP2: HuBERT Frozen Baseline")
            print("=" * 70)
            from experiments.exp2_hubert_frozen import run_exp2
            run_exp2(force=args.force)

        if 7 in exps:
            print("\n" + "=" * 70)
            print("EXP7: Layer Probing")
            print("=" * 70)
            from experiments.exp7_layer_probing import run_exp7
            run_exp7(force=args.force)

        if 3 in exps:
            print("\n" + "=" * 70)
            print("EXP3: xHuBERT Fine-tune")
            print("=" * 70)
            from experiments.exp3_xhubert_finetune import run_exp3
            run_exp3(force=args.force, quick=args.quick)

        if 4 in exps:
            print("\n" + "=" * 70)
            print("EXP4: Ablation Study")
            print("=" * 70)
            from experiments.exp4_ablation import run_exp4
            run_exp4(force=args.force, quick=args.quick)

        if 5 in exps:
            print("\n" + "=" * 70)
            print("EXP5: Fusion Redundancy")
            print("=" * 70)
            from experiments.exp5_fusion import run_exp5
            run_exp5(force=args.force)

        if 6 in exps:
            print("\n" + "=" * 70)
            print("EXP6: Speaker-Adversarial Training")
            print("=" * 70)
            from experiments.exp6_speaker_adversarial import run_exp6
            run_exp6(force=args.force, quick=args.quick)

        if 8 in exps:
            print("\n" + "=" * 70)
            print("EXP8: SOTA Comparison")
            print("=" * 70)
            from experiments.exp8_sota_comparison import run_exp8
            run_exp8(force=args.force, quick=args.quick)

    print("\n[INFO] All done!")


if __name__ == "__main__":
    main()
