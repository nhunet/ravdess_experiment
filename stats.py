"""Statistical tests: paired t-test, Cohen's d, TOST, Wilcoxon."""

import numpy as np
from scipy import stats


def paired_ttest(scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    """Paired t-test on per-fold accuracy/F1."""
    scores_a, scores_b = np.asarray(scores_a), np.asarray(scores_b)
    t, p = stats.ttest_rel(scores_a, scores_b)
    diff = scores_a - scores_b
    d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0
    return {"t": float(t), "p": float(p), "cohens_d": float(d)}


def tost_test(scores_a: np.ndarray, scores_b: np.ndarray,
              delta: float = 2.0) -> dict:
    """
    Two One-Sided Tests for equivalence.
    delta = equivalence margin in percentage points.
    """
    scores_a, scores_b = np.asarray(scores_a), np.asarray(scores_b)
    diff = scores_a - scores_b
    n = len(diff)
    mean_d = np.mean(diff)
    se = np.std(diff, ddof=1) / np.sqrt(n)
    if se == 0:
        return {"p_tost": 0.0, "equivalent": True}
    t1 = (mean_d - (-delta)) / se
    t2 = (delta - mean_d) / se
    p1 = 1 - stats.t.cdf(t1, n - 1)
    p2 = 1 - stats.t.cdf(t2, n - 1)
    p_tost = max(p1, p2)
    return {"p_tost": float(p_tost), "equivalent": p_tost < 0.05}


def wilcoxon_test(scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    """Non-parametric alternative (minimum p for n=6 is 0.0312)."""
    scores_a, scores_b = np.asarray(scores_a), np.asarray(scores_b)
    diff = scores_a - scores_b
    if np.all(diff == 0):
        return {"stat": 0.0, "p": 1.0}
    stat, p = stats.wilcoxon(scores_a, scores_b)
    return {"stat": float(stat), "p": float(p)}


def format_result(name_a: str, name_b: str, metric: str,
                  scores_a: np.ndarray, scores_b: np.ndarray) -> str:
    """Format a comparison result as a readable string."""
    tt = paired_ttest(scores_a, scores_b)
    sig = "***" if tt["p"] < 0.001 else "**" if tt["p"] < 0.01 else "*" if tt["p"] < 0.05 else "n.s."
    return (
        f"{name_a} vs {name_b} ({metric}): "
        f"Δ={np.mean(scores_a) - np.mean(scores_b):+.2f}, "
        f"t={tt['t']:.3f}, p={tt['p']:.4f} {sig}, "
        f"d={tt['cohens_d']:.3f}"
    )
