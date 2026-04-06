"""配对 bootstrap 统计。"""

from __future__ import annotations

import numpy as np


def paired_differences(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs = np.asarray(lhs, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    if lhs.shape != rhs.shape:
        raise ValueError("配对样本形状必须一致")
    return lhs - rhs


def paired_bootstrap_ci(
    lhs: np.ndarray,
    rhs: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 0,
    ci: float = 95.0,
) -> dict[str, float]:
    diffs = paired_differences(lhs, rhs)
    rng = np.random.default_rng(seed)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample_indices = rng.integers(0, len(diffs), size=len(diffs))
        bootstrap_means.append(float(np.mean(diffs[sample_indices])))

    alpha = (100.0 - ci) / 2.0
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_low": float(np.percentile(bootstrap_means, alpha)),
        "ci_high": float(np.percentile(bootstrap_means, 100.0 - alpha)),
    }
