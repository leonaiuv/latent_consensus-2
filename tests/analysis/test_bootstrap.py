import numpy as np

from latent_consensus.analysis.bootstrap import paired_bootstrap_ci, paired_differences


def test_paired_differences_returns_elementwise_gap() -> None:
    lhs = np.array([1.0, 0.0, 1.0, 1.0])
    rhs = np.array([0.0, 0.0, 1.0, 0.0])

    diff = paired_differences(lhs, rhs)

    assert np.allclose(diff, np.array([1.0, 0.0, 0.0, 1.0]))


def test_paired_bootstrap_ci_is_reproducible() -> None:
    lhs = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    rhs = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0])

    first = paired_bootstrap_ci(lhs, rhs, n_bootstrap=500, seed=7)
    second = paired_bootstrap_ci(lhs, rhs, n_bootstrap=500, seed=7)

    assert first == second
    assert first["mean_diff"] > 0
