from latent_consensus.analysis.gate import classify_gate2


def test_gate2_positive_when_ci_above_zero_and_ood_non_negative() -> None:
    result = classify_gate2(
        id_ci_low=0.05,
        id_ci_high=0.20,
        id_mean_diff=0.12,
        ood_mean_diff=0.01,
        seed_directions=[1, 1],
    )

    assert result["label"] == "Positive"


def test_gate2_weak_when_ci_crosses_zero() -> None:
    result = classify_gate2(
        id_ci_low=-0.02,
        id_ci_high=0.10,
        id_mean_diff=0.03,
        ood_mean_diff=0.01,
        seed_directions=[1, -1],
    )

    assert result["label"] == "Weak"


def test_gate2_negative_when_all_directions_non_positive() -> None:
    result = classify_gate2(
        id_ci_low=-0.10,
        id_ci_high=-0.01,
        id_mean_diff=-0.04,
        ood_mean_diff=-0.02,
        seed_directions=[-1, -1],
    )

    assert result["label"] == "Negative"
