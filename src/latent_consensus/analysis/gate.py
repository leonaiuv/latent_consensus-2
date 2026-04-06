"""Gate 2 判定逻辑。"""

from __future__ import annotations


def classify_gate2(
    id_ci_low: float,
    id_ci_high: float,
    id_mean_diff: float,
    ood_mean_diff: float,
    seed_directions: list[int],
) -> dict[str, object]:
    if id_ci_low > 0 and ood_mean_diff >= 0:
        label = "Positive"
    elif id_mean_diff <= 0 and ood_mean_diff <= 0 and all(direction <= 0 for direction in seed_directions):
        label = "Negative"
    else:
        label = "Weak"

    return {
        "label": label,
        "id_ci_low": id_ci_low,
        "id_ci_high": id_ci_high,
        "id_mean_diff": id_mean_diff,
        "ood_mean_diff": ood_mean_diff,
        "seed_directions": seed_directions,
    }
