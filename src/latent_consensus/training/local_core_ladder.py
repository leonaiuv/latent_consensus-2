"""本地 18-run 核心梯子 orchestrator。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from latent_consensus.analysis.bootstrap import paired_bootstrap_ci
from latent_consensus.analysis.gate import classify_gate2
from latent_consensus.training.arithmetic_runner import run_arithmetic_experiment
from latent_consensus.training.brs_runner import run_brs_experiment


ARITHMETIC_EXPERIMENT_IDS = [
    "EXP-A01",
    "EXP-A02",
    "EXP-A03",
    "EXP-A04",
    "EXP-A05",
    "EXP-A06",
]

BRS_EXPERIMENT_IDS = [
    "EXP-B01",
    "EXP-B02",
    "EXP-B03",
    "EXP-B04",
    "EXP-B05",
    "EXP-B06",
    "EXP-B07",
    "EXP-B08",
    "EXP-B09",
    "EXP-B10",
    "EXP-B11",
    "EXP-B12",
]


def _seed_direction(lhs_accuracy: float, rhs_accuracy: float) -> int:
    if lhs_accuracy > rhs_accuracy:
        return 1
    if lhs_accuracy < rhs_accuracy:
        return -1
    return 0


def _summarize_gate2(brs_results: dict[str, dict[str, object]]) -> dict[str, object]:
    pair_specs = [
        ("LC-2-S", ["EXP-B05", "EXP-B06"], ["EXP-B07", "EXP-B08"]),
        ("LC-3-S", ["EXP-B09", "EXP-B10"], ["EXP-B11", "EXP-B12"]),
    ]
    pair_reports = []

    for label, lc_ids, ind_ids in pair_specs:
        id_lc = []
        id_ind = []
        ood_lc = []
        ood_ind = []
        seed_directions = []
        for lc_id, ind_id in zip(lc_ids, ind_ids):
            lc_summary = brs_results[lc_id]
            ind_summary = brs_results[ind_id]
            id_lc.extend(lc_summary["id_predictions"])
            id_ind.extend(ind_summary["id_predictions"])
            ood_lc.extend(lc_summary["ood_predictions"])
            ood_ind.extend(ind_summary["ood_predictions"])
            seed_directions.append(
                _seed_direction(lc_summary["id_accuracy"], ind_summary["id_accuracy"])
            )

        id_ci = paired_bootstrap_ci(
            lhs=np.asarray(id_lc, dtype=float),
            rhs=np.asarray(id_ind, dtype=float),
            n_bootstrap=500,
            seed=7,
        )
        ood_ci = paired_bootstrap_ci(
            lhs=np.asarray(ood_lc, dtype=float),
            rhs=np.asarray(ood_ind, dtype=float),
            n_bootstrap=500,
            seed=11,
        )
        gate = classify_gate2(
            id_ci_low=id_ci["ci_low"],
            id_ci_high=id_ci["ci_high"],
            id_mean_diff=id_ci["mean_diff"],
            ood_mean_diff=ood_ci["mean_diff"],
            seed_directions=seed_directions,
        )
        pair_reports.append(
            {
                "label": label,
                "id_ci": id_ci,
                "ood_ci": ood_ci,
                "gate": gate,
            }
        )

    best_pair = max(pair_reports, key=lambda item: item["id_ci"]["mean_diff"])
    return {"pairs": pair_reports, "recommended_pair": best_pair["label"], "recommended_gate": best_pair["gate"]}


def run_local_core_ladder(
    configs_dir: Path,
    output_root: Path,
    mode: str = "all",
    train_samples: int = 8,
    val_samples: int = 4,
    test_samples: int = 4,
) -> dict[str, object]:
    output_root = Path(output_root)
    completed_experiments: list[str] = []
    arithmetic_results: dict[str, dict[str, object]] = {}
    brs_results: dict[str, dict[str, object]] = {}

    if mode not in {"all", "arithmetic", "brs"}:
        raise ValueError("mode 仅支持 all / arithmetic / brs")

    if mode in {"all", "arithmetic"}:
        for experiment_id in ARITHMETIC_EXPERIMENT_IDS:
            arithmetic_results[experiment_id] = run_arithmetic_experiment(
                experiment_id=experiment_id,
                configs_dir=configs_dir,
                output_root=output_root / "arithmetic_debug",
                train_samples=train_samples,
                val_samples=val_samples,
            )
            completed_experiments.append(experiment_id)

    gate2_summary = None
    if mode in {"all", "brs"}:
        for experiment_id in BRS_EXPERIMENT_IDS:
            brs_results[experiment_id] = run_brs_experiment(
                experiment_id=experiment_id,
                configs_dir=configs_dir,
                output_root=output_root / "brs_main",
                train_samples=train_samples,
                val_samples=val_samples,
                test_samples=test_samples,
            )
            completed_experiments.append(experiment_id)
        gate2_summary = _summarize_gate2(brs_results)

    report = {
        "mode": mode,
        "completed_experiments": completed_experiments,
        "gate2_summary": gate2_summary,
    }
    report_path = output_root / "local_core_ladder_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
