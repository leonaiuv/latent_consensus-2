"""本地 18-run 核心梯子 orchestrator。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from latent_consensus.analysis.bootstrap import paired_bootstrap_ci
from latent_consensus.analysis.gate import classify_gate2
from latent_consensus.analysis.phase1_gate import (
    is_gate1_open_for_phase2,
    load_gate1_report,
    summarize_gate1,
)
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
    return {
        "pairs": pair_reports,
        "recommended_pair": best_pair["label"],
        "recommended_gate": best_pair["gate"],
    }


def _load_summary(summary_path: Path) -> dict[str, object]:
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _can_resume_arithmetic(summary: dict[str, object]) -> bool:
    required_keys = {"experiment_id", "history", "test_metrics", "val_metrics", "ood_metrics"}
    return required_keys.issubset(summary)


def _can_resume_brs(summary: dict[str, object]) -> bool:
    required_keys = {
        "experiment_id",
        "id_accuracy",
        "ood_accuracy",
        "id_predictions",
        "ood_predictions",
    }
    return required_keys.issubset(summary)


def _load_or_run_arithmetic_experiment(
    experiment_id: str,
    configs_dir: Path,
    output_root: Path,
    train_samples: int,
    val_samples: int,
    runtime_mode: str,
    data_dir: Path | None,
    model_name: str | None,
    device: str,
    hf_endpoint: str | None,
    step_counts: tuple[int, ...] | None,
    train_limit_per_step: int | None,
    val_limit_per_step: int | None,
    test_limit_per_step: int | None,
    ood_limit_per_step: int | None,
    max_epochs: int | None,
    batch_size: int | None,
    gradient_accumulation_steps: int | None,
    learning_rate: float | None,
    seq_len: int | None,
    resume: bool,
) -> dict[str, object]:
    summary_path = output_root / experiment_id / "summary.json"
    if resume and summary_path.is_file():
        summary = _load_summary(summary_path)
        if _can_resume_arithmetic(summary):
            return summary
    return run_arithmetic_experiment(
        experiment_id=experiment_id,
        configs_dir=configs_dir,
        output_root=output_root,
        train_samples=train_samples,
        val_samples=val_samples,
        runtime_mode=runtime_mode,
        data_dir=data_dir,
        model_name=model_name,
        device=device,
        hf_endpoint=hf_endpoint,
        step_counts=step_counts,
        train_limit_per_step=train_limit_per_step,
        val_limit_per_step=val_limit_per_step,
        test_limit_per_step=test_limit_per_step,
        ood_limit_per_step=ood_limit_per_step,
        max_epochs=max_epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        seq_len=seq_len,
    )


def _load_or_run_brs_experiment(
    experiment_id: str,
    configs_dir: Path,
    output_root: Path,
    train_samples: int,
    val_samples: int,
    test_samples: int,
    runtime_mode: str,
    data_dir: Path | None,
    model_name: str | None,
    device: str,
    hf_endpoint: str | None,
    step_counts: tuple[int, ...] | None,
    train_limit_per_step: int | None,
    val_limit_per_step: int | None,
    test_limit_per_step: int | None,
    ood_limit_per_step: int | None,
    max_epochs: int | None,
    batch_size: int | None,
    gradient_accumulation_steps: int | None,
    learning_rate: float | None,
    seq_len: int | None,
    resume: bool,
) -> dict[str, object]:
    summary_path = output_root / experiment_id / "summary.json"
    if resume and summary_path.is_file():
        summary = _load_summary(summary_path)
        if _can_resume_brs(summary):
            return summary
    return run_brs_experiment(
        experiment_id=experiment_id,
        configs_dir=configs_dir,
        output_root=output_root,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        runtime_mode=runtime_mode,
        data_dir=data_dir,
        model_name=model_name,
        device=device,
        hf_endpoint=hf_endpoint,
        step_counts=step_counts,
        train_limit_per_step=train_limit_per_step,
        val_limit_per_step=val_limit_per_step,
        test_limit_per_step=test_limit_per_step,
        ood_limit_per_step=ood_limit_per_step,
        max_epochs=max_epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        seq_len=seq_len,
    )


def _resolve_gate1_report_path(output_root: Path, gate1_report_path: Path | None) -> Path:
    if gate1_report_path is not None:
        return Path(gate1_report_path)
    return Path(output_root) / "gate1_report.json"


def run_local_core_ladder(
    configs_dir: Path,
    output_root: Path,
    mode: str = "all",
    train_samples: int = 8,
    val_samples: int = 4,
    test_samples: int = 4,
    runtime_mode: str = "smoke",
    data_dir: Path | None = None,
    model_name: str | None = None,
    device: str = "cpu",
    hf_endpoint: str | None = None,
    step_counts: tuple[int, ...] | None = None,
    train_limit_per_step: int | None = None,
    val_limit_per_step: int | None = None,
    test_limit_per_step: int | None = None,
    ood_limit_per_step: int | None = None,
    max_epochs: int | None = None,
    batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
    learning_rate: float | None = None,
    seq_len: int | None = None,
    resume: bool = True,
    gate1_report_path: Path | None = None,
) -> dict[str, object]:
    output_root = Path(output_root)
    completed_experiments: list[str] = []
    arithmetic_results: dict[str, dict[str, object]] = {}
    brs_results: dict[str, dict[str, object]] = {}
    gate1_summary: dict[str, object] | None = None
    brs_blocked_by_gate1 = False

    if mode not in {"all", "arithmetic", "brs"}:
        raise ValueError("mode 仅支持 all / arithmetic / brs")

    arithmetic_output_root = output_root / "arithmetic_debug"
    brs_output_root = output_root / "brs_main"

    if mode in {"all", "arithmetic"}:
        for experiment_id in ARITHMETIC_EXPERIMENT_IDS:
            arithmetic_results[experiment_id] = _load_or_run_arithmetic_experiment(
                experiment_id=experiment_id,
                configs_dir=configs_dir,
                output_root=arithmetic_output_root,
                train_samples=train_samples,
                val_samples=val_samples,
                runtime_mode=runtime_mode,
                data_dir=data_dir,
                model_name=model_name,
                device=device,
                hf_endpoint=hf_endpoint,
                step_counts=step_counts,
                train_limit_per_step=train_limit_per_step,
                val_limit_per_step=val_limit_per_step,
                test_limit_per_step=test_limit_per_step,
                ood_limit_per_step=ood_limit_per_step,
                max_epochs=max_epochs,
                batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                seq_len=seq_len,
                resume=resume,
            )
            completed_experiments.append(experiment_id)

        gate1_summary = summarize_gate1(
            arithmetic_results=arithmetic_results,
            artifacts_root=arithmetic_output_root,
        )
        resolved_gate1_report_path = _resolve_gate1_report_path(
            output_root=output_root,
            gate1_report_path=gate1_report_path,
        )
        resolved_gate1_report_path.write_text(
            json.dumps(gate1_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    gate2_summary = None
    if mode in {"all", "brs"}:
        if mode == "brs":
            resolved_gate1_report_path = _resolve_gate1_report_path(
                output_root=output_root,
                gate1_report_path=gate1_report_path,
            )
            if not resolved_gate1_report_path.is_file():
                raise ValueError("进入 BRS 模式前必须提供已通过的 Gate 1 报告")
            gate1_summary = load_gate1_report(resolved_gate1_report_path)

        if not gate1_summary or not is_gate1_open_for_phase2(gate1_summary):
            brs_blocked_by_gate1 = True
        else:
            for experiment_id in BRS_EXPERIMENT_IDS:
                brs_results[experiment_id] = _load_or_run_brs_experiment(
                    experiment_id=experiment_id,
                    configs_dir=configs_dir,
                    output_root=brs_output_root,
                    train_samples=train_samples,
                    val_samples=val_samples,
                    test_samples=test_samples,
                    runtime_mode=runtime_mode,
                    data_dir=data_dir,
                    model_name=model_name,
                    device=device,
                    hf_endpoint=hf_endpoint,
                    step_counts=step_counts,
                    train_limit_per_step=train_limit_per_step,
                    val_limit_per_step=val_limit_per_step,
                    test_limit_per_step=test_limit_per_step,
                    ood_limit_per_step=ood_limit_per_step,
                    max_epochs=max_epochs,
                    batch_size=batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    learning_rate=learning_rate,
                    seq_len=seq_len,
                    resume=resume,
                )
                completed_experiments.append(experiment_id)
            gate2_summary = _summarize_gate2(brs_results)

    report = {
        "mode": mode,
        "completed_experiments": completed_experiments,
        "gate1_summary": gate1_summary,
        "gate2_summary": gate2_summary,
        "brs_blocked_by_gate1": brs_blocked_by_gate1,
    }
    report_path = output_root / "local_core_ladder_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
