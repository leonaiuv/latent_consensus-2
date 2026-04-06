"""Phase 1 Gate 1 的正式判定逻辑。"""

from __future__ import annotations

import json
import math
from pathlib import Path


REQUIRED_ARITHMETIC_EXPERIMENT_IDS = (
    "EXP-A01",
    "EXP-A02",
    "EXP-A03",
    "EXP-A04",
    "EXP-A05",
    "EXP-A06",
)

REQUIRED_ARTIFACT_FILES = (
    "summary.json",
    "best_checkpoint.pt",
    "final_checkpoint.pt",
    "test_predictions.jsonl",
)


def summarize_gate1(
    arithmetic_results: dict[str, dict[str, object]],
    artifacts_root: Path,
    min_step: int = 2,
    max_step: int = 6,
    min_pass_count: int = 4,
    min_mean_delta: float = 0.05,
) -> dict[str, object]:
    """根据 Arithmetic 6-run 结果生成 Gate 1 判定报告。"""

    missing_experiment_ids = [
        experiment_id
        for experiment_id in REQUIRED_ARITHMETIC_EXPERIMENT_IDS
        if experiment_id not in arithmetic_results
    ]
    if missing_experiment_ids:
        raise ValueError(
            f"缺少 Arithmetic 实验结果：{', '.join(missing_experiment_ids)}"
        )

    stability_checks = {
        experiment_id: _summarize_stability(arithmetic_results[experiment_id])
        for experiment_id in REQUIRED_ARITHMETIC_EXPERIMENT_IDS
    }
    gradient_check = _summarize_gradient(
        arithmetic_results=arithmetic_results,
        min_step=min_step,
        max_step=max_step,
        min_pass_count=min_pass_count,
        min_mean_delta=min_mean_delta,
    )
    artifact_check = _summarize_artifacts(artifacts_root=Path(artifacts_root))

    cot_lc1_stable = (
        stability_checks["EXP-A01"]["stable"] and stability_checks["EXP-A02"]["stable"]
    )
    shared_runs_complete = all(
        stability_checks[experiment_id]["stable"]
        for experiment_id in ("EXP-A03", "EXP-A04", "EXP-A05", "EXP-A06")
    )
    artifacts_complete = artifact_check["passed"]

    passed = (
        cot_lc1_stable
        and shared_runs_complete
        and gradient_check["passed"]
        and artifacts_complete
    )

    notes: list[str] = []
    if not cot_lc1_stable:
        notes.append("CoT 或 LC-1 未达到稳定收敛的工程代理标准。")
    if not shared_runs_complete:
        notes.append("LC-2-S / Ind-2-S / LC-3-S / Ind-3-S 中至少有一条训练链路不稳定。")
    if not gradient_check["passed"]:
        notes.append(
            "2-step 相对 6-step 的难度梯度仍不清晰，当前不能视为正式 Gate 1 通过。"
        )
    if not artifacts_complete:
        notes.append("至少一个 Arithmetic 实验缺少 summary/checkpoint/predictions 产物。")
    if passed:
        notes.append("Phase 1 已满足进入 BRS 主线的工程 gate 条件。")

    return {
        "stage": "phase1_gate1",
        "passed": passed,
        "cot_lc1_stable": cot_lc1_stable,
        "shared_runs_complete": shared_runs_complete,
        "artifacts_complete": artifacts_complete,
        "stability_checks": stability_checks,
        "gradient_check": gradient_check,
        "artifact_check": artifact_check,
        "notes": notes,
    }


def load_gate1_report(report_path: Path) -> dict[str, object]:
    return json.loads(Path(report_path).read_text(encoding="utf-8"))


def apply_gate1_research_waiver(
    report: dict[str, object],
    reason: str,
) -> dict[str, object]:
    if not reason.strip():
        raise ValueError("研究豁免必须提供非空原因")

    waived_report = dict(report)
    waived_report["waived_for_phase2"] = True
    waived_report["waiver_reason"] = reason.strip()
    waived_report["phase2_entry_mode"] = "research_waiver"
    notes = list(waived_report.get("notes", []))
    notes.append(
        "研究决策：接受 Arithmetic 作为链路验证已完成，允许进入 BRS 主线。"
    )
    waived_report["notes"] = notes
    return waived_report


def is_gate1_open_for_phase2(report: dict[str, object]) -> bool:
    return bool(report.get("passed", False) or report.get("waived_for_phase2", False))


def _summarize_stability(summary: dict[str, object]) -> dict[str, object]:
    history = summary.get("history", {})
    epochs = history.get("epochs", []) if isinstance(history, dict) else []
    has_epochs = len(epochs) > 0
    finite_losses = has_epochs and all(
        _is_finite_number(epoch.get("train_loss"))
        and _is_finite_number(epoch.get("val_loss"))
        for epoch in epochs
        if isinstance(epoch, dict)
    )
    best_epoch = history.get("best_epoch") if isinstance(history, dict) else None
    stable = has_epochs and finite_losses and best_epoch is not None
    return {
        "stable": stable,
        "epoch_count": len(epochs),
        "best_epoch": best_epoch,
        "finite_losses": finite_losses,
    }


def _summarize_gradient(
    arithmetic_results: dict[str, dict[str, object]],
    min_step: int,
    max_step: int,
    min_pass_count: int,
    min_mean_delta: float,
) -> dict[str, object]:
    per_experiment: list[dict[str, object]] = []
    deltas: list[float] = []

    for experiment_id in REQUIRED_ARITHMETIC_EXPERIMENT_IDS:
        step_accuracy = arithmetic_results[experiment_id]["test_metrics"]["step_accuracy"]
        low_step_accuracy = float(step_accuracy[str(min_step)])
        high_step_accuracy = float(step_accuracy[str(max_step)])
        delta = low_step_accuracy - high_step_accuracy
        deltas.append(delta)
        per_experiment.append(
            {
                "experiment_id": experiment_id,
                "step_accuracy_low": low_step_accuracy,
                "step_accuracy_high": high_step_accuracy,
                "delta": delta,
                "passed": delta > 0,
            }
        )

    pass_count = sum(1 for item in per_experiment if item["passed"])
    mean_delta = sum(deltas) / len(deltas)
    passed = pass_count >= min_pass_count and mean_delta >= min_mean_delta
    return {
        "min_step": min_step,
        "max_step": max_step,
        "min_pass_count": min_pass_count,
        "min_mean_delta": min_mean_delta,
        "pass_count": pass_count,
        "mean_delta": mean_delta,
        "passed": passed,
        "per_experiment": per_experiment,
    }


def _summarize_artifacts(artifacts_root: Path) -> dict[str, object]:
    missing_by_experiment: dict[str, list[str]] = {}
    for experiment_id in REQUIRED_ARITHMETIC_EXPERIMENT_IDS:
        experiment_dir = artifacts_root / experiment_id
        missing_files = [
            file_name
            for file_name in REQUIRED_ARTIFACT_FILES
            if not (experiment_dir / file_name).is_file()
        ]
        if missing_files:
            missing_by_experiment[experiment_id] = missing_files
    return {
        "passed": not missing_by_experiment,
        "missing_by_experiment": missing_by_experiment,
    }


def _is_finite_number(value: object) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
