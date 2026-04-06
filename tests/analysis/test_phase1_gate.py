from pathlib import Path

import pytest

from latent_consensus.analysis.phase1_gate import (
    apply_gate1_research_waiver,
    is_gate1_open_for_phase2,
    summarize_gate1,
)


def _build_summary(
    experiment_id: str,
    model_family: str,
    step2_accuracy: float,
    step6_accuracy: float,
) -> dict[str, object]:
    return {
        "experiment_id": experiment_id,
        "model_family": model_family,
        "history": {
            "epochs": [
                {
                    "epoch": 0,
                    "train_loss": 1.0,
                    "val_loss": 0.8,
                    "val_exact_match": 0.6,
                },
                {
                    "epoch": 1,
                    "train_loss": 0.7,
                    "val_loss": 0.5,
                    "val_exact_match": 0.8,
                },
            ],
            "best_epoch": 1,
        },
        "test_metrics": {
            "exact_match": 0.8,
            "step_accuracy": {
                "2": step2_accuracy,
                "4": 0.75,
                "6": step6_accuracy,
            },
        },
        "val_metrics": {
            "exact_match": 0.75,
            "step_accuracy": {
                "2": step2_accuracy,
                "4": 0.75,
                "6": step6_accuracy,
            },
        },
        "ood_metrics": {
            "exact_match": 0.72,
            "step_accuracy": {
                "2": step2_accuracy,
                "4": 0.7,
                "6": step6_accuracy,
            },
        },
    }


def _write_artifacts(root: Path, experiment_id: str) -> None:
    experiment_dir = root / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    for file_name in (
        "summary.json",
        "best_checkpoint.pt",
        "final_checkpoint.pt",
        "test_predictions.jsonl",
    ):
        (experiment_dir / file_name).write_text("{}", encoding="utf-8")


def test_summarize_gate1_passes_when_gradient_clear_and_artifacts_complete(
    tmp_path: Path,
) -> None:
    summaries = {
        "EXP-A01": _build_summary("EXP-A01", "cot", 0.95, 0.70),
        "EXP-A02": _build_summary("EXP-A02", "lc1", 0.88, 0.65),
        "EXP-A03": _build_summary("EXP-A03", "lcn_shared", 0.90, 0.72),
        "EXP-A04": _build_summary("EXP-A04", "ind_n_shared", 0.86, 0.68),
        "EXP-A05": _build_summary("EXP-A05", "lcn_shared", 0.91, 0.69),
        "EXP-A06": _build_summary("EXP-A06", "ind_n_shared", 0.84, 0.66),
    }
    for experiment_id in summaries:
        _write_artifacts(tmp_path, experiment_id)

    report = summarize_gate1(
        arithmetic_results=summaries,
        artifacts_root=tmp_path,
    )

    assert report["passed"] is True
    assert report["gradient_check"]["passed"] is True
    assert report["gradient_check"]["pass_count"] == 6
    assert report["artifacts_complete"] is True


def test_summarize_gate1_fails_when_gradient_not_clear(tmp_path: Path) -> None:
    summaries = {
        "EXP-A01": _build_summary("EXP-A01", "cot", 0.80, 0.82),
        "EXP-A02": _build_summary("EXP-A02", "lc1", 0.70, 0.72),
        "EXP-A03": _build_summary("EXP-A03", "lcn_shared", 0.75, 0.75),
        "EXP-A04": _build_summary("EXP-A04", "ind_n_shared", 0.81, 0.83),
        "EXP-A05": _build_summary("EXP-A05", "lcn_shared", 0.79, 0.80),
        "EXP-A06": _build_summary("EXP-A06", "ind_n_shared", 0.78, 0.79),
    }
    for experiment_id in summaries:
        _write_artifacts(tmp_path, experiment_id)

    report = summarize_gate1(
        arithmetic_results=summaries,
        artifacts_root=tmp_path,
    )

    assert report["passed"] is False
    assert report["gradient_check"]["passed"] is False
    assert "2-step" in "\n".join(report["notes"])


def test_summarize_gate1_requires_full_experiment_matrix(tmp_path: Path) -> None:
    summaries = {
        "EXP-A01": _build_summary("EXP-A01", "cot", 0.9, 0.7),
    }
    _write_artifacts(tmp_path, "EXP-A01")

    with pytest.raises(ValueError, match="缺少 Arithmetic 实验结果"):
        summarize_gate1(
            arithmetic_results=summaries,
            artifacts_root=tmp_path,
        )


def test_apply_gate1_research_waiver_opens_phase2_with_reason(tmp_path: Path) -> None:
    summaries = {
        "EXP-A01": _build_summary("EXP-A01", "cot", 0.80, 0.82),
        "EXP-A02": _build_summary("EXP-A02", "lc1", 0.70, 0.72),
        "EXP-A03": _build_summary("EXP-A03", "lcn_shared", 0.75, 0.75),
        "EXP-A04": _build_summary("EXP-A04", "ind_n_shared", 0.81, 0.83),
        "EXP-A05": _build_summary("EXP-A05", "lcn_shared", 0.79, 0.80),
        "EXP-A06": _build_summary("EXP-A06", "ind_n_shared", 0.78, 0.79),
    }
    for experiment_id in summaries:
        _write_artifacts(tmp_path, experiment_id)

    report = summarize_gate1(
        arithmetic_results=summaries,
        artifacts_root=tmp_path,
    )
    waived_report = apply_gate1_research_waiver(
        report,
        reason="Arithmetic 已出现天花板效应，转入 BRS 主线验证。",
    )

    assert report["passed"] is False
    assert waived_report["waived_for_phase2"] is True
    assert "天花板效应" in waived_report["waiver_reason"]
    assert is_gate1_open_for_phase2(waived_report) is True
