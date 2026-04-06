from pathlib import Path

import pytest

from latent_consensus.training.local_core_ladder import run_local_core_ladder


ROOT = Path(__file__).resolve().parents[2]


def test_local_core_ladder_runs_arithmetic_mode(tmp_path, monkeypatch) -> None:
    def fake_run_arithmetic_experiment(**kwargs):
        experiment_id = kwargs["experiment_id"]
        experiment_dir = kwargs["output_root"] / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        for file_name in (
            "summary.json",
            "best_checkpoint.pt",
            "final_checkpoint.pt",
            "test_predictions.jsonl",
        ):
            (experiment_dir / file_name).write_text("{}", encoding="utf-8")
        return {
            "experiment_id": experiment_id,
            "model_family": "cot" if experiment_id == "EXP-A01" else "lc1",
            "history": {
                "epochs": [
                    {
                        "epoch": 0,
                        "train_loss": 1.0,
                        "val_loss": 0.5,
                        "val_exact_match": 0.8,
                    }
                ],
                "best_epoch": 0,
            },
            "test_metrics": {
                "exact_match": 0.8,
                "step_accuracy": {"2": 0.9, "4": 0.8, "6": 0.7},
            },
            "val_metrics": {
                "exact_match": 0.75,
                "step_accuracy": {"2": 0.9, "4": 0.8, "6": 0.7},
            },
            "ood_metrics": {
                "exact_match": 0.7,
                "step_accuracy": {"2": 0.88, "4": 0.78, "6": 0.68},
            },
        }

    monkeypatch.setattr(
        "latent_consensus.training.local_core_ladder.run_arithmetic_experiment",
        fake_run_arithmetic_experiment,
    )

    report = run_local_core_ladder(
        configs_dir=ROOT / "configs",
        output_root=tmp_path,
        mode="arithmetic",
        train_samples=6,
        val_samples=4,
        test_samples=4,
    )

    assert report["mode"] == "arithmetic"
    assert len(report["completed_experiments"]) == 6
    assert report["gate1_summary"]["passed"] is True


def test_local_core_ladder_blocks_all_mode_when_gate1_fails(tmp_path, monkeypatch) -> None:
    def fake_run_arithmetic_experiment(**kwargs):
        experiment_id = kwargs["experiment_id"]
        experiment_dir = kwargs["output_root"] / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        for file_name in (
            "summary.json",
            "best_checkpoint.pt",
            "final_checkpoint.pt",
            "test_predictions.jsonl",
        ):
            (experiment_dir / file_name).write_text("{}", encoding="utf-8")
        return {
            "experiment_id": experiment_id,
            "model_family": "lc1",
            "history": {
                "epochs": [
                    {
                        "epoch": 0,
                        "train_loss": 1.0,
                        "val_loss": 0.9,
                        "val_exact_match": 0.6,
                    }
                ],
                "best_epoch": 0,
            },
            "test_metrics": {
                "exact_match": 0.6,
                "step_accuracy": {"2": 0.7, "4": 0.7, "6": 0.72},
            },
            "val_metrics": {
                "exact_match": 0.6,
                "step_accuracy": {"2": 0.7, "4": 0.7, "6": 0.72},
            },
            "ood_metrics": {
                "exact_match": 0.58,
                "step_accuracy": {"2": 0.68, "4": 0.67, "6": 0.71},
            },
        }

    def fake_run_brs_experiment(**kwargs):
        raise AssertionError("Gate 1 未通过时不应执行 BRS")

    monkeypatch.setattr(
        "latent_consensus.training.local_core_ladder.run_arithmetic_experiment",
        fake_run_arithmetic_experiment,
    )
    monkeypatch.setattr(
        "latent_consensus.training.local_core_ladder.run_brs_experiment",
        fake_run_brs_experiment,
    )

    report = run_local_core_ladder(
        configs_dir=ROOT / "configs",
        output_root=tmp_path,
        mode="all",
        train_samples=6,
        val_samples=4,
        test_samples=4,
    )

    assert report["gate1_summary"]["passed"] is False
    assert report["brs_blocked_by_gate1"] is True
    assert report["gate2_summary"] is None


def test_local_core_ladder_requires_gate1_report_for_brs_mode(tmp_path) -> None:
    with pytest.raises(ValueError, match="Gate 1"):
        run_local_core_ladder(
            configs_dir=ROOT / "configs",
            output_root=tmp_path,
            mode="brs",
            train_samples=6,
            val_samples=4,
            test_samples=4,
        )


def test_local_core_ladder_allows_brs_mode_with_gate1_waiver(tmp_path, monkeypatch) -> None:
    gate1_report_path = tmp_path / "gate1_report.json"
    gate1_report_path.write_text(
        (
            '{"stage":"phase1_gate1","passed":false,"waived_for_phase2":true,'
            '"waiver_reason":"Arithmetic 已出现天花板效应，研究决策允许进入 BRS 主线。"}'
        ),
        encoding="utf-8",
    )

    def fake_run_brs_experiment(**kwargs):
        experiment_id = kwargs["experiment_id"]
        experiment_dir = kwargs["output_root"] / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "experiment_id": experiment_id,
            "id_accuracy": 0.7,
            "ood_accuracy": 0.6,
            "id_predictions": [1, 0],
            "ood_predictions": [1, 0],
            "id_targets": [1, 0],
            "ood_targets": [1, 0],
        }
        (experiment_dir / "summary.json").write_text("{}", encoding="utf-8")
        return summary

    monkeypatch.setattr(
        "latent_consensus.training.local_core_ladder.run_brs_experiment",
        fake_run_brs_experiment,
    )

    report = run_local_core_ladder(
        configs_dir=ROOT / "configs",
        output_root=tmp_path,
        mode="brs",
        gate1_report_path=gate1_report_path,
        resume=False,
    )

    assert report["brs_blocked_by_gate1"] is False
    assert len(report["completed_experiments"]) == 12


def test_local_core_ladder_resume_skips_completed_experiment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    arithmetic_root = tmp_path / "arithmetic_debug"
    experiment_dir = arithmetic_root / "EXP-A01"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "summary.json").write_text(
        (
            '{"experiment_id":"EXP-A01","model_family":"cot","history":{"epochs":[{"epoch":0,"train_loss":1.0,"val_loss":0.5,"val_exact_match":0.8}],"best_epoch":0},'
            '"test_metrics":{"exact_match":0.8,"step_accuracy":{"2":0.9,"4":0.8,"6":0.7}},'
            '"val_metrics":{"exact_match":0.75,"step_accuracy":{"2":0.9,"4":0.8,"6":0.7}},'
            '"ood_metrics":{"exact_match":0.7,"step_accuracy":{"2":0.88,"4":0.78,"6":0.68}}}'
        ),
        encoding="utf-8",
    )
    for file_name in (
        "best_checkpoint.pt",
        "final_checkpoint.pt",
        "test_predictions.jsonl",
    ):
        (experiment_dir / file_name).write_text("{}", encoding="utf-8")

    calls: list[str] = []

    def fake_run_arithmetic_experiment(**kwargs):
        experiment_id = kwargs["experiment_id"]
        calls.append(experiment_id)
        experiment_dir = kwargs["output_root"] / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        for file_name in (
            "summary.json",
            "best_checkpoint.pt",
            "final_checkpoint.pt",
            "test_predictions.jsonl",
        ):
            (experiment_dir / file_name).write_text("{}", encoding="utf-8")
        return {
            "experiment_id": experiment_id,
            "model_family": "cot" if experiment_id == "EXP-A01" else "lc1",
            "history": {
                "epochs": [
                    {
                        "epoch": 0,
                        "train_loss": 1.0,
                        "val_loss": 0.5,
                        "val_exact_match": 0.8,
                    }
                ],
                "best_epoch": 0,
            },
            "test_metrics": {
                "exact_match": 0.8,
                "step_accuracy": {"2": 0.9, "4": 0.8, "6": 0.7},
            },
            "val_metrics": {
                "exact_match": 0.75,
                "step_accuracy": {"2": 0.9, "4": 0.8, "6": 0.7},
            },
            "ood_metrics": {
                "exact_match": 0.7,
                "step_accuracy": {"2": 0.88, "4": 0.78, "6": 0.68},
            },
        }

    monkeypatch.setattr(
        "latent_consensus.training.local_core_ladder.run_arithmetic_experiment",
        fake_run_arithmetic_experiment,
    )

    report = run_local_core_ladder(
        configs_dir=ROOT / "configs",
        output_root=tmp_path,
        mode="arithmetic",
        resume=True,
    )

    assert "EXP-A01" in report["completed_experiments"]
    assert "EXP-A01" not in calls
