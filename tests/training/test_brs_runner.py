from pathlib import Path

from latent_consensus.training.brs_runner import run_brs_experiment


ROOT = Path(__file__).resolve().parents[2]


def test_run_brs_experiment_writes_summary_and_checkpoints(tmp_path) -> None:
    result = run_brs_experiment(
        experiment_id="EXP-B05",
        configs_dir=ROOT / "configs",
        output_root=tmp_path,
        train_samples=8,
        val_samples=4,
        test_samples=4,
        seed=42,
    )

    experiment_dir = tmp_path / "EXP-B05"
    assert result["experiment_id"] == "EXP-B05"
    assert result["model_family"] == "lcn_shared"
    assert (experiment_dir / "summary.json").is_file()
    assert (experiment_dir / "best_checkpoint.json").is_file()
    assert (experiment_dir / "final_checkpoint.json").is_file()


def test_run_brs_experiment_supports_ind_shared_variant(tmp_path) -> None:
    result = run_brs_experiment(
        experiment_id="EXP-B07",
        configs_dir=ROOT / "configs",
        output_root=tmp_path,
        train_samples=8,
        val_samples=4,
        test_samples=4,
        seed=42,
    )

    assert result["experiment_id"] == "EXP-B07"
    assert result["model_family"] == "ind_n_shared"
