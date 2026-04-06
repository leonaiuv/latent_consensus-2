from pathlib import Path

from latent_consensus.training.local_core_ladder import run_local_core_ladder


ROOT = Path(__file__).resolve().parents[2]


def test_local_core_ladder_runs_arithmetic_mode(tmp_path) -> None:
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


def test_local_core_ladder_runs_brs_mode(tmp_path) -> None:
    report = run_local_core_ladder(
        configs_dir=ROOT / "configs",
        output_root=tmp_path,
        mode="brs",
        train_samples=6,
        val_samples=4,
        test_samples=4,
    )

    assert report["mode"] == "brs"
    assert len(report["completed_experiments"]) == 12
