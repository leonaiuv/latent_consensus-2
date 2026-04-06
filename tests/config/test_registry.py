from pathlib import Path

from latent_consensus.config.registry import (
    get_arithmetic_experiment_spec,
    get_brs_experiment_spec,
)


ROOT = Path(__file__).resolve().parents[2]


def test_registry_resolves_ind_experiment_as_observe_off_shared_variant() -> None:
    spec = get_arithmetic_experiment_spec("EXP-A04", ROOT / "configs")

    assert spec.experiment_id == "EXP-A04"
    assert spec.seed == 42
    assert spec.dataset_name == "arithmetic_debug"
    assert spec.model_family == "ind_n_shared"
    assert spec.resolved_config["model"]["observe"] == "off"
    assert spec.resolved_config["model"]["n_processors"] == 2


def test_registry_resolves_lc3_shared_experiment() -> None:
    spec = get_arithmetic_experiment_spec("EXP-A05", ROOT / "configs")

    assert spec.experiment_id == "EXP-A05"
    assert spec.model_family == "lcn_shared"
    assert spec.resolved_config["model"]["observe"] == "on"
    assert spec.resolved_config["model"]["n_processors"] == 3


def test_registry_resolves_brs_shared_experiment_against_phase2_baseline() -> None:
    spec = get_brs_experiment_spec("EXP-B05", ROOT / "configs")

    assert spec.experiment_id == "EXP-B05"
    assert spec.dataset_name == "brs"
    assert spec.model_family == "lcn_shared"
    assert spec.resolved_config["experiment"]["phase"] == "phase2"
    assert spec.resolved_config["training"]["seq_len"] == 256
    assert spec.resolved_config["model"]["observe"] == "on"
    assert spec.resolved_config["model"]["n_processors"] == 2


def test_registry_resolves_brs_independent_experiment_against_phase2_baseline() -> None:
    spec = get_brs_experiment_spec("EXP-B11", ROOT / "configs")

    assert spec.experiment_id == "EXP-B11"
    assert spec.dataset_name == "brs"
    assert spec.model_family == "ind_n_shared"
    assert spec.resolved_config["experiment"]["phase"] == "phase2"
    assert spec.resolved_config["training"]["seq_len"] == 256
    assert spec.resolved_config["model"]["observe"] == "off"
    assert spec.resolved_config["model"]["n_processors"] == 3
