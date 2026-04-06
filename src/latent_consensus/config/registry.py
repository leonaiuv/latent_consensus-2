"""Arithmetic-Debug 实验注册表。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from latent_consensus.config.loader import load_config


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    dataset_name: str
    model_family: str
    seed: int
    resolved_config: dict[str, Any]


ARITHMETIC_EXPERIMENTS = {
    "EXP-A01": {
        "dataset_name": "arithmetic_debug",
        "model_family": "cot",
        "seed": 42,
        "config_file": "arithmetic_debug.yaml",
        "model_overrides": {"name": "cot", "n_processors": 1, "observe": "off"},
    },
    "EXP-A02": {
        "dataset_name": "arithmetic_debug",
        "model_family": "lc1",
        "seed": 42,
        "config_file": "arithmetic_debug.yaml",
        "model_overrides": {"name": "lc1", "n_processors": 1, "observe": "on"},
    },
    "EXP-A03": {
        "dataset_name": "arithmetic_debug",
        "model_family": "lcn_shared",
        "seed": 42,
        "config_file": "lc2_shared.yaml",
        "model_overrides": {"observe": "on"},
    },
    "EXP-A04": {
        "dataset_name": "arithmetic_debug",
        "model_family": "ind_n_shared",
        "seed": 42,
        "config_file": "lc2_shared.yaml",
        "model_overrides": {"name": "ind2_shared", "observe": "off"},
    },
    "EXP-A05": {
        "dataset_name": "arithmetic_debug",
        "model_family": "lcn_shared",
        "seed": 42,
        "config_file": "lc3_shared.yaml",
        "model_overrides": {"observe": "on"},
    },
    "EXP-A06": {
        "dataset_name": "arithmetic_debug",
        "model_family": "ind_n_shared",
        "seed": 42,
        "config_file": "lc3_shared.yaml",
        "model_overrides": {"name": "ind3_shared", "observe": "off"},
    },
}


BRS_EXPERIMENTS = {
    "EXP-B01": {
        "dataset_name": "brs",
        "model_family": "cot",
        "seed": 42,
        "config_file": "brs_main.yaml",
        "model_overrides": {"name": "cot", "n_processors": 1, "observe": "off"},
    },
    "EXP-B02": {
        "dataset_name": "brs",
        "model_family": "cot",
        "seed": 123,
        "config_file": "brs_main.yaml",
        "model_overrides": {"name": "cot", "n_processors": 1, "observe": "off"},
    },
    "EXP-B03": {
        "dataset_name": "brs",
        "model_family": "lc1",
        "seed": 42,
        "config_file": "brs_main.yaml",
        "model_overrides": {"name": "lc1", "n_processors": 1, "observe": "on"},
    },
    "EXP-B04": {
        "dataset_name": "brs",
        "model_family": "lc1",
        "seed": 123,
        "config_file": "brs_main.yaml",
        "model_overrides": {"name": "lc1", "n_processors": 1, "observe": "on"},
    },
    "EXP-B05": {
        "dataset_name": "brs",
        "model_family": "lcn_shared",
        "seed": 42,
        "config_file": "lc2_shared.yaml",
        "model_overrides": {"observe": "on"},
    },
    "EXP-B06": {
        "dataset_name": "brs",
        "model_family": "lcn_shared",
        "seed": 123,
        "config_file": "lc2_shared.yaml",
        "model_overrides": {"observe": "on"},
    },
    "EXP-B07": {
        "dataset_name": "brs",
        "model_family": "ind_n_shared",
        "seed": 42,
        "config_file": "lc2_shared.yaml",
        "model_overrides": {"name": "ind2_shared", "observe": "off"},
    },
    "EXP-B08": {
        "dataset_name": "brs",
        "model_family": "ind_n_shared",
        "seed": 123,
        "config_file": "lc2_shared.yaml",
        "model_overrides": {"name": "ind2_shared", "observe": "off"},
    },
    "EXP-B09": {
        "dataset_name": "brs",
        "model_family": "lcn_shared",
        "seed": 42,
        "config_file": "lc3_shared.yaml",
        "model_overrides": {"observe": "on"},
    },
    "EXP-B10": {
        "dataset_name": "brs",
        "model_family": "lcn_shared",
        "seed": 123,
        "config_file": "lc3_shared.yaml",
        "model_overrides": {"observe": "on"},
    },
    "EXP-B11": {
        "dataset_name": "brs",
        "model_family": "ind_n_shared",
        "seed": 42,
        "config_file": "lc3_shared.yaml",
        "model_overrides": {"name": "ind3_shared", "observe": "off"},
    },
    "EXP-B12": {
        "dataset_name": "brs",
        "model_family": "ind_n_shared",
        "seed": 123,
        "config_file": "lc3_shared.yaml",
        "model_overrides": {"name": "ind3_shared", "observe": "off"},
    },
}


def get_arithmetic_experiment_spec(experiment_id: str, configs_dir: Path) -> ExperimentSpec:
    return _get_experiment_spec(
        experiment_id=experiment_id,
        configs_dir=configs_dir,
        registry=ARITHMETIC_EXPERIMENTS,
    )


def get_brs_experiment_spec(experiment_id: str, configs_dir: Path) -> ExperimentSpec:
    return _get_experiment_spec(
        experiment_id=experiment_id,
        configs_dir=configs_dir,
        registry=BRS_EXPERIMENTS,
    )


def _get_experiment_spec(
    experiment_id: str,
    configs_dir: Path,
    registry: dict[str, dict[str, object]],
) -> ExperimentSpec:
    if experiment_id not in registry:
        raise KeyError(f"未知实验 ID：{experiment_id}")

    spec = registry[experiment_id]
    config_path = Path(configs_dir) / spec["config_file"]
    resolved_config = load_config(config_path)
    resolved_config["data"]["dataset"] = spec["dataset_name"]
    resolved_config["model"] = {
        **resolved_config.get("model", {}),
        **spec["model_overrides"],
    }

    return ExperimentSpec(
        experiment_id=experiment_id,
        dataset_name=spec["dataset_name"],
        model_family=spec["model_family"],
        seed=spec["seed"],
        resolved_config=resolved_config,
    )
