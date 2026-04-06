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


def get_arithmetic_experiment_spec(experiment_id: str, configs_dir: Path) -> ExperimentSpec:
    if experiment_id not in ARITHMETIC_EXPERIMENTS:
        raise KeyError(f"未知实验 ID：{experiment_id}")

    spec = ARITHMETIC_EXPERIMENTS[experiment_id]
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
