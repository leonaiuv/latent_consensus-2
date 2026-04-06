"""Arithmetic-Debug 最小实验 runner。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from latent_consensus.config.registry import get_arithmetic_experiment_spec
from latent_consensus.models.ind_n import IndNSharedModel
from latent_consensus.models.lc1 import LC1Model
from latent_consensus.models.lcn_shared import LCNSharedModel
from latent_consensus.training.trainer import Trainer


def _build_smoke_dataset(
    sample_count: int,
    hidden_size: int,
    num_classes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    inputs = rng.normal(0.0, 1.0, size=(sample_count, hidden_size))
    teacher = rng.normal(0.0, 1.0, size=(hidden_size, num_classes))
    targets = np.argmax(inputs @ teacher, axis=1)
    return inputs, targets


def _build_model(model_family: str, resolved_config: dict[str, object], seed: int):
    model_config = resolved_config["model"]
    hidden_size = int(model_config.get("hidden_size", 4))
    num_classes = int(model_config.get("num_classes", 2))
    k_steps = int(model_config.get("k_steps", resolved_config["model"].get("k", 5)))
    mutation_scale = float(model_config.get("mutation_scale", 0.1))
    n_processors = int(model_config.get("n_processors", 1))

    if model_family == "cot":
        return LC1Model(
            hidden_size=hidden_size,
            num_classes=num_classes,
            k_steps=1,
            mutation_scale=0.0,
            seed=seed,
        )
    if model_family == "lc1":
        return LC1Model(
            hidden_size=hidden_size,
            num_classes=num_classes,
            k_steps=k_steps,
            mutation_scale=mutation_scale,
            seed=seed,
        )
    if model_family == "lcn_shared":
        return LCNSharedModel(
            hidden_size=hidden_size,
            num_classes=num_classes,
            n_processors=n_processors,
            k_steps=k_steps,
            observe=True,
            mutation_scale=mutation_scale,
            seed=seed,
        )
    if model_family == "ind_n_shared":
        return IndNSharedModel(
            hidden_size=hidden_size,
            num_classes=num_classes,
            n_processors=n_processors,
            k_steps=k_steps,
            mutation_scale=mutation_scale,
            seed=seed,
        )
    raise ValueError(f"未知模型家族：{model_family}")


def run_arithmetic_experiment(
    experiment_id: str,
    configs_dir: Path,
    output_root: Path,
    train_samples: int = 16,
    val_samples: int = 8,
    seed: int | None = None,
) -> dict[str, object]:
    spec = get_arithmetic_experiment_spec(experiment_id, configs_dir)
    effective_seed = spec.seed if seed is None else seed
    model = _build_model(spec.model_family, spec.resolved_config, seed=effective_seed)

    hidden_size = int(spec.resolved_config["model"].get("hidden_size", 4))
    num_classes = int(spec.resolved_config["model"].get("num_classes", 2))
    train_inputs, train_targets = _build_smoke_dataset(
        sample_count=train_samples,
        hidden_size=hidden_size,
        num_classes=num_classes,
        seed=effective_seed,
    )
    val_inputs, val_targets = _build_smoke_dataset(
        sample_count=val_samples,
        hidden_size=hidden_size,
        num_classes=num_classes,
        seed=effective_seed + 1,
    )

    experiment_dir = Path(output_root) / experiment_id
    trainer = Trainer(
        model=model,
        output_dir=experiment_dir,
        learning_rate=float(spec.resolved_config["training"].get("learning_rate", 0.1)),
        max_epochs=int(spec.resolved_config["training"].get("smoke_epochs", 2)),
    )
    history = trainer.fit(
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
    )

    summary = {
        "experiment_id": experiment_id,
        "dataset_name": spec.dataset_name,
        "model_family": spec.model_family,
        "seed": effective_seed,
        "history": history,
    }
    (experiment_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
