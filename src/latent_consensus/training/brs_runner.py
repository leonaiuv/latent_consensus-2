"""BRS 主线最小实验 runner。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from latent_consensus.config.registry import get_brs_experiment_spec
from latent_consensus.data.brs import BRSConfig, BRSSample, brs_template_signature, generate_brs_sample
from latent_consensus.models.ind_n import IndNSharedModel
from latent_consensus.models.lc1 import LC1Model
from latent_consensus.models.lcn_shared import LCNSharedModel
from latent_consensus.training.trainer import Trainer


def _build_model(model_family: str, resolved_config: dict[str, object], seed: int):
    model_config = resolved_config["model"]
    hidden_size = int(model_config.get("hidden_size", 8))
    num_classes = int(model_config.get("num_classes", 2))
    k_steps = int(model_config.get("k_steps", model_config.get("k", 5)))
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


def _sample_to_features(sample: BRSSample, hidden_size: int) -> np.ndarray:
    step_count = len(sample.teacher_steps)
    source_index = ord(sample.source) - ord("A")
    target_index = ord(sample.target) - ord("A")
    attach_points = len({edge[0] for edge in sample.dead_end_branch})
    signature_score = sum(ord(character) for character in brs_template_signature(sample)) % 17
    features = np.array(
        [
            step_count / 6.0,
            len(sample.entities) / 14.0,
            len(sample.facts) / 20.0,
            len(sample.dead_end_branch) / 8.0,
            source_index / 25.0,
            target_index / 25.0,
            attach_points / 4.0,
            signature_score / 17.0,
        ],
        dtype=float,
    )
    if hidden_size <= len(features):
        return features[:hidden_size]
    padding = np.zeros(hidden_size - len(features), dtype=float)
    return np.concatenate([features, padding])


def _sample_to_target(sample: BRSSample) -> int:
    attach_points = len({edge[0] for edge in sample.dead_end_branch})
    parity = (
        len(sample.teacher_steps)
        + len(sample.entities)
        + len(sample.dead_end_branch)
        + attach_points
    ) % 2
    return int(parity)


def _bundle_to_arrays(
    step_mapping: dict[int, list[BRSSample]],
    hidden_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    inputs: list[np.ndarray] = []
    targets: list[int] = []
    for step_count in sorted(step_mapping):
        for sample in step_mapping[step_count]:
            inputs.append(_sample_to_features(sample, hidden_size=hidden_size))
            targets.append(_sample_to_target(sample))
    return np.stack(inputs, axis=0), np.asarray(targets, dtype=int)


def _build_smoke_brs_bundle(
    step_counts: tuple[int, ...],
    train_samples: int,
    val_samples: int,
    test_samples: int,
    base_seed: int,
) -> dict[str, dict[int, list[BRSSample]]]:
    bundle = {"train": {}, "val": {}, "test": {}, "ood": {}}
    current_seed = base_seed

    for step_count in step_counts:
        split_specs = {
            "train": (train_samples, BRSConfig(entity_count=8, distractor_count=2, step_count=step_count)),
            "val": (val_samples, BRSConfig(entity_count=8, distractor_count=2, step_count=step_count)),
            "test": (test_samples, BRSConfig(entity_count=8, distractor_count=2, step_count=step_count)),
            "ood": (test_samples, BRSConfig(entity_count=12, distractor_count=4, step_count=step_count)),
        }
        for split_name, (sample_count, config) in split_specs.items():
            samples: list[BRSSample] = []
            for _ in range(sample_count):
                samples.append(generate_brs_sample(config=config, seed=current_seed))
                current_seed += 1
            bundle[split_name][step_count] = samples

    return bundle


def _accuracy_per_step(
    step_mapping: dict[int, list[BRSSample]],
    predictions: np.ndarray,
) -> dict[str, float]:
    cursor = 0
    results: dict[str, float] = {}
    for step_count in sorted(step_mapping):
        samples = step_mapping[step_count]
        slice_predictions = predictions[cursor : cursor + len(samples)]
        targets = np.asarray([_sample_to_target(sample) for sample in samples], dtype=int)
        results[str(step_count)] = float(np.mean(slice_predictions == targets))
        cursor += len(samples)
    return results


def run_brs_experiment(
    experiment_id: str,
    configs_dir: Path,
    output_root: Path,
    train_samples: int = 16,
    val_samples: int = 8,
    test_samples: int = 8,
    seed: int | None = None,
) -> dict[str, object]:
    spec = get_brs_experiment_spec(experiment_id, configs_dir)
    effective_seed = spec.seed if seed is None else seed
    model = _build_model(spec.model_family, spec.resolved_config, seed=effective_seed)

    hidden_size = int(spec.resolved_config["model"].get("hidden_size", 8))
    data_config = spec.resolved_config["data"]
    step_counts = tuple(data_config.get("steps", [2, 4, 6]))

    bundle = _build_smoke_brs_bundle(
        step_counts=tuple(step_counts),
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        base_seed=effective_seed * 100,
    )

    train_inputs, train_targets = _bundle_to_arrays(bundle["train"], hidden_size=hidden_size)
    val_inputs, val_targets = _bundle_to_arrays(bundle["val"], hidden_size=hidden_size)
    id_inputs, id_targets = _bundle_to_arrays(bundle["test"], hidden_size=hidden_size)
    ood_inputs, ood_targets = _bundle_to_arrays(bundle["ood"], hidden_size=hidden_size)

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

    id_predictions = model.predict(id_inputs)
    ood_predictions = model.predict(ood_inputs)
    id_accuracy = float(np.mean(id_predictions == id_targets))
    ood_accuracy = float(np.mean(ood_predictions == ood_targets))

    summary = {
        "experiment_id": experiment_id,
        "dataset_name": spec.dataset_name,
        "model_family": spec.model_family,
        "seed": effective_seed,
        "history": history,
        "id_accuracy": id_accuracy,
        "ood_accuracy": ood_accuracy,
        "macro_acc_id": id_accuracy,
        "macro_acc_ood": ood_accuracy,
        "id_predictions": id_predictions.tolist(),
        "id_targets": id_targets.tolist(),
        "ood_predictions": ood_predictions.tolist(),
        "ood_targets": ood_targets.tolist(),
        "id_step_accuracy": _accuracy_per_step(bundle["test"], id_predictions),
        "ood_step_accuracy": _accuracy_per_step(bundle["ood"], ood_predictions),
    }
    (experiment_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
