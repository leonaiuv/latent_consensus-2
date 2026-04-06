"""BRS 主线实验 runner。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from latent_consensus.config.registry import get_brs_experiment_spec
from latent_consensus.data.brs import BRSConfig, BRSSample, brs_template_signature, generate_brs_sample
from latent_consensus.models.ind_n import IndNSharedModel
from latent_consensus.models.lc1 import LC1Model
from latent_consensus.models.latent_consensus_causal_lm import LatentConsensusCausalLM
from latent_consensus.models.lcn_shared import LCNSharedModel
from latent_consensus.runtime.gate0_runtime import apply_hf_endpoint
from latent_consensus.training.causal_lm_trainer import CausalLMTrainer
from latent_consensus.training.text_tasks import build_brs_lm_examples, tokenize_lm_examples
from latent_consensus.training.trainer import Trainer


def _build_smoke_model(model_family: str, resolved_config: dict[str, object], seed: int):
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


def _load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _build_real_model(spec, model_name: str):
    model_config = spec.resolved_config["model"]
    k_steps = int(model_config.get("k_steps", model_config.get("k", 5)))
    alpha_init = float(model_config.get("alpha_init", 0.1))
    dropout = float(model_config.get("dropout", 0.1))
    noise_std = float(model_config.get("noise_std", 0.005))
    n_processors = int(model_config.get("n_processors", 1))
    observe = model_config.get("observe", "on") == "on"

    if spec.model_family == "cot":
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(model_name)

    if spec.model_family in {"lc1", "lcn_shared", "ind_n_shared"}:
        return LatentConsensusCausalLM.from_pretrained(
            model_name=model_name,
            n_processors=n_processors,
            k_steps=k_steps,
            observe=observe,
            alpha_init=alpha_init,
            dropout=dropout,
            noise_std=noise_std,
        )

    raise ValueError(f"未知模型家族：{spec.model_family}")


def _resolve_limit(step_limit: int | None, fallback: int | None) -> int | None:
    return step_limit if step_limit is not None else fallback


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


def _load_answer_correctness(prediction_path: Path) -> list[int]:
    return [
        1 if json.loads(line)["answer_correct"] else 0
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
    ]


def _run_smoke_brs_experiment(
    spec,
    output_root: Path,
    train_samples: int,
    val_samples: int,
    test_samples: int,
    seed: int | None,
) -> dict[str, object]:
    effective_seed = spec.seed if seed is None else seed
    model = _build_smoke_model(spec.model_family, spec.resolved_config, seed=effective_seed)

    hidden_size = int(spec.resolved_config["model"].get("hidden_size", 8))
    data_config = spec.resolved_config["data"]
    step_counts = tuple(data_config.get("steps", [2, 4, 6]))

    bundle = _build_smoke_brs_bundle(
        step_counts=step_counts,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        base_seed=effective_seed * 100,
    )

    train_inputs, train_targets = _bundle_to_arrays(bundle["train"], hidden_size=hidden_size)
    val_inputs, val_targets = _bundle_to_arrays(bundle["val"], hidden_size=hidden_size)
    id_inputs, id_targets = _bundle_to_arrays(bundle["test"], hidden_size=hidden_size)
    ood_inputs, ood_targets = _bundle_to_arrays(bundle["ood"], hidden_size=hidden_size)

    experiment_dir = Path(output_root) / spec.experiment_id
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
        "experiment_id": spec.experiment_id,
        "dataset_name": spec.dataset_name,
        "model_family": spec.model_family,
        "seed": effective_seed,
        "runtime_mode": "smoke",
        "history": history,
        "id_accuracy": id_accuracy,
        "ood_accuracy": ood_accuracy,
        "macro_acc_id": id_accuracy,
        "macro_acc_ood": ood_accuracy,
        "id_predictions": (id_predictions == id_targets).astype(int).tolist(),
        "id_targets": [1] * len(id_targets),
        "ood_predictions": (ood_predictions == ood_targets).astype(int).tolist(),
        "ood_targets": [1] * len(ood_targets),
        "id_step_accuracy": _accuracy_per_step(bundle["test"], id_predictions),
        "ood_step_accuracy": _accuracy_per_step(bundle["ood"], ood_predictions),
    }
    (experiment_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _run_real_brs_experiment(
    spec,
    output_root: Path,
    data_dir: Path,
    model_name: str,
    device: str,
    hf_endpoint: str | None,
    step_counts: tuple[int, ...] | None,
    train_limit_per_step: int | None,
    val_limit_per_step: int | None,
    test_limit_per_step: int | None,
    ood_limit_per_step: int | None,
    max_epochs: int | None,
    batch_size: int | None,
    gradient_accumulation_steps: int | None,
    learning_rate: float | None,
    seq_len: int | None,
    seed: int | None,
) -> dict[str, object]:
    import torch

    effective_seed = spec.seed if seed is None else seed
    torch.manual_seed(effective_seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(effective_seed)

    apply_hf_endpoint(hf_endpoint)
    experiment_dir = Path(output_root) / spec.experiment_id
    model = _build_real_model(spec, model_name=model_name)
    tokenizer = _load_tokenizer(model_name)

    resolved_step_counts = step_counts or tuple(spec.resolved_config["data"].get("steps", [2, 4, 6]))
    training_config = spec.resolved_config["training"]
    trainer = CausalLMTrainer(
        model=model,
        tokenizer=tokenizer,
        output_dir=experiment_dir,
        learning_rate=float(learning_rate or training_config.get("learning_rate", 5e-5)),
        max_epochs=int(max_epochs or training_config.get("max_epoch", 1)),
        batch_size=int(batch_size or training_config.get("micro_batch_size", 2)),
        gradient_accumulation_steps=int(
            gradient_accumulation_steps
            or training_config.get("gradient_accumulation_steps", 1)
        ),
        early_stopping_patience=int(training_config.get("early_stopping_patience", 1)),
        device=device,
        grad_clip_norm=float(training_config.get("grad_clip_norm", 1.0)),
    )

    train_examples = build_brs_lm_examples(
        data_dir=data_dir,
        split_name="train",
        step_counts=resolved_step_counts,
        sample_limit_per_step=_resolve_limit(train_limit_per_step, None),
    )
    val_examples = build_brs_lm_examples(
        data_dir=data_dir,
        split_name="val",
        step_counts=resolved_step_counts,
        sample_limit_per_step=_resolve_limit(val_limit_per_step, None),
    )
    test_examples = build_brs_lm_examples(
        data_dir=data_dir,
        split_name="test",
        step_counts=resolved_step_counts,
        sample_limit_per_step=_resolve_limit(test_limit_per_step, None),
    )
    ood_examples = build_brs_lm_examples(
        data_dir=data_dir,
        split_name="ood",
        step_counts=resolved_step_counts,
        sample_limit_per_step=_resolve_limit(ood_limit_per_step, None),
    )

    resolved_seq_len = int(seq_len or training_config.get("seq_len", 192))
    train_dataset = tokenize_lm_examples(train_examples, tokenizer=tokenizer, seq_len=resolved_seq_len)
    val_dataset = tokenize_lm_examples(val_examples, tokenizer=tokenizer, seq_len=resolved_seq_len)
    test_dataset = tokenize_lm_examples(test_examples, tokenizer=tokenizer, seq_len=resolved_seq_len)
    ood_dataset = tokenize_lm_examples(ood_examples, tokenizer=tokenizer, seq_len=resolved_seq_len)

    history = trainer.fit(train_dataset=train_dataset, val_dataset=val_dataset)
    checkpoint = torch.load(experiment_dir / "best_checkpoint.pt", map_location=device)
    trainer.model.load_state_dict(checkpoint["model_state"])
    val_metrics = trainer.evaluate(val_dataset, split_name="val")
    id_metrics = trainer.evaluate(test_dataset, split_name="test")
    ood_metrics = trainer.evaluate(ood_dataset, split_name="ood")

    id_correctness = _load_answer_correctness(experiment_dir / "test_predictions.jsonl")
    ood_correctness = _load_answer_correctness(experiment_dir / "ood_predictions.jsonl")

    summary = {
        "experiment_id": spec.experiment_id,
        "dataset_name": spec.dataset_name,
        "model_family": spec.model_family,
        "seed": effective_seed,
        "runtime_mode": "real",
        "history": history,
        "seq_len": resolved_seq_len,
        "step_counts": list(resolved_step_counts),
        "val_metrics": val_metrics,
        "id_metrics": id_metrics,
        "ood_metrics": ood_metrics,
        "id_accuracy": id_metrics["exact_match"],
        "ood_accuracy": ood_metrics["exact_match"],
        "macro_acc_id": id_metrics["exact_match"],
        "macro_acc_ood": ood_metrics["exact_match"],
        "id_predictions": id_correctness,
        "id_targets": [1] * len(id_correctness),
        "ood_predictions": ood_correctness,
        "ood_targets": [1] * len(ood_correctness),
        "id_step_accuracy": id_metrics["step_accuracy"],
        "ood_step_accuracy": ood_metrics["step_accuracy"],
    }
    (experiment_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def run_brs_experiment(
    experiment_id: str,
    configs_dir: Path,
    output_root: Path,
    train_samples: int = 16,
    val_samples: int = 8,
    test_samples: int = 8,
    seed: int | None = None,
    runtime_mode: str = "smoke",
    data_dir: Path | None = None,
    model_name: str | None = None,
    device: str = "cpu",
    hf_endpoint: str | None = None,
    step_counts: tuple[int, ...] | None = None,
    train_limit_per_step: int | None = None,
    val_limit_per_step: int | None = None,
    test_limit_per_step: int | None = None,
    ood_limit_per_step: int | None = None,
    max_epochs: int | None = None,
    batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
    learning_rate: float | None = None,
    seq_len: int | None = None,
) -> dict[str, object]:
    spec = get_brs_experiment_spec(experiment_id, configs_dir)
    if runtime_mode == "smoke":
        return _run_smoke_brs_experiment(
            spec=spec,
            output_root=output_root,
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            seed=seed,
        )
    if runtime_mode == "real":
        resolved_data_dir = Path(data_dir or "data/processed/brs")
        resolved_model_name = model_name or str(spec.resolved_config["model"].get("base_model", "gpt2"))
        return _run_real_brs_experiment(
            spec=spec,
            output_root=output_root,
            data_dir=resolved_data_dir,
            model_name=resolved_model_name,
            device=device,
            hf_endpoint=hf_endpoint,
            step_counts=step_counts,
            train_limit_per_step=train_limit_per_step,
            val_limit_per_step=val_limit_per_step,
            test_limit_per_step=test_limit_per_step,
            ood_limit_per_step=ood_limit_per_step,
            max_epochs=max_epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            seq_len=seq_len,
            seed=seed,
        )
    raise ValueError("runtime_mode 仅支持 smoke 或 real")
