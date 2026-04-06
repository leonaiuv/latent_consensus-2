"""参数量 / FLOP 账本估算与校验。"""

from __future__ import annotations

from dataclasses import dataclass

from latent_consensus.training.metrics_schema import REQUIRED_MODEL_ACCOUNTING_FIELDS


@dataclass(frozen=True)
class ModelAccountingSpec:
    model_name: str
    processors: int
    weight_mode: str
    layers: int
    hidden_size: int
    vocab_size: int
    context_length: int


def estimate_model_accounting_entry(spec: ModelAccountingSpec) -> dict[str, object]:
    if spec.processors <= 0:
        raise ValueError("processors 必须为正整数")
    if spec.weight_mode not in {"shared", "independent"}:
        raise ValueError("weight_mode 仅支持 shared 或 independent")

    embedding_params = spec.vocab_size * spec.hidden_size
    block_params = 12 * (spec.hidden_size**2) * spec.layers
    lm_head_params = spec.hidden_size * spec.vocab_size
    base_params = embedding_params + block_params + lm_head_params

    processor_multiplier = 1 if spec.weight_mode == "shared" else spec.processors
    params = base_params * processor_multiplier
    sequence_factor = spec.context_length * spec.layers * spec.hidden_size
    train_flops_per_step = params * sequence_factor * 2
    infer_flops_per_sample = params * spec.context_length

    return {
        "model_name": spec.model_name,
        "params": params,
        "train_flops_per_step": train_flops_per_step,
        "infer_flops_per_sample": infer_flops_per_sample,
        "weight_mode": spec.weight_mode,
        "processors": spec.processors,
        "estimate_mode": "analytical",
    }


def validate_model_accounting_entry(entry: dict[str, object]) -> None:
    missing_fields = REQUIRED_MODEL_ACCOUNTING_FIELDS.difference(entry)
    if missing_fields:
        raise ValueError(f"账本缺少字段: {sorted(missing_fields)}")

    for field_name in ("params", "train_flops_per_step", "infer_flops_per_sample"):
        value = entry[field_name]
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name} 必须是正整数")
