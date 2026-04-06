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


def resolve_transformer_dimensions(config: object) -> dict[str, int]:
    """从 transformer config 提取账本所需的核心维度。"""

    layers = getattr(config, "n_layer", None)
    hidden_size = getattr(config, "n_embd", None)
    vocab_size = getattr(config, "vocab_size", None)
    context_length = getattr(config, "n_positions", None)
    if context_length is None:
        context_length = getattr(config, "n_ctx", None)

    values = {
        "layers": layers,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "context_length": context_length,
    }
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise ValueError(f"config 缺少账本维度: {missing}")

    return {name: int(value) for name, value in values.items()}


def build_default_model_accounting_specs(
    layers: int,
    hidden_size: int,
    vocab_size: int,
    context_length: int,
) -> list[ModelAccountingSpec]:
    """统一维护本地主线账本条目，避免脚本内散落重复定义。"""

    return [
        ModelAccountingSpec(
            model_name="lc1",
            processors=1,
            weight_mode="shared",
            layers=layers,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            context_length=context_length,
        ),
        ModelAccountingSpec(
            model_name="lc2_shared",
            processors=2,
            weight_mode="shared",
            layers=layers,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            context_length=context_length,
        ),
        ModelAccountingSpec(
            model_name="lc3_shared",
            processors=3,
            weight_mode="shared",
            layers=layers,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            context_length=context_length,
        ),
        ModelAccountingSpec(
            model_name="optional_independent",
            processors=2,
            weight_mode="independent",
            layers=layers,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            context_length=context_length,
        ),
    ]


def estimate_model_accounting_entry_from_base_params(
    spec: ModelAccountingSpec,
    base_params: int,
    estimate_mode: str,
) -> dict[str, object]:
    """基于给定基础模型参数量构建账本条目。"""

    if spec.processors <= 0:
        raise ValueError("processors 必须为正整数")
    if spec.weight_mode not in {"shared", "independent"}:
        raise ValueError("weight_mode 仅支持 shared 或 independent")
    if base_params <= 0:
        raise ValueError("base_params 必须是正整数")

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
        "estimate_mode": estimate_mode,
    }


def estimate_model_accounting_entry(spec: ModelAccountingSpec) -> dict[str, object]:
    embedding_params = spec.vocab_size * spec.hidden_size
    block_params = 12 * (spec.hidden_size**2) * spec.layers
    lm_head_params = spec.hidden_size * spec.vocab_size
    base_params = embedding_params + block_params + lm_head_params
    return estimate_model_accounting_entry_from_base_params(
        spec=spec,
        base_params=base_params,
        estimate_mode="analytical",
    )


def validate_model_accounting_entry(entry: dict[str, object]) -> None:
    missing_fields = REQUIRED_MODEL_ACCOUNTING_FIELDS.difference(entry)
    if missing_fields:
        raise ValueError(f"账本缺少字段: {sorted(missing_fields)}")

    for field_name in ("params", "train_flops_per_step", "infer_flops_per_sample"):
        value = entry[field_name]
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name} 必须是正整数")
