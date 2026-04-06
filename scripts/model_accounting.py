"""Phase 0: 参数量 / FLOP 账本脚本骨架。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from latent_consensus.utils.model_accounting import (
    build_default_model_accounting_specs,
    estimate_model_accounting_entry,
    estimate_model_accounting_entry_from_base_params,
    resolve_transformer_dimensions,
    validate_model_accounting_entry,
)


def _resolve_dimensions(
    model_name: str | None,
    hf_endpoint: str | None,
    layers: int,
    hidden_size: int,
    vocab_size: int,
    context_length: int,
) -> dict[str, int]:
    if model_name is None:
        return {
            "layers": layers,
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "context_length": context_length,
        }

    try:
        if hf_endpoint:
            import os

            os.environ["HF_ENDPOINT"] = hf_endpoint
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name)
        dimensions = resolve_transformer_dimensions(config)
        dimensions["context_length"] = context_length
        return dimensions
    except Exception:  # noqa: BLE001 - 账本脚本需要允许离线回退
        return {
            "layers": layers,
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "context_length": context_length,
        }


def _resolve_exact_base_params(
    model_name: str | None,
    hf_endpoint: str | None,
) -> int | None:
    if model_name is None:
        return None

    try:
        if hf_endpoint:
            import os

            os.environ["HF_ENDPOINT"] = hf_endpoint
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(config)
        return int(sum(parameter.numel() for parameter in model.parameters()))
    except Exception:  # noqa: BLE001 - 账本脚本允许回退到解析估算
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--hf-endpoint", type=str, default=None)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--context-length", type=int, default=192)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/accounting/model_accounting_report.json"),
    )
    args = parser.parse_args()

    dimensions = _resolve_dimensions(
        model_name=args.model_name,
        hf_endpoint=args.hf_endpoint,
        layers=args.layers,
        hidden_size=args.hidden_size,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
    )
    exact_base_params = _resolve_exact_base_params(
        model_name=args.model_name,
        hf_endpoint=args.hf_endpoint,
    )
    specs = build_default_model_accounting_specs(**dimensions)
    if exact_base_params is None:
        entries = [estimate_model_accounting_entry(spec) for spec in specs]
    else:
        entries = [
            estimate_model_accounting_entry_from_base_params(
                spec=spec,
                base_params=exact_base_params,
                estimate_mode="instantiated",
            )
            for spec in specs
        ]
    for entry in entries:
        validate_model_accounting_entry(entry)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"已生成账本报告：{output_path}")


if __name__ == "__main__":
    main()
