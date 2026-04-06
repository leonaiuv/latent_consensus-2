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
    ModelAccountingSpec,
    estimate_model_accounting_entry,
    validate_model_accounting_entry,
)


def main() -> None:
    parser = argparse.ArgumentParser()
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

    specs = [
        ModelAccountingSpec(
            model_name="lc1",
            processors=1,
            weight_mode="shared",
            layers=args.layers,
            hidden_size=args.hidden_size,
            vocab_size=args.vocab_size,
            context_length=args.context_length,
        ),
        ModelAccountingSpec(
            model_name="lc2_shared",
            processors=2,
            weight_mode="shared",
            layers=args.layers,
            hidden_size=args.hidden_size,
            vocab_size=args.vocab_size,
            context_length=args.context_length,
        ),
        ModelAccountingSpec(
            model_name="lc3_shared",
            processors=3,
            weight_mode="shared",
            layers=args.layers,
            hidden_size=args.hidden_size,
            vocab_size=args.vocab_size,
            context_length=args.context_length,
        ),
        ModelAccountingSpec(
            model_name="optional_independent",
            processors=2,
            weight_mode="independent",
            layers=args.layers,
            hidden_size=args.hidden_size,
            vocab_size=args.vocab_size,
            context_length=args.context_length,
        ),
    ]
    entries = [estimate_model_accounting_entry(spec) for spec in specs]
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
