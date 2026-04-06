"""Phase 0: 数据去重与 OOD 校验脚本骨架。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from latent_consensus.data.validation import (
    ValidationError,
    build_dataset_validation_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arithmetic-dir",
        type=Path,
        default=Path("data/processed/arithmetic_debug"),
    )
    parser.add_argument(
        "--brs-dir",
        type=Path,
        default=Path("data/processed/brs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/gate0/dataset_validation_report.json"),
    )
    args = parser.parse_args()

    try:
        report = build_dataset_validation_report(
            arithmetic_dir=args.arithmetic_dir,
            brs_dir=args.brs_dir,
        )
    except ValidationError as error:
        raise SystemExit(str(error)) from error

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"已生成数据校验报告：{output_path}")


if __name__ == "__main__":
    main()
