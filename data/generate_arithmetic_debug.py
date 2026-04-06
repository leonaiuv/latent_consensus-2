"""Arithmetic-Debug 数据集 CLI 入口。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from latent_consensus.data.arithmetic_debug import ArithmeticConfig, build_arithmetic_dataset_bundle
from latent_consensus.data.io import export_arithmetic_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-counts", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--train-count", type=int, default=2000)
    parser.add_argument("--val-count", type=int, default=200)
    parser.add_argument("--test-count", type=int, default=200)
    parser.add_argument("--ood-count", type=int, default=200)
    parser.add_argument("--id-min-value", type=int, default=1)
    parser.add_argument("--id-max-value", type=int, default=99)
    parser.add_argument("--ood-min-value", type=int, default=100)
    parser.add_argument("--ood-max-value", type=int, default=199)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/arithmetic_debug"),
    )
    args = parser.parse_args()

    bundle = build_arithmetic_dataset_bundle(
        step_counts=tuple(args.step_counts),
        split_sizes={
            "train": args.train_count,
            "val": args.val_count,
            "test": args.test_count,
            "ood": args.ood_count,
        },
        id_config=ArithmeticConfig(
            min_value=args.id_min_value,
            max_value=args.id_max_value,
        ),
        ood_config=ArithmeticConfig(
            min_value=args.ood_min_value,
            max_value=args.ood_max_value,
        ),
        base_seed=args.base_seed,
    )
    export_arithmetic_bundle(bundle, args.output_dir)
    print(f"已生成 Arithmetic-Debug 数据集：{args.output_dir}")


if __name__ == "__main__":
    main()
