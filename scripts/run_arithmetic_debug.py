"""Arithmetic-Debug 执行入口占位。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from latent_consensus.training.arithmetic_runner import run_arithmetic_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=str, default="EXP-A02")
    parser.add_argument("--train-samples", type=int, default=16)
    parser.add_argument("--val-samples", type=int, default=8)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/arithmetic_debug"),
    )
    args = parser.parse_args()

    summary = run_arithmetic_experiment(
        experiment_id=args.experiment_id,
        configs_dir=ROOT / "configs",
        output_root=args.output_root,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
    )
    print(f"已完成 Arithmetic-Debug smoke：{summary['experiment_id']}")


if __name__ == "__main__":
    main()
