"""编排本地 18-run 核心梯子的最小入口。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from latent_consensus.training.local_core_ladder import run_local_core_ladder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="all")
    parser.add_argument("--train-samples", type=int, default=8)
    parser.add_argument("--val-samples", type=int, default=4)
    parser.add_argument("--test-samples", type=int, default=4)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results"),
    )
    args = parser.parse_args()

    report = run_local_core_ladder(
        configs_dir=ROOT / "configs",
        output_root=args.output_root,
        mode=args.mode,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
    )
    print(f"已完成本地核心梯子：mode={report['mode']}")


if __name__ == "__main__":
    main()
