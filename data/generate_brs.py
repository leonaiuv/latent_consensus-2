"""BRS 数据集 CLI 入口。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from latent_consensus.data.brs import BRSConfig, build_brs_dataset_bundle
from latent_consensus.data.io import export_brs_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-counts", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--train-count", type=int, default=4000)
    parser.add_argument("--val-count", type=int, default=400)
    parser.add_argument("--test-count", type=int, default=400)
    parser.add_argument("--ood-count", type=int, default=400)
    parser.add_argument("--id-entity-count", type=int, default=8)
    parser.add_argument("--ood-entity-count", type=int, default=12)
    parser.add_argument("--id-distractor-count", type=int, default=2)
    parser.add_argument("--ood-distractor-count", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/brs"),
    )
    args = parser.parse_args()

    bundle = build_brs_dataset_bundle(
        step_counts=tuple(args.step_counts),
        split_sizes={
            "train": args.train_count,
            "val": args.val_count,
            "test": args.test_count,
            "ood": args.ood_count,
        },
        id_config=BRSConfig(
            entity_count=args.id_entity_count,
            distractor_count=args.id_distractor_count,
            step_count=args.step_counts[0],
        ),
        ood_config=BRSConfig(
            entity_count=args.ood_entity_count,
            distractor_count=args.ood_distractor_count,
            step_count=args.step_counts[0],
        ),
        base_seed=args.base_seed,
    )
    export_brs_bundle(bundle, args.output_dir)
    print(f"已生成 BRS 数据集：{args.output_dir}")


if __name__ == "__main__":
    main()
