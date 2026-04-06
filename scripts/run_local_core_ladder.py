"""编排本地 18-run 核心梯子的入口。"""

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
    parser.add_argument("--runtime-mode", type=str, default="smoke")
    parser.add_argument("--train-samples", type=int, default=8)
    parser.add_argument("--val-samples", type=int, default=4)
    parser.add_argument("--test-samples", type=int, default=4)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/arithmetic_debug")
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--hf-endpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--train-limit-per-step", type=int, default=None)
    parser.add_argument("--val-limit-per-step", type=int, default=None)
    parser.add_argument("--test-limit-per-step", type=int, default=None)
    parser.add_argument("--ood-limit-per-step", type=int, default=None)
    parser.add_argument("--step-counts", type=int, nargs="*", default=None)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--gate1-report-path", type=Path, default=None)
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
        runtime_mode=args.runtime_mode,
        data_dir=args.data_dir,
        model_name=args.model_name,
        device=args.device,
        hf_endpoint=args.hf_endpoint,
        step_counts=tuple(args.step_counts) if args.step_counts else None,
        train_limit_per_step=args.train_limit_per_step,
        val_limit_per_step=args.val_limit_per_step,
        test_limit_per_step=args.test_limit_per_step,
        ood_limit_per_step=args.ood_limit_per_step,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        seq_len=args.seq_len,
        resume=args.resume,
        gate1_report_path=args.gate1_report_path,
    )
    print(f"已完成本地核心梯子：mode={report['mode']}")


if __name__ == "__main__":
    main()
