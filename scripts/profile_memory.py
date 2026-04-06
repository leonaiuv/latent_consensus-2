"""Phase 0: 10-step 真配置压测脚本骨架。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from latent_consensus.utils.profile_report import (
    build_profile_memory_report,
    validate_profile_memory_report,
)


def _build_placeholder_report(step_count: int) -> dict[str, object]:
    report = build_profile_memory_report(step_count=step_count, probe=None)
    report["runtime_status"] = "integration_pending"
    report["error_message"] = "当前尚未接入真实训练 step；需要在 trainer 可用后接入真实 probe。"
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-count", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/gate0/profile_memory_report.json"),
    )
    args = parser.parse_args()

    report = _build_placeholder_report(step_count=args.step_count)
    validate_profile_memory_report(report)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"已生成 profiling 报告：{output_path}")


if __name__ == "__main__":
    main()
