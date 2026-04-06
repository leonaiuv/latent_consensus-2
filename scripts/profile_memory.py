"""Phase 0: 10-step 真配置压测脚本。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from latent_consensus.runtime.gate0_runtime import (
    DEFAULT_PROFILE_TEXT,
    apply_hf_endpoint,
    build_failure_profile_report,
    build_language_model_batch,
    build_runtime_summary,
    collect_memory_snapshot,
    run_profile_loop,
)
from latent_consensus.utils.profile_report import (
    build_profile_memory_report,
    validate_profile_memory_report,
)

def _run_real_profile(
    model_name: str,
    step_count: int,
    seq_len: int,
    micro_batch_size: int,
    learning_rate: float,
    device: str,
    hf_endpoint: str | None,
    text: str,
) -> dict[str, object]:
    try:
        import datasets
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        return build_failure_profile_report(
            step_count=step_count,
            device=device,
            runtime_status="dependency_missing",
            error_message=str(error),
            fallback_triggered=device == "mps",
        )

    apply_hf_endpoint(hf_endpoint)
    runtime_summary = build_runtime_summary(
        torch_module=torch,
        transformers_module=transformers,
        datasets_module=datasets,
    )
    if device == "mps" and not torch.backends.mps.is_available():
        report = build_failure_profile_report(
            step_count=step_count,
            device=device,
            runtime_status="device_unavailable",
            error_message="MPS 不可用，不能执行 Gate 0 真配置压测。",
            fallback_triggered=True,
        )
        report["runtime_summary"] = runtime_summary
        return report

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        batch = build_language_model_batch(
            tokenizer=tokenizer,
            text=text,
            seq_len=seq_len,
            micro_batch_size=micro_batch_size,
            device=device,
            torch_module=torch,
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        model = model.to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        if device == "mps":
            torch.mps.empty_cache()
            torch.mps.synchronize()

        def step_runner(_: int) -> float:
            optimizer.zero_grad(set_to_none=True)
            start_time = time.perf_counter()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            if device == "mps":
                torch.mps.synchronize()
            return (time.perf_counter() - start_time) * 1000.0

        report = run_profile_loop(
            step_count=step_count,
            device=device,
            step_runner=step_runner,
            memory_reader=lambda: collect_memory_snapshot(torch, device),
        )
        report["runtime_summary"] = runtime_summary
        report["model_name"] = model_name
        report["seq_len"] = seq_len
        report["micro_batch_size"] = micro_batch_size
        report["learning_rate"] = learning_rate
        return report
    except Exception as error:  # noqa: BLE001 - Gate 0 需要完整落盘失败态
        oom = isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
        report = build_failure_profile_report(
            step_count=step_count,
            device=device,
            runtime_status="runtime_error",
            error_message=str(error),
            fallback_triggered=device == "mps",
            oom=oom,
        )
        report["runtime_summary"] = runtime_summary
        report["model_name"] = model_name
        report["seq_len"] = seq_len
        report["micro_batch_size"] = micro_batch_size
        report["learning_rate"] = learning_rate
        return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--step-count", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--hf-endpoint", type=str, default=None)
    parser.add_argument("--text", type=str, default=DEFAULT_PROFILE_TEXT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/gate0/profile_memory_report.json"),
    )
    args = parser.parse_args()

    report = _run_real_profile(
        model_name=args.model_name,
        step_count=args.step_count,
        seq_len=args.seq_len,
        micro_batch_size=args.micro_batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        hf_endpoint=args.hf_endpoint,
        text=args.text,
    )
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
