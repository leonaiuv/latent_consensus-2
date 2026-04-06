"""Gate 0 真实运行时探针与 profiling 组件。"""

from __future__ import annotations

import os
import platform
from collections.abc import Callable
from typing import Any

from latent_consensus.utils.profile_report import build_profile_memory_report

DEFAULT_PROFILE_TEXT = (
    "Latent consensus profiling sample. "
    "We repeat this sentence to build a stable GPT-2 language modeling batch. "
    "The goal is to exercise the real forward and backward path on MPS."
)


def apply_hf_endpoint(hf_endpoint: str | None) -> str | None:
    """统一管理 HF_ENDPOINT，避免脚本和测试各自散落设置逻辑。"""

    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
        return hf_endpoint
    return os.environ.get("HF_ENDPOINT")


def build_runtime_summary(
    torch_module: Any,
    transformers_module: Any,
    datasets_module: Any | None = None,
) -> dict[str, object]:
    """汇总 Gate 0 运行时状态，供报告落盘与故障归因使用。"""

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": getattr(torch_module, "__version__", "unknown"),
        "transformers_version": getattr(
            transformers_module,
            "__version__",
            "unknown",
        ),
        "datasets_version": getattr(datasets_module, "__version__", None),
        "mps_built": bool(torch_module.backends.mps.is_built()),
        "mps_available": bool(torch_module.backends.mps.is_available()),
    }


def ensure_tokenizer_pad_token(tokenizer: Any) -> int | None:
    """GPT-2 默认无 pad token，这里显式回落到 eos token。"""

    if getattr(tokenizer, "pad_token", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is None:
            raise ValueError("tokenizer 缺少 pad_token 且 eos_token 不可用")
        tokenizer.pad_token = eos_token
    return getattr(tokenizer, "pad_token_id", None)


def build_language_model_batch(
    tokenizer: Any,
    text: str,
    seq_len: int,
    micro_batch_size: int,
    device: str,
    torch_module: Any,
) -> dict[str, Any]:
    """构建固定长度的自回归语言模型 batch。"""

    if seq_len <= 0:
        raise ValueError("seq_len 必须为正整数")
    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size 必须为正整数")

    ensure_tokenizer_pad_token(tokenizer)
    encoded = tokenizer(
        [text] * micro_batch_size,
        truncation=True,
        padding="max_length",
        max_length=seq_len,
    )
    input_ids = torch_module.tensor(
        encoded["input_ids"],
        dtype=torch_module.long,
        device=device,
    )
    attention_mask = torch_module.tensor(
        encoded["attention_mask"],
        dtype=torch_module.long,
        device=device,
    )
    labels = input_ids.clone()
    labels = labels.masked_fill(attention_mask == 0, -100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def collect_memory_snapshot(torch_module: Any, device: str) -> dict[str, float | None]:
    """读取当前设备的显存统计；非 MPS 路径返回零值占位。"""

    if device != "mps" or not torch_module.backends.mps.is_available():
        return {
            "current_allocated_memory": 0.0,
            "driver_allocated_memory": 0.0,
            "recommended_max_memory": None,
        }

    return {
        "current_allocated_memory": float(torch_module.mps.current_allocated_memory()),
        "driver_allocated_memory": float(torch_module.mps.driver_allocated_memory()),
        "recommended_max_memory": float(torch_module.mps.recommended_max_memory()),
    }


def run_profile_loop(
    step_count: int,
    device: str,
    step_runner: Callable[[int], float],
    memory_reader: Callable[[], dict[str, float | None]],
    runtime_status: str = "ok",
) -> dict[str, object]:
    """执行 step 循环并产出统一 profiling 报告。"""

    def probe(step_index: int) -> dict[str, float | None]:
        measurement = memory_reader()
        return {
            "step_time_ms": float(step_runner(step_index)),
            "current_allocated_memory": measurement["current_allocated_memory"],
            "driver_allocated_memory": measurement["driver_allocated_memory"],
            "recommended_max_memory": measurement["recommended_max_memory"],
        }

    report = build_profile_memory_report(
        step_count=step_count,
        probe=probe,
        device=device,
    )
    report["runtime_status"] = runtime_status
    report["error_message"] = ""
    return report


def build_failure_profile_report(
    step_count: int,
    device: str,
    runtime_status: str,
    error_message: str,
    fallback_triggered: bool,
    oom: bool = False,
) -> dict[str, object]:
    """构建失败态 profiling 报告，保证 Gate 0 失败时也有结构化证据。"""

    report = build_profile_memory_report(step_count=step_count, probe=None, device=device)
    report["oom"] = oom
    report["fallback_triggered"] = fallback_triggered
    report["runtime_status"] = runtime_status
    report["error_message"] = error_message
    return report
