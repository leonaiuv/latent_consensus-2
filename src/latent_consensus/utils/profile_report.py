"""profiling 报告构建与校验。"""

from __future__ import annotations

from collections.abc import Callable

from latent_consensus.training.metrics_schema import REQUIRED_PROFILE_MEMORY_FIELDS


def build_profile_memory_report(
    step_count: int,
    probe: Callable[[int], dict[str, float]] | None = None,
    device: str = "mps",
) -> dict[str, object]:
    report = {
        "device": device,
        "step_count": step_count,
        "step_time_ms": [],
        "current_allocated_memory": [],
        "driver_allocated_memory": [],
        "recommended_max_memory": None,
        "oom": False,
        "fallback_triggered": False,
        "runtime_status": "ok",
        "error_message": "",
    }

    if probe is None:
        report["runtime_status"] = "probe_not_provided"
        return report

    for step_index in range(step_count):
        measurement = probe(step_index)
        report["step_time_ms"].append(measurement["step_time_ms"])
        report["current_allocated_memory"].append(
            measurement["current_allocated_memory"]
        )
        report["driver_allocated_memory"].append(
            measurement["driver_allocated_memory"]
        )
        report["recommended_max_memory"] = measurement["recommended_max_memory"]

    return report


def validate_profile_memory_report(report: dict[str, object]) -> None:
    missing_fields = REQUIRED_PROFILE_MEMORY_FIELDS.difference(report)
    if missing_fields:
        raise ValueError(f"profiling 报告缺少字段: {sorted(missing_fields)}")

    step_count = report["step_count"]
    if not isinstance(step_count, int) or step_count <= 0:
        raise ValueError("profiling 报告的 step_count 必须为正整数")

    for field_name in (
        "step_time_ms",
        "current_allocated_memory",
        "driver_allocated_memory",
    ):
        values = report[field_name]
        if not isinstance(values, list):
            raise ValueError(f"{field_name} 必须是列表")
        if values and len(values) != step_count:
            raise ValueError(f"{field_name} 的长度必须等于 step_count")
