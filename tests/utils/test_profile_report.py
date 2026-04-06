from latent_consensus.utils.profile_report import (
    build_profile_memory_report,
    validate_profile_memory_report,
)


def test_build_profile_memory_report_collects_step_metrics() -> None:
    call_count = 0

    def fake_probe(step_index: int) -> dict[str, float]:
        nonlocal call_count
        call_count += 1
        return {
            "step_time_ms": 10.0 + step_index,
            "current_allocated_memory": 100.0 + step_index,
            "driver_allocated_memory": 200.0 + step_index,
            "recommended_max_memory": 500.0,
        }

    report = build_profile_memory_report(step_count=3, probe=fake_probe)

    assert call_count == 3
    assert report["step_count"] == 3
    assert report["step_time_ms"] == [10.0, 11.0, 12.0]
    assert report["current_allocated_memory"] == [100.0, 101.0, 102.0]


def test_validate_profile_memory_report_rejects_missing_field() -> None:
    invalid_report = {
        "device": "mps",
        "step_count": 2,
        "step_time_ms": [],
        "current_allocated_memory": [],
        "driver_allocated_memory": [],
        "recommended_max_memory": None,
        "oom": False,
    }

    try:
        validate_profile_memory_report(invalid_report)
    except ValueError as error:
        assert "fallback_triggered" in str(error)
    else:
        raise AssertionError("应检测到缺失字段")
