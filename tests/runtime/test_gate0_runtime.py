import types

import torch

from latent_consensus.runtime.gate0_runtime import (
    apply_hf_endpoint,
    build_failure_profile_report,
    build_language_model_batch,
    build_runtime_summary,
    collect_memory_snapshot,
    ensure_tokenizer_pad_token,
    run_profile_loop,
)


class FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token = None
        self.pad_token_id = None
        self.eos_token = "<eos>"
        self.eos_token_id = 99

    def __call__(
        self,
        texts: list[str],
        truncation: bool,
        padding: str,
        max_length: int,
    ) -> dict[str, list[list[int]]]:
        assert truncation is True
        assert padding == "max_length"
        assert len(texts) == 2
        assert max_length == 4
        return {
            "input_ids": [[1, 2, 3, 0], [4, 5, 0, 0]],
            "attention_mask": [[1, 1, 1, 0], [1, 1, 0, 0]],
        }


def test_apply_hf_endpoint_sets_environment(monkeypatch) -> None:
    monkeypatch.delenv("HF_ENDPOINT", raising=False)

    resolved = apply_hf_endpoint("https://hf-mirror.com")

    assert resolved == "https://hf-mirror.com"
    assert apply_hf_endpoint(None) == "https://hf-mirror.com"


def test_build_runtime_summary_collects_versions_and_mps_state() -> None:
    fake_torch = types.SimpleNamespace(
        __version__="2.11.0",
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(
                is_built=lambda: True,
                is_available=lambda: True,
            )
        ),
    )
    fake_transformers = types.SimpleNamespace(__version__="5.5.0")
    fake_datasets = types.SimpleNamespace(__version__="4.8.4")

    summary = build_runtime_summary(
        torch_module=fake_torch,
        transformers_module=fake_transformers,
        datasets_module=fake_datasets,
    )

    assert summary["torch_version"] == "2.11.0"
    assert summary["transformers_version"] == "5.5.0"
    assert summary["datasets_version"] == "4.8.4"
    assert summary["mps_built"] is True
    assert summary["mps_available"] is True


def test_ensure_tokenizer_pad_token_falls_back_to_eos() -> None:
    tokenizer = FakeTokenizer()

    pad_token_id = ensure_tokenizer_pad_token(tokenizer)

    assert tokenizer.pad_token == "<eos>"
    assert pad_token_id is None


def test_build_language_model_batch_masks_padding_tokens() -> None:
    tokenizer = FakeTokenizer()
    batch = build_language_model_batch(
        tokenizer=tokenizer,
        text="ignored",
        seq_len=4,
        micro_batch_size=2,
        device="cpu",
        torch_module=torch,
    )

    assert batch["input_ids"].shape == (2, 4)
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0], [1, 1, 0, 0]]
    assert batch["labels"].tolist() == [[1, 2, 3, -100], [4, 5, -100, -100]]


def test_build_language_model_batch_rejects_invalid_seq_len() -> None:
    tokenizer = FakeTokenizer()

    try:
        build_language_model_batch(
            tokenizer=tokenizer,
            text="ignored",
            seq_len=0,
            micro_batch_size=2,
            device="cpu",
            torch_module=torch,
        )
    except ValueError as error:
        assert "seq_len" in str(error)
    else:
        raise AssertionError("应拒绝非法 seq_len")


def test_collect_memory_snapshot_returns_zero_for_non_mps_device() -> None:
    fake_torch = types.SimpleNamespace(
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: True)
        )
    )

    snapshot = collect_memory_snapshot(fake_torch, device="cpu")

    assert snapshot["current_allocated_memory"] == 0.0
    assert snapshot["driver_allocated_memory"] == 0.0
    assert snapshot["recommended_max_memory"] is None


def test_collect_memory_snapshot_reads_mps_counters() -> None:
    fake_mps = types.SimpleNamespace(
        is_available=lambda: True,
        current_allocated_memory=lambda: 128.0,
        driver_allocated_memory=lambda: 256.0,
        recommended_max_memory=lambda: 512.0,
    )
    fake_torch = types.SimpleNamespace(
        backends=types.SimpleNamespace(mps=fake_mps),
        mps=fake_mps,
    )

    snapshot = collect_memory_snapshot(fake_torch, device="mps")

    assert snapshot["current_allocated_memory"] == 128.0
    assert snapshot["driver_allocated_memory"] == 256.0
    assert snapshot["recommended_max_memory"] == 512.0


def test_run_profile_loop_collects_measurements_and_status() -> None:
    call_order: list[int] = []

    def step_runner(step_index: int) -> float:
        call_order.append(step_index)
        return 1.5 + step_index

    def memory_reader() -> dict[str, float | None]:
        return {
            "current_allocated_memory": 100.0,
            "driver_allocated_memory": 150.0,
            "recommended_max_memory": 200.0,
        }

    report = run_profile_loop(
        step_count=3,
        device="mps",
        step_runner=step_runner,
        memory_reader=memory_reader,
    )

    assert call_order == [0, 1, 2]
    assert report["runtime_status"] == "ok"
    assert report["error_message"] == ""
    assert report["step_time_ms"] == [1.5, 2.5, 3.5]


def test_build_failure_profile_report_marks_failure_details() -> None:
    report = build_failure_profile_report(
        step_count=10,
        device="mps",
        runtime_status="device_unavailable",
        error_message="MPS 不可用",
        fallback_triggered=True,
        oom=False,
    )

    assert report["step_count"] == 10
    assert report["runtime_status"] == "device_unavailable"
    assert report["error_message"] == "MPS 不可用"
    assert report["fallback_triggered"] is True
