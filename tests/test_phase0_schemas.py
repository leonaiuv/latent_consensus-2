from latent_consensus.training.metrics_schema import (
    REQUIRED_METRIC_FIELDS,
    REQUIRED_MODEL_ACCOUNTING_FIELDS,
    REQUIRED_PROFILE_MEMORY_FIELDS,
    REQUIRED_RESULTS_DIRECTORIES,
)


def test_metrics_schema_covers_required_fields() -> None:
    expected_fields = {
        "macro_acc_id",
        "macro_acc_ood",
        "observe_gain_N",
        "observe_off_delta",
        "scramble_delta",
        "synergy_rate",
        "processor_disagreement",
        "leave_one_out_delta",
        "latency_per_sample",
        "accuracy_per_param",
        "accuracy_per_flop",
    }
    assert expected_fields.issubset(REQUIRED_METRIC_FIELDS)


def test_profile_memory_schema_covers_required_fields() -> None:
    expected_fields = {
        "device",
        "step_count",
        "step_time_ms",
        "current_allocated_memory",
        "driver_allocated_memory",
        "recommended_max_memory",
        "oom",
        "fallback_triggered",
    }
    assert expected_fields.issubset(REQUIRED_PROFILE_MEMORY_FIELDS)


def test_model_accounting_schema_covers_required_fields() -> None:
    expected_fields = {
        "model_name",
        "params",
        "train_flops_per_step",
        "infer_flops_per_sample",
    }
    assert expected_fields.issubset(REQUIRED_MODEL_ACCOUNTING_FIELDS)


def test_results_directories_are_declared() -> None:
    expected_directories = {
        "results/gate0",
        "results/arithmetic_debug",
        "results/brs_main",
        "results/interventions",
        "results/promotion",
        "results/accounting",
    }
    assert expected_directories.issubset(REQUIRED_RESULTS_DIRECTORIES)
