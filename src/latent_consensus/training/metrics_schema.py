"""集中维护 Phase 0 所需的结果字段与目录约束。"""

from __future__ import annotations


REQUIRED_METRIC_FIELDS = {
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


REQUIRED_PROFILE_MEMORY_FIELDS = {
    "device",
    "step_count",
    "step_time_ms",
    "current_allocated_memory",
    "driver_allocated_memory",
    "recommended_max_memory",
    "oom",
    "fallback_triggered",
    "runtime_status",
    "error_message",
}


REQUIRED_MODEL_ACCOUNTING_FIELDS = {
    "model_name",
    "params",
    "train_flops_per_step",
    "infer_flops_per_sample",
}


REQUIRED_RESULTS_DIRECTORIES = {
    "results/gate0",
    "results/arithmetic_debug",
    "results/brs_main",
    "results/interventions",
    "results/promotion",
    "results/accounting",
}
