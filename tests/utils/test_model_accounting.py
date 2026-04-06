from latent_consensus.utils.model_accounting import (
    ModelAccountingSpec,
    estimate_model_accounting_entry,
    validate_model_accounting_entry,
)


def test_estimate_model_accounting_entry_returns_positive_values() -> None:
    spec = ModelAccountingSpec(
        model_name="lc2_shared",
        processors=2,
        weight_mode="shared",
        layers=12,
        hidden_size=768,
        vocab_size=50257,
        context_length=192,
    )

    entry = estimate_model_accounting_entry(spec)

    assert entry["params"] > 0
    assert entry["train_flops_per_step"] > 0
    assert entry["infer_flops_per_sample"] > 0


def test_independent_weight_mode_has_more_params_than_shared() -> None:
    shared_spec = ModelAccountingSpec(
        model_name="lc2_shared",
        processors=2,
        weight_mode="shared",
        layers=12,
        hidden_size=768,
        vocab_size=50257,
        context_length=192,
    )
    independent_spec = ModelAccountingSpec(
        model_name="lc2_independent",
        processors=2,
        weight_mode="independent",
        layers=12,
        hidden_size=768,
        vocab_size=50257,
        context_length=192,
    )

    shared_entry = estimate_model_accounting_entry(shared_spec)
    independent_entry = estimate_model_accounting_entry(independent_spec)

    assert independent_entry["params"] > shared_entry["params"]


def test_validate_model_accounting_entry_rejects_missing_field() -> None:
    invalid_entry = {
        "model_name": "lc1",
        "params": 1,
        "train_flops_per_step": 1,
    }

    try:
        validate_model_accounting_entry(invalid_entry)
    except ValueError as error:
        assert "infer_flops_per_sample" in str(error)
    else:
        raise AssertionError("应检测到账本缺失字段")
