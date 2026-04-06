from latent_consensus.utils.model_accounting import (
    ModelAccountingSpec,
    build_default_model_accounting_specs,
    estimate_model_accounting_entry,
    estimate_model_accounting_entry_from_base_params,
    resolve_transformer_dimensions,
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


def test_resolve_transformer_dimensions_reads_gpt2_style_config() -> None:
    class FakeConfig:
        n_layer = 12
        n_embd = 768
        vocab_size = 50257
        n_positions = 1024

    dimensions = resolve_transformer_dimensions(FakeConfig())

    assert dimensions == {
        "layers": 12,
        "hidden_size": 768,
        "vocab_size": 50257,
        "context_length": 1024,
    }


def test_build_default_model_accounting_specs_returns_core_variants() -> None:
    specs = build_default_model_accounting_specs(
        layers=12,
        hidden_size=768,
        vocab_size=50257,
        context_length=192,
    )

    assert [spec.model_name for spec in specs] == [
        "lc1",
        "lc2_shared",
        "lc3_shared",
        "optional_independent",
    ]


def test_estimate_model_accounting_entry_from_base_params_respects_exact_value() -> None:
    spec = ModelAccountingSpec(
        model_name="lc1",
        processors=1,
        weight_mode="shared",
        layers=12,
        hidden_size=768,
        vocab_size=50257,
        context_length=192,
    )

    entry = estimate_model_accounting_entry_from_base_params(
        spec=spec,
        base_params=124_439_808,
        estimate_mode="instantiated",
    )

    assert entry["params"] == 124_439_808
    assert entry["estimate_mode"] == "instantiated"


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
