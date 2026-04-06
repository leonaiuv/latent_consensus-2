from latent_consensus.data.arithmetic_debug import (
    ArithmeticConfig,
    build_arithmetic_dataset_bundle,
    generate_arithmetic_sample,
)


def test_generate_arithmetic_sample_uses_expected_step_format() -> None:
    config = ArithmeticConfig(min_value=1, max_value=99)
    sample = generate_arithmetic_sample(step_count=4, config=config, seed=7)

    assert len(sample.teacher_steps) == 4
    assert all(step.startswith("[STEP ") for step in sample.teacher_steps)
    assert sample.answer == str(sample.result)


def test_generate_arithmetic_sample_respects_result_range() -> None:
    config = ArithmeticConfig(min_value=1, max_value=99, max_result=9999)
    sample = generate_arithmetic_sample(step_count=6, config=config, seed=11)

    assert 0 <= sample.result <= 9999


def test_generate_arithmetic_sample_ood_range_is_separate() -> None:
    id_config = ArithmeticConfig(min_value=1, max_value=99)
    ood_config = ArithmeticConfig(min_value=100, max_value=199)

    id_sample = generate_arithmetic_sample(step_count=2, config=id_config, seed=3)
    ood_sample = generate_arithmetic_sample(step_count=2, config=ood_config, seed=3)

    assert max(id_sample.operands) <= 99
    assert min(ood_sample.operands) >= 100


def test_build_arithmetic_dataset_bundle_counts_by_split_and_step() -> None:
    bundle = build_arithmetic_dataset_bundle(
        step_counts=(2, 4),
        split_sizes={"train": 3, "val": 2, "test": 2, "ood": 1},
        id_config=ArithmeticConfig(min_value=1, max_value=99),
        ood_config=ArithmeticConfig(min_value=100, max_value=199),
        base_seed=100,
    )

    assert set(bundle) == {"train", "val", "test", "ood"}
    assert len(bundle["train"][2]) == 3
    assert len(bundle["val"][4]) == 2
    assert len(bundle["test"][2]) == 2
    assert len(bundle["ood"][4]) == 1
    assert min(bundle["ood"][2][0].operands) >= 100
