from latent_consensus.data.arithmetic_debug import (
    ArithmeticConfig,
    ArithmeticSample,
    build_arithmetic_dataset_bundle,
)
from latent_consensus.data.brs import (
    BRSConfig,
    BRSSample,
    build_brs_dataset_bundle,
)
from latent_consensus.data.validation import (
    ValidationError,
    validate_arithmetic_bundle,
    validate_brs_bundle,
)


def test_validate_arithmetic_bundle_reports_clean_dataset() -> None:
    bundle = build_arithmetic_dataset_bundle(
        step_counts=(2,),
        split_sizes={"train": 2, "val": 2, "test": 2, "ood": 1},
        id_config=ArithmeticConfig(min_value=1, max_value=99),
        ood_config=ArithmeticConfig(min_value=100, max_value=199),
        base_seed=10,
    )

    report = validate_arithmetic_bundle(bundle)

    assert report["duplicates"] == 0
    assert report["ood_leaks"] == 0


def test_validate_arithmetic_bundle_detects_duplicate_sample() -> None:
    sample = ArithmeticSample(
        step_count=2,
        operands=[1, 2, 3],
        operations=["+", "+"],
        teacher_steps=["[STEP 1] 1 + 2 = 3", "[STEP 2] 1 + 2 + 3 = 6"],
        expression="1 + 2 + 3",
        result=6,
        answer="6",
    )
    bundle = {"train": {2: [sample]}, "val": {2: [sample]}, "test": {2: []}, "ood": {2: []}}

    try:
        validate_arithmetic_bundle(bundle)
    except ValidationError as error:
        assert "重复样本" in str(error)
    else:
        raise AssertionError("应检测到重复样本")


def test_validate_brs_bundle_reports_clean_dataset() -> None:
    bundle = build_brs_dataset_bundle(
        step_counts=(2,),
        split_sizes={"train": 3, "val": 2, "test": 2, "ood": 1},
        id_config=BRSConfig(entity_count=8, distractor_count=2, step_count=2),
        ood_config=BRSConfig(entity_count=12, distractor_count=4, step_count=2),
        base_seed=50,
    )

    report = validate_brs_bundle(bundle)

    assert report["duplicates"] == 0
    assert report["ood_leaks"] == 0


def test_validate_brs_bundle_detects_template_overlap() -> None:
    sample = BRSSample(
        entities=["A", "B", "C", "D"],
        facts=[("A", "B"), ("B", "C"), ("A", "D")],
        source="A",
        target="C",
        query="A ? C",
        teacher_steps=["[STEP 1] A > B", "[STEP 2] A > C"],
        answer="A > C",
        dead_end_branch=[("A", "D")],
    )
    bundle = {"train": {2: [sample]}, "val": {2: [sample]}, "test": {2: []}, "ood": {2: []}}

    try:
        validate_brs_bundle(bundle)
    except ValidationError as error:
        assert "模板泄漏" in str(error)
    else:
        raise AssertionError("应检测到模板泄漏")
