from latent_consensus.data.brs import (
    BRSConfig,
    build_brs_dataset_bundle,
    brs_template_signature,
    count_paths,
    generate_brs_sample,
)
from latent_consensus.data.validation import validate_brs_bundle


def test_generate_brs_sample_has_unique_solution_path() -> None:
    config = BRSConfig(entity_count=6, distractor_count=2, step_count=3)
    sample = generate_brs_sample(config=config, seed=17)

    assert count_paths(sample.facts, sample.source, sample.target) == 1
    assert len(sample.teacher_steps) == 3
    assert sample.answer == f"{sample.source} > {sample.target}"


def test_generate_brs_sample_includes_dead_end_branch() -> None:
    config = BRSConfig(entity_count=7, distractor_count=2, step_count=3)
    sample = generate_brs_sample(config=config, seed=23)

    assert sample.dead_end_branch, "必须至少存在一条死路分支"
    branch_start, branch_end = sample.dead_end_branch[0]
    assert branch_start == sample.source
    assert count_paths(sample.facts, branch_end, sample.target) == 0


def test_generate_brs_ood_complexity_is_higher_than_id() -> None:
    id_config = BRSConfig(entity_count=8, distractor_count=2, step_count=4)
    ood_config = BRSConfig(entity_count=12, distractor_count=4, step_count=4)

    id_sample = generate_brs_sample(config=id_config, seed=29)
    ood_sample = generate_brs_sample(config=ood_config, seed=29)

    assert len(ood_sample.entities) > len(id_sample.entities)
    assert len(ood_sample.dead_end_branch) >= len(id_sample.dead_end_branch)


def test_generate_brs_teacher_steps_use_expected_format() -> None:
    config = BRSConfig(entity_count=5, distractor_count=1, step_count=2)
    sample = generate_brs_sample(config=config, seed=31)

    assert all(step.startswith("[STEP ") for step in sample.teacher_steps)


def test_build_brs_dataset_bundle_counts_and_ood_complexity() -> None:
    bundle = build_brs_dataset_bundle(
        step_counts=(2, 4),
        split_sizes={"train": 3, "val": 2, "test": 2, "ood": 1},
        id_config=BRSConfig(entity_count=8, distractor_count=2, step_count=2),
        ood_config=BRSConfig(entity_count=12, distractor_count=4, step_count=2),
        base_seed=200,
    )

    assert len(bundle["train"][2]) == 3
    assert len(bundle["val"][4]) == 2
    assert len(bundle["ood"][2]) == 1
    assert len(bundle["ood"][2][0].entities) > len(bundle["train"][2][0].entities)


def test_brs_template_signatures_do_not_overlap_across_splits() -> None:
    bundle = build_brs_dataset_bundle(
        step_counts=(2,),
        split_sizes={"train": 4, "val": 3, "test": 3, "ood": 2},
        id_config=BRSConfig(entity_count=8, distractor_count=2, step_count=2),
        ood_config=BRSConfig(entity_count=12, distractor_count=4, step_count=2),
        base_seed=300,
    )

    train_signatures = {
        brs_template_signature(sample)
        for sample in bundle["train"][2]
    }
    val_signatures = {
        brs_template_signature(sample)
        for sample in bundle["val"][2]
    }
    test_signatures = {
        brs_template_signature(sample)
        for sample in bundle["test"][2]
    }
    ood_signatures = {
        brs_template_signature(sample)
        for sample in bundle["ood"][2]
    }

    assert train_signatures.isdisjoint(val_signatures)
    assert train_signatures.isdisjoint(test_signatures)
    assert train_signatures.isdisjoint(ood_signatures)
    assert val_signatures.isdisjoint(test_signatures)


def test_build_brs_dataset_bundle_supports_larger_phase2_like_split_sizes() -> None:
    bundle = build_brs_dataset_bundle(
        step_counts=(2, 4, 6),
        split_sizes={"train": 60, "val": 12, "test": 12, "ood": 12},
        id_config=BRSConfig(entity_count=12, distractor_count=3, step_count=2),
        ood_config=BRSConfig(entity_count=16, distractor_count=5, step_count=2),
        base_seed=500,
    )

    report = validate_brs_bundle(bundle)

    assert len(bundle["train"][6]) == 60
    assert len(bundle["ood"][6]) == 12
    assert report["duplicates"] == 0
    assert report["template_overlaps"] == 0
