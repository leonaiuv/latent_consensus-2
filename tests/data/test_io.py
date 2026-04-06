import json

from latent_consensus.data.arithmetic_debug import (
    ArithmeticConfig,
    build_arithmetic_dataset_bundle,
)
from latent_consensus.data.brs import BRSConfig, build_brs_dataset_bundle
from latent_consensus.data.io import (
    export_arithmetic_bundle,
    export_brs_bundle,
    load_arithmetic_bundle,
    load_brs_bundle,
)
from latent_consensus.data.validation import build_dataset_validation_report


def test_export_and_load_arithmetic_bundle_roundtrip(tmp_path) -> None:
    bundle = build_arithmetic_dataset_bundle(
        step_counts=(2, 4),
        split_sizes={"train": 2, "val": 1, "test": 1, "ood": 1},
        id_config=ArithmeticConfig(min_value=1, max_value=99),
        ood_config=ArithmeticConfig(min_value=100, max_value=199),
        base_seed=5,
    )

    output_dir = tmp_path / "arithmetic_debug"
    export_arithmetic_bundle(bundle, output_dir)
    loaded_bundle = load_arithmetic_bundle(output_dir)

    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "train_step2.jsonl").is_file()
    assert len(loaded_bundle["train"][2]) == 2
    assert len(loaded_bundle["ood"][4]) == 1


def test_export_and_load_brs_bundle_roundtrip(tmp_path) -> None:
    bundle = build_brs_dataset_bundle(
        step_counts=(2,),
        split_sizes={"train": 2, "val": 1, "test": 1, "ood": 1},
        id_config=BRSConfig(entity_count=8, distractor_count=2, step_count=2),
        ood_config=BRSConfig(entity_count=12, distractor_count=4, step_count=2),
        base_seed=40,
    )

    output_dir = tmp_path / "brs"
    export_brs_bundle(bundle, output_dir)
    loaded_bundle = load_brs_bundle(output_dir)

    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "train_step2.jsonl").is_file()
    assert len(loaded_bundle["train"][2]) == 2
    assert len(loaded_bundle["ood"][2]) == 1


def test_build_dataset_validation_report_from_exported_bundles(tmp_path) -> None:
    arithmetic_bundle = build_arithmetic_dataset_bundle(
        step_counts=(2,),
        split_sizes={"train": 2, "val": 1, "test": 1, "ood": 1},
        id_config=ArithmeticConfig(min_value=1, max_value=99),
        ood_config=ArithmeticConfig(min_value=100, max_value=199),
        base_seed=10,
    )
    brs_bundle = build_brs_dataset_bundle(
        step_counts=(2,),
        split_sizes={"train": 2, "val": 1, "test": 1, "ood": 1},
        id_config=BRSConfig(entity_count=8, distractor_count=2, step_count=2),
        ood_config=BRSConfig(entity_count=12, distractor_count=4, step_count=2),
        base_seed=60,
    )

    arithmetic_dir = tmp_path / "arithmetic_debug"
    brs_dir = tmp_path / "brs"
    export_arithmetic_bundle(arithmetic_bundle, arithmetic_dir)
    export_brs_bundle(brs_bundle, brs_dir)

    report = build_dataset_validation_report(
        arithmetic_dir=arithmetic_dir,
        brs_dir=brs_dir,
    )

    assert report["arithmetic_debug"]["duplicates"] == 0
    assert report["brs"]["duplicates"] == 0
    assert report["arithmetic_debug"]["summary_path"].endswith("summary.json")
    assert report["brs"]["summary_path"].endswith("summary.json")

    report_path = tmp_path / "dataset_validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    assert report_path.is_file()
