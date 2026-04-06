"""数据 bundle 的文件导出与回读。"""

from __future__ import annotations

import json
from pathlib import Path

from latent_consensus.data.arithmetic_debug import (
    ArithmeticBundle,
    ArithmeticSample,
    arithmetic_sample_signature,
    serialize_arithmetic_sample,
)
from latent_consensus.data.brs import (
    BRSBundle,
    BRSSample,
    brs_template_signature,
    serialize_brs_sample,
)


def _summary_from_bundle(bundle: dict[str, dict[int, list[object]]], dataset_name: str) -> dict[str, object]:
    split_counts: dict[str, dict[str, int]] = {}
    for split_name, step_mapping in bundle.items():
        split_counts[split_name] = {
            str(step_count): len(samples)
            for step_count, samples in step_mapping.items()
        }
    return {"dataset": dataset_name, "split_counts": split_counts}


def export_arithmetic_bundle(bundle: ArithmeticBundle, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary_from_bundle(bundle, dataset_name="arithmetic_debug")

    for split_name, step_mapping in bundle.items():
        for step_count, samples in step_mapping.items():
            output_path = output_dir / f"{split_name}_step{step_count}.jsonl"
            with output_path.open("w", encoding="utf-8") as file:
                for sample_index, sample in enumerate(samples):
                    record = serialize_arithmetic_sample(sample)
                    record.update(
                        {
                            "dataset": "arithmetic_debug",
                            "split": split_name,
                            "sample_id": f"{split_name}-step{step_count}-{sample_index}",
                            "template_signature": arithmetic_sample_signature(sample),
                        }
                    )
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def export_brs_bundle(bundle: BRSBundle, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary_from_bundle(bundle, dataset_name="brs")

    for split_name, step_mapping in bundle.items():
        for step_count, samples in step_mapping.items():
            output_path = output_dir / f"{split_name}_step{step_count}.jsonl"
            with output_path.open("w", encoding="utf-8") as file:
                for sample_index, sample in enumerate(samples):
                    record = serialize_brs_sample(sample)
                    record.update(
                        {
                            "dataset": "brs",
                            "split": split_name,
                            "sample_id": f"{split_name}-step{step_count}-{sample_index}",
                            "step_count": step_count,
                            "template_signature": brs_template_signature(sample),
                        }
                    )
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_arithmetic_bundle(output_dir: Path) -> ArithmeticBundle:
    bundle: ArithmeticBundle = {}
    for file_path in sorted(output_dir.glob("*_step*.jsonl")):
        split_name, step_value = file_path.stem.split("_step")
        step_count = int(step_value)
        bundle.setdefault(split_name, {})
        samples: list[ArithmeticSample] = []
        for line in file_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            samples.append(
                ArithmeticSample(
                    step_count=record["step_count"],
                    operands=record["operands"],
                    operations=record["operations"],
                    teacher_steps=record["teacher_steps"],
                    expression=record["expression"],
                    result=record["result"],
                    answer=record["answer"],
                )
            )
        bundle[split_name][step_count] = samples
    return bundle


def load_brs_bundle(output_dir: Path) -> BRSBundle:
    bundle: BRSBundle = {}
    for file_path in sorted(output_dir.glob("*_step*.jsonl")):
        split_name, step_value = file_path.stem.split("_step")
        step_count = int(step_value)
        bundle.setdefault(split_name, {})
        samples: list[BRSSample] = []
        for line in file_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            samples.append(
                BRSSample(
                    entities=record["entities"],
                    facts=[tuple(edge) for edge in record["facts"]],
                    source=record["source"],
                    target=record["target"],
                    query=record["query"],
                    teacher_steps=record["teacher_steps"],
                    answer=record["answer"],
                    dead_end_branch=[tuple(edge) for edge in record["dead_end_branch"]],
                )
            )
        bundle[split_name][step_count] = samples
    return bundle
