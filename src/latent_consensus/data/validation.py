"""Phase 0 数据去重与 OOD 校验辅助。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from latent_consensus.data.arithmetic_debug import (
    ArithmeticBundle,
    ArithmeticSample,
    arithmetic_sample_signature,
)
from latent_consensus.data.brs import BRSBundle, BRSSample, brs_template_signature
from latent_consensus.data.io import load_arithmetic_bundle, load_brs_bundle


class ValidationError(ValueError):
    """数据校验失败。"""


@dataclass(frozen=True)
class BundleReport:
    duplicates: int
    ood_leaks: int
    template_overlaps: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "duplicates": self.duplicates,
            "ood_leaks": self.ood_leaks,
            "template_overlaps": self.template_overlaps,
        }


def _iter_samples(bundle: dict[str, dict[int, list[object]]]) -> list[tuple[str, int, object]]:
    rows: list[tuple[str, int, object]] = []
    for split_name, step_mapping in bundle.items():
        for step_count, samples in step_mapping.items():
            for sample in samples:
                rows.append((split_name, step_count, sample))
    return rows


def validate_arithmetic_bundle(bundle: ArithmeticBundle) -> dict[str, int]:
    duplicates = 0
    seen_signatures: set[str] = set()
    id_max_operand = 0

    for split_name, _step_count, sample in _iter_samples(bundle):
        assert isinstance(sample, ArithmeticSample)
        signature = arithmetic_sample_signature(sample)
        if signature in seen_signatures:
            duplicates += 1
        seen_signatures.add(signature)

        if split_name != "ood":
            id_max_operand = max(id_max_operand, max(sample.operands))

    if duplicates:
        raise ValidationError(f"发现重复样本：{duplicates}")

    ood_leaks = 0
    for sample in bundle.get("ood", {}).values():
        for item in sample:
            if min(item.operands) <= id_max_operand:
                ood_leaks += 1

    if ood_leaks:
        raise ValidationError(f"发现 Arithmetic OOD 泄漏：{ood_leaks}")

    return BundleReport(duplicates=0, ood_leaks=0).to_dict()


def validate_brs_bundle(bundle: BRSBundle) -> dict[str, int]:
    duplicates = 0
    exact_signatures_by_split: dict[str, set[str]] = {}
    split_templates: dict[str, set[str]] = {}

    for split_name, _step_count, sample in _iter_samples(bundle):
        assert isinstance(sample, BRSSample)
        exact_signature = f"{tuple(sorted(sample.facts))}|{sample.query}|{sample.answer}"
        split_exact_signatures = exact_signatures_by_split.setdefault(split_name, set())
        if exact_signature in split_exact_signatures:
            duplicates += 1
        split_exact_signatures.add(exact_signature)
        split_templates.setdefault(split_name, set()).add(brs_template_signature(sample))

    split_names = list(split_templates)
    template_overlaps = 0
    for index, split_name in enumerate(split_names):
        for other_split in split_names[index + 1 :]:
            overlap = split_templates[split_name] & split_templates[other_split]
            template_overlaps += len(overlap)

    if template_overlaps:
        raise ValidationError(f"发现 BRS 模板泄漏：{template_overlaps}")
    if duplicates:
        raise ValidationError(f"发现重复样本：{duplicates}")

    return BundleReport(duplicates=0, ood_leaks=0, template_overlaps=0).to_dict()


def build_dataset_validation_report(
    arithmetic_dir: Path,
    brs_dir: Path,
) -> dict[str, dict[str, object]]:
    arithmetic_bundle = load_arithmetic_bundle(arithmetic_dir)
    brs_bundle = load_brs_bundle(brs_dir)

    arithmetic_report = validate_arithmetic_bundle(arithmetic_bundle)
    arithmetic_report["summary_path"] = str(arithmetic_dir / "summary.json")

    brs_report = validate_brs_bundle(brs_bundle)
    brs_report["summary_path"] = str(brs_dir / "summary.json")

    return {
        "arithmetic_debug": arithmetic_report,
        "brs": brs_report,
    }
