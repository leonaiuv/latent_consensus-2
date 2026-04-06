"""Arithmetic-Debug 数据生成逻辑。"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import TypeAlias


@dataclass(frozen=True)
class ArithmeticConfig:
    min_value: int
    max_value: int
    max_result: int = 9999
    operations: tuple[str, ...] = ("+", "-", "*")


@dataclass(frozen=True)
class ArithmeticSample:
    step_count: int
    operands: list[int]
    operations: list[str]
    teacher_steps: list[str]
    expression: str
    result: int
    answer: str


ArithmeticBundle: TypeAlias = dict[str, dict[int, list[ArithmeticSample]]]


def _valid_operands(current: int, operation: str, config: ArithmeticConfig) -> list[int]:
    candidates: list[int] = []
    for operand in range(config.min_value, config.max_value + 1):
        if operation == "+" and current + operand <= config.max_result:
            candidates.append(operand)
        elif operation == "-" and current - operand >= 0:
            candidates.append(operand)
        elif operation == "*" and current * operand <= config.max_result:
            candidates.append(operand)
    return candidates


def _apply_operation(current: int, operation: str, operand: int) -> int:
    if operation == "+":
        return current + operand
    if operation == "-":
        return current - operand
    if operation == "*":
        return current * operand
    raise ValueError(f"不支持的操作: {operation}")


def generate_arithmetic_sample(
    step_count: int,
    config: ArithmeticConfig,
    seed: int,
) -> ArithmeticSample:
    rng = random.Random(seed)
    current = rng.randint(config.min_value, config.max_value)
    operands = [current]
    operations: list[str] = []
    teacher_steps: list[str] = []
    expression_parts = [str(current)]

    for step_index in range(1, step_count + 1):
        valid_operations = [
            operation
            for operation in config.operations
            if _valid_operands(current, operation, config)
        ]
        if not valid_operations:
            raise ValueError("当前配置下无法继续生成合法算术样本")

        operation = rng.choice(valid_operations)
        operand = rng.choice(_valid_operands(current, operation, config))
        current = _apply_operation(current, operation, operand)

        operands.append(operand)
        operations.append(operation)
        expression_parts.extend([operation, str(operand)])
        teacher_steps.append(
            f"[STEP {step_index}] {' '.join(expression_parts)} = {current}"
        )

    expression = " ".join(expression_parts)
    return ArithmeticSample(
        step_count=step_count,
        operands=operands,
        operations=operations,
        teacher_steps=teacher_steps,
        expression=expression,
        result=current,
        answer=str(current),
    )


def arithmetic_sample_signature(sample: ArithmeticSample) -> str:
    return f"{sample.step_count}|{sample.expression}|{sample.answer}"


def serialize_arithmetic_sample(sample: ArithmeticSample) -> dict[str, object]:
    return {
        "step_count": sample.step_count,
        "operands": sample.operands,
        "operations": sample.operations,
        "teacher_steps": sample.teacher_steps,
        "expression": sample.expression,
        "result": sample.result,
        "answer": sample.answer,
    }


def build_arithmetic_dataset_bundle(
    step_counts: tuple[int, ...],
    split_sizes: dict[str, int],
    id_config: ArithmeticConfig,
    ood_config: ArithmeticConfig,
    base_seed: int = 0,
) -> ArithmeticBundle:
    bundle: ArithmeticBundle = {split: {} for split in split_sizes}
    seen_signatures: set[str] = set()
    current_seed = base_seed

    for step_count in step_counts:
        for split_name, sample_count in split_sizes.items():
            config = ood_config if split_name == "ood" else id_config
            samples: list[ArithmeticSample] = []
            attempt_count = 0

            while len(samples) < sample_count:
                sample = generate_arithmetic_sample(
                    step_count=step_count,
                    config=config,
                    seed=current_seed,
                )
                current_seed += 1
                attempt_count += 1

                signature = arithmetic_sample_signature(sample)
                if signature in seen_signatures:
                    if attempt_count > sample_count * 50:
                        raise ValueError("Arithmetic 数据生成未能在限制内产出足够唯一样本")
                    continue

                seen_signatures.add(signature)
                samples.append(sample)

            bundle[split_name][step_count] = samples

    return bundle
