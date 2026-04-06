"""文本任务的数据读取、格式化与编码。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch


ARITHMETIC_PROMPT_TEMPLATE = "题目：{expression}\n请逐步计算并给出最终答案。\n"
ARITHMETIC_ANSWER_PREFIX = "\n答案： "
BRS_ANSWER_PREFIX = "\n答案： "


@dataclass(frozen=True)
class LMExample:
    sample_id: str
    step_count: int
    prompt_text: str
    target_text: str
    answer_text: str
    answer_prefix_text: str


@dataclass(frozen=True)
class TokenizedLMExample:
    sample_id: str
    step_count: int
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    answer_mask: list[int]
    answer_text: str


def build_arithmetic_lm_examples(
    data_dir: Path,
    split_name: str,
    step_counts: tuple[int, ...],
    sample_limit_per_step: int | None = None,
) -> list[LMExample]:
    data_dir = Path(data_dir)
    examples: list[LMExample] = []

    for step_count in step_counts:
        file_path = data_dir / f"{split_name}_step{step_count}.jsonl"
        if not file_path.is_file():
            raise FileNotFoundError(f"Arithmetic 数据文件不存在：{file_path}")

        split_records = file_path.read_text(encoding="utf-8").splitlines()
        if sample_limit_per_step is not None:
            split_records = split_records[:sample_limit_per_step]

        for line in split_records:
            record = json.loads(line)
            prompt_text = ARITHMETIC_PROMPT_TEMPLATE.format(
                expression=record["expression"]
            )
            reasoning_text = "\n".join(record["teacher_steps"])
            target_text = f"{reasoning_text}{ARITHMETIC_ANSWER_PREFIX}{record['answer']}"
            answer_prefix_text = (
                f"{prompt_text}{reasoning_text}{ARITHMETIC_ANSWER_PREFIX}"
            )
            examples.append(
                LMExample(
                    sample_id=record["sample_id"],
                    step_count=record["step_count"],
                    prompt_text=prompt_text,
                    target_text=target_text,
                    answer_text=str(record["answer"]),
                    answer_prefix_text=answer_prefix_text,
                )
            )

    return examples


def build_brs_lm_examples(
    data_dir: Path,
    split_name: str,
    step_counts: tuple[int, ...],
    sample_limit_per_step: int | None = None,
) -> list[LMExample]:
    data_dir = Path(data_dir)
    examples: list[LMExample] = []

    for step_count in step_counts:
        file_path = data_dir / f"{split_name}_step{step_count}.jsonl"
        if not file_path.is_file():
            raise FileNotFoundError(f"BRS 数据文件不存在：{file_path}")

        split_records = file_path.read_text(encoding="utf-8").splitlines()
        if sample_limit_per_step is not None:
            split_records = split_records[:sample_limit_per_step]

        for line in split_records:
            record = json.loads(line)
            prompt_text = _build_brs_prompt_text(record)
            reasoning_text = "\n".join(record["teacher_steps"])
            target_text = f"{reasoning_text}{BRS_ANSWER_PREFIX}{record['answer']}"
            answer_prefix_text = f"{prompt_text}{reasoning_text}{BRS_ANSWER_PREFIX}"
            examples.append(
                LMExample(
                    sample_id=record["sample_id"],
                    step_count=record["step_count"],
                    prompt_text=prompt_text,
                    target_text=target_text,
                    answer_text=str(record["answer"]),
                    answer_prefix_text=answer_prefix_text,
                )
            )

    return examples


def tokenize_lm_examples(
    examples: list[LMExample],
    tokenizer,
    seq_len: int,
) -> list[TokenizedLMExample]:
    if seq_len <= 0:
        raise ValueError("seq_len 必须为正整数")

    _ensure_padding_token(tokenizer)
    encoded_examples: list[TokenizedLMExample] = []

    for example in examples:
        full_text = _join_text_segments(example.answer_prefix_text, example.answer_text)
        prompt_ids = _encode_without_padding(tokenizer, example.prompt_text)
        answer_prefix_ids = _encode_without_padding(tokenizer, example.answer_prefix_text)
        input_ids, attention_mask, offsets = _encode_with_padding(
            tokenizer=tokenizer,
            text=full_text,
            seq_len=seq_len,
            include_offsets=True,
        )
        actual_length = sum(attention_mask)
        if offsets is None:
            labels, answer_mask = _build_masks_from_token_lengths(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_token_count=len(prompt_ids),
                answer_prefix_token_count=len(answer_prefix_ids),
            )
        else:
            labels, answer_mask = _build_masks_from_offsets(
                input_ids=input_ids,
                attention_mask=attention_mask,
                offsets=offsets,
                prompt_char_end=len(example.prompt_text),
                answer_char_start=len(example.answer_prefix_text),
            )

        if all(label == -100 for label in labels):
            raise ValueError(f"{example.sample_id} 的 target 被完全截断")
        if not any(answer_mask):
            raise ValueError(f"{example.sample_id} 的 answer 被完全截断")

        encoded_examples.append(
            TokenizedLMExample(
                sample_id=example.sample_id,
                step_count=example.step_count,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                answer_mask=answer_mask,
                answer_text=example.answer_text,
            )
        )

    return encoded_examples


def collate_tokenized_examples(
    examples: list[TokenizedLMExample],
) -> dict[str, torch.Tensor | list[str] | list[int]]:
    if not examples:
        raise ValueError("collate 不能为空 batch")

    return {
        "input_ids": torch.tensor([example.input_ids for example in examples], dtype=torch.long),
        "attention_mask": torch.tensor(
            [example.attention_mask for example in examples],
            dtype=torch.long,
        ),
        "labels": torch.tensor([example.labels for example in examples], dtype=torch.long),
        "answer_mask": torch.tensor(
            [example.answer_mask for example in examples],
            dtype=torch.long,
        ),
        "sample_ids": [example.sample_id for example in examples],
        "step_counts": [example.step_count for example in examples],
        "answer_texts": [example.answer_text for example in examples],
    }


def decode_token_ids(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def _build_brs_prompt_text(record: dict[str, object]) -> str:
    entities = ", ".join(record["entities"])
    facts = "\n".join(
        f"- {left} > {right}"
        for left, right in record["facts"]
    )
    query = str(record["query"])
    return (
        f"实体：{entities}\n"
        f"事实：\n{facts}\n"
        f"问题：{query}\n"
        "请找出唯一正确链路，并逐步推理后给出最终答案。\n"
    )


def _ensure_padding_token(tokenizer) -> None:
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return

    eos_token = getattr(tokenizer, "eos_token", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token is None or eos_token_id is None:
        raise ValueError("tokenizer 缺少 pad_token 与 eos_token，无法安全 padding")
    tokenizer.pad_token = eos_token
    tokenizer.pad_token_id = eos_token_id


def _encode_without_padding(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    return list(encoded["input_ids"])


def _encode_with_padding(
    tokenizer,
    text: str,
    seq_len: int,
    include_offsets: bool = False,
) -> tuple[list[int], list[int], list[tuple[int, int]] | None]:
    encode_kwargs = {
        "add_special_tokens": False,
        "truncation": True,
        "max_length": seq_len,
        "padding": "max_length",
    }
    try:
        encoded = tokenizer(
            text,
            return_offsets_mapping=include_offsets,
            **encode_kwargs,
        )
    except TypeError:
        encoded = tokenizer(text, **encode_kwargs)

    offsets = encoded.get("offset_mapping")
    resolved_offsets = None
    if offsets is not None:
        resolved_offsets = [tuple(item) for item in offsets]
    return (
        list(encoded["input_ids"]),
        list(encoded["attention_mask"]),
        resolved_offsets,
    )


def _join_text_segments(left: str, right: str) -> str:
    if not left or not right:
        return f"{left}{right}"
    if left[-1].isspace() or right[0].isspace():
        return f"{left}{right}"
    return f"{left} {right}"


def _build_masks_from_token_lengths(
    input_ids: list[int],
    attention_mask: list[int],
    prompt_token_count: int,
    answer_prefix_token_count: int,
) -> tuple[list[int], list[int]]:
    actual_length = sum(attention_mask)
    labels = input_ids.copy()
    label_cutoff = min(prompt_token_count, actual_length)
    answer_start = min(answer_prefix_token_count, actual_length)

    for index in range(len(labels)):
        if index < label_cutoff or index >= actual_length:
            labels[index] = -100

    answer_mask = [0] * len(input_ids)
    for index in range(answer_start, actual_length):
        answer_mask[index] = 1
    return labels, answer_mask


def _build_masks_from_offsets(
    input_ids: list[int],
    attention_mask: list[int],
    offsets: list[tuple[int, int]],
    prompt_char_end: int,
    answer_char_start: int,
) -> tuple[list[int], list[int]]:
    actual_length = sum(attention_mask)
    labels = input_ids.copy()
    answer_mask = [0] * len(input_ids)

    for index in range(len(labels)):
        if index >= actual_length:
            labels[index] = -100
            continue

        start_char, end_char = offsets[index]
        if end_char <= prompt_char_end:
            labels[index] = -100
        if end_char > answer_char_start:
            answer_mask[index] = 1

    return labels, answer_mask
