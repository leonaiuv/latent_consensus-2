import json
from pathlib import Path

from latent_consensus.training.text_tasks import (
    LMExample,
    build_arithmetic_lm_examples,
    collate_tokenized_examples,
    tokenize_lm_examples,
)


class DummyTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 99
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self._vocab = {"<pad>": 0, "<eos>": 99}

    def _encode_text(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for token in text.split():
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab) + 1
            token_ids.append(self._vocab[token])
        return token_ids

    def __call__(
        self,
        text,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        padding: str | bool = False,
    ) -> dict[str, list[int] | list[list[int]]]:
        if isinstance(text, list):
            encoded = [
                self.__call__(
                    item,
                    add_special_tokens=add_special_tokens,
                    truncation=truncation,
                    max_length=max_length,
                    padding=padding,
                )
                for item in text
            ]
            return {
                "input_ids": [item["input_ids"] for item in encoded],
                "attention_mask": [item["attention_mask"] for item in encoded],
            }

        input_ids = self._encode_text(text)
        if add_special_tokens:
            input_ids = input_ids + [self.eos_token_id]
        if truncation and max_length is not None:
            input_ids = input_ids[:max_length]
        attention_mask = [1] * len(input_ids)
        if padding == "max_length" and max_length is not None:
            pad_size = max_length - len(input_ids)
            if pad_size > 0:
                input_ids = input_ids + [self.pad_token_id] * pad_size
                attention_mask = attention_mask + [0] * pad_size
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        reverse_vocab = {value: key for key, value in self._vocab.items()}
        tokens = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in {self.pad_token_id, self.eos_token_id}:
                continue
            tokens.append(reverse_vocab.get(int(token_id), "<unk>"))
        return " ".join(tokens)


def test_build_arithmetic_lm_examples_reads_jsonl_files(tmp_path) -> None:
    sample = {
        "sample_id": "train-step2-0",
        "step_count": 2,
        "expression": "1 + 2",
        "teacher_steps": ["[STEP 1] 1 + 2 = 3", "[STEP 2] 3 + 4 = 7"],
        "answer": "7",
    }
    data_path = tmp_path / "train_step2.jsonl"
    data_path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

    examples = build_arithmetic_lm_examples(
        data_dir=tmp_path,
        split_name="train",
        step_counts=(2,),
    )

    assert len(examples) == 1
    assert examples[0].sample_id == "train-step2-0"
    assert examples[0].step_count == 2
    assert "答案：" in examples[0].target_text


def test_tokenize_lm_examples_masks_prompt_and_marks_answer_tokens() -> None:
    tokenizer = DummyTokenizer()
    examples = [
        LMExample(
            sample_id="sample-1",
            step_count=2,
            prompt_text="题目： 1 + 2 请逐步计算。",
            target_text="[STEP 1] 1 + 2 = 3 答案： 3",
            answer_text="3",
            answer_prefix_text="题目： 1 + 2 请逐步计算。 [STEP 1] 1 + 2 = 3 答案：",
        )
    ]

    encoded = tokenize_lm_examples(
        examples=examples,
        tokenizer=tokenizer,
        seq_len=32,
    )

    assert len(encoded) == 1
    item = encoded[0]
    assert item.labels.count(-100) > 0
    assert any(item.answer_mask)
    assert item.answer_text == "3"


def test_collate_tokenized_examples_stacks_tensors_and_metadata() -> None:
    tokenizer = DummyTokenizer()
    examples = [
        LMExample(
            sample_id="sample-1",
            step_count=2,
            prompt_text="题目： 1 + 2",
            target_text="[STEP 1] 1 + 2 = 3 答案： 3",
            answer_text="3",
            answer_prefix_text="题目： 1 + 2 [STEP 1] 1 + 2 = 3 答案：",
        ),
        LMExample(
            sample_id="sample-2",
            step_count=4,
            prompt_text="题目： 2 + 2",
            target_text="[STEP 1] 2 + 2 = 4 答案： 4",
            answer_text="4",
            answer_prefix_text="题目： 2 + 2 [STEP 1] 2 + 2 = 4 答案：",
        ),
    ]
    encoded = tokenize_lm_examples(examples=examples, tokenizer=tokenizer, seq_len=24)

    batch = collate_tokenized_examples(encoded)

    assert batch["input_ids"].shape == (2, 24)
    assert batch["labels"].shape == (2, 24)
    assert batch["sample_ids"] == ["sample-1", "sample-2"]
    assert batch["step_counts"] == [2, 4]
