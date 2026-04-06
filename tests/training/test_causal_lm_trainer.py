from pathlib import Path

import torch
from transformers import GPT2Config

from latent_consensus.models.latent_consensus_causal_lm import LatentConsensusCausalLM
from latent_consensus.training.causal_lm_trainer import (
    CausalLMTrainer,
    _accumulate_loss_value,
    _finalize_loss_value,
)
from latent_consensus.training.text_tasks import LMExample, collate_tokenized_examples, tokenize_lm_examples


class TinyTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 99
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self._vocab = {"<pad>": 0, "<eos>": 99}

    def _encode(self, text: str) -> list[int]:
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
    ):
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
        input_ids = self._encode(text)
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


def _tiny_examples() -> list[LMExample]:
    return [
        LMExample(
            sample_id="train-1",
            step_count=2,
            prompt_text="题目： 1 + 1",
            target_text="[STEP 1] 1 + 1 = 2 答案： 2",
            answer_text="2",
            answer_prefix_text="题目： 1 + 1 [STEP 1] 1 + 1 = 2 答案：",
        ),
        LMExample(
            sample_id="train-2",
            step_count=4,
            prompt_text="题目： 2 + 2",
            target_text="[STEP 1] 2 + 2 = 4 答案： 4",
            answer_text="4",
            answer_prefix_text="题目： 2 + 2 [STEP 1] 2 + 2 = 4 答案：",
        ),
    ]


def test_causal_lm_trainer_writes_checkpoints_and_predictions(tmp_path: Path) -> None:
    tokenizer = TinyTokenizer()
    encoded = tokenize_lm_examples(_tiny_examples(), tokenizer=tokenizer, seq_len=24)
    model = LatentConsensusCausalLM.from_config(
        config=GPT2Config(
            vocab_size=128,
            n_positions=32,
            n_ctx=32,
            n_embd=32,
            n_layer=2,
            n_head=4,
        ),
        n_processors=2,
        k_steps=2,
        observe=True,
        dropout=0.0,
        noise_std=0.0,
    )
    trainer = CausalLMTrainer(
        model=model,
        tokenizer=tokenizer,
        output_dir=tmp_path,
        learning_rate=1e-3,
        max_epochs=1,
        batch_size=2,
        gradient_accumulation_steps=1,
        early_stopping_patience=1,
        device="cpu",
    )

    history = trainer.fit(train_dataset=encoded, val_dataset=encoded)

    assert len(history["epochs"]) == 1
    assert (tmp_path / "best_checkpoint.pt").is_file()
    assert (tmp_path / "final_checkpoint.pt").is_file()
    assert (tmp_path / "val_predictions.jsonl").is_file()

    metrics = trainer.evaluate(encoded, split_name="eval")
    assert "exact_match" in metrics
    assert "step_accuracy" in metrics


def test_finalize_loss_value_returns_zero_for_empty_accumulator() -> None:
    assert _finalize_loss_value(None, count=0) == 0.0


def test_accumulate_and_finalize_loss_value_keeps_mean_semantics() -> None:
    total_loss = None
    total_loss = _accumulate_loss_value(total_loss, torch.tensor(2.0))
    total_loss = _accumulate_loss_value(total_loss, torch.tensor(4.0))

    assert _finalize_loss_value(total_loss, count=2) == 3.0
