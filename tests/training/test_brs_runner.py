import json
from pathlib import Path

from transformers import GPT2Config

from latent_consensus.training.brs_runner import run_brs_experiment


ROOT = Path(__file__).resolve().parents[2]


def test_run_brs_experiment_real_mode_writes_predictions(monkeypatch, tmp_path) -> None:
    from latent_consensus.models.latent_consensus_causal_lm import LatentConsensusCausalLM

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

    tokenizer = TinyTokenizer()

    def fake_load_tokenizer(model_name: str):
        assert model_name == "tiny-gpt2"
        return tokenizer

    def fake_build_real_model(spec, model_name: str):
        assert model_name == "tiny-gpt2"
        if spec.model_family == "cot":
            return LatentConsensusCausalLM.from_config(
                config=GPT2Config(
                    vocab_size=128,
                    n_positions=64,
                    n_ctx=64,
                    n_embd=32,
                    n_layer=2,
                    n_head=4,
                ),
                n_processors=1,
                k_steps=1,
                observe=False,
                dropout=0.0,
                noise_std=0.0,
            )
        return LatentConsensusCausalLM.from_config(
            config=GPT2Config(
                vocab_size=128,
                n_positions=64,
                n_ctx=64,
                n_embd=32,
                n_layer=2,
                n_head=4,
            ),
            n_processors=int(spec.resolved_config["model"]["n_processors"]),
            k_steps=2,
            observe=spec.resolved_config["model"]["observe"] == "on",
            dropout=0.0,
            noise_std=0.0,
        )

    monkeypatch.setattr(
        "latent_consensus.training.brs_runner._load_tokenizer",
        fake_load_tokenizer,
    )
    monkeypatch.setattr(
        "latent_consensus.training.brs_runner._build_real_model",
        fake_build_real_model,
    )

    data_dir = tmp_path / "brs"
    data_dir.mkdir(parents=True)
    for split_name in ("train", "val", "test", "ood"):
        file_path = data_dir / f"{split_name}_step2.jsonl"
        file_path.write_text(
            json.dumps(
                {
                    "sample_id": f"{split_name}-0",
                    "step_count": 2,
                    "entities": ["A", "B", "C", "D"],
                    "facts": [["A", "B"], ["B", "C"], ["A", "D"]],
                    "source": "A",
                    "target": "C",
                    "query": "A ? C",
                    "teacher_steps": ["[STEP 1] A > B", "[STEP 2] A > C"],
                    "answer": "A > C",
                    "dead_end_branch": [["A", "D"]],
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    result = run_brs_experiment(
        experiment_id="EXP-B03",
        configs_dir=ROOT / "configs",
        output_root=tmp_path / "results",
        runtime_mode="real",
        data_dir=data_dir,
        model_name="tiny-gpt2",
        device="cpu",
        step_counts=(2,),
        train_limit_per_step=1,
        val_limit_per_step=1,
        test_limit_per_step=1,
        ood_limit_per_step=1,
        max_epochs=1,
        batch_size=1,
        gradient_accumulation_steps=1,
        seq_len=64,
    )

    experiment_dir = tmp_path / "results" / "EXP-B03"
    assert result["runtime_mode"] == "real"
    assert (experiment_dir / "summary.json").is_file()
    assert (experiment_dir / "best_checkpoint.pt").is_file()
    assert (experiment_dir / "test_predictions.jsonl").is_file()
    assert "macro_acc_id" in result
