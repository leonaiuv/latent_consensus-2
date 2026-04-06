"""Causal LM 的最小训练与评估闭环。"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from latent_consensus.training.text_tasks import collate_tokenized_examples, decode_token_ids


class CausalLMTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        output_dir: Path,
        learning_rate: float,
        max_epochs: int,
        batch_size: int,
        gradient_accumulation_steps: int,
        early_stopping_patience: int,
        device: str,
        grad_clip_norm: float = 1.0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正整数")
        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps 必须为正整数")

        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self.grad_clip_norm = grad_clip_norm
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def fit(
        self,
        train_dataset: list,
        val_dataset: list,
    ) -> dict[str, object]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        history: list[dict[str, float | int]] = []
        best_val_loss = float("inf")
        best_epoch = 0
        patience = 0

        for epoch in range(self.max_epochs):
            train_loss = self._train_epoch(train_dataset)
            val_metrics = self.evaluate(val_dataset, split_name="val")
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_exact_match": val_metrics["exact_match"],
            }
            history.append(epoch_record)

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                patience = 0
                self._save_checkpoint(
                    output_path=self.output_dir / "best_checkpoint.pt",
                    history=history,
                    best_epoch=best_epoch,
                )
            else:
                patience += 1

            if patience >= self.early_stopping_patience:
                break

        self._save_checkpoint(
            output_path=self.output_dir / "final_checkpoint.pt",
            history=history,
            best_epoch=best_epoch,
        )
        (self.output_dir / "history.json").write_text(
            json.dumps({"epochs": history, "best_epoch": best_epoch}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {"epochs": history, "best_epoch": best_epoch}

    def evaluate(
        self,
        dataset: list,
        split_name: str,
    ) -> dict[str, object]:
        dataloader = self._build_dataloader(dataset, shuffle=False)
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        exact_match_count = 0
        prediction_records: list[dict[str, object]] = []
        step_correct: dict[int, int] = defaultdict(int)
        step_total: dict[int, int] = defaultdict(int)

        with torch.no_grad():
            for batch in dataloader:
                device_batch = self._to_device(batch)
                output = self.model(
                    input_ids=device_batch["input_ids"],
                    attention_mask=device_batch["attention_mask"],
                    labels=device_batch["labels"],
                )
                total_loss += float(output.loss.item())
                batch_count += 1

                batch_records = self._build_prediction_records(
                    batch=batch,
                    logits=output.logits.detach().cpu(),
                )
                prediction_records.extend(batch_records)
                for record in batch_records:
                    step_total[int(record["step_count"])] += 1
                    if record["answer_correct"]:
                        exact_match_count += 1
                        step_correct[int(record["step_count"])] += 1

        prediction_path = self.output_dir / f"{split_name}_predictions.jsonl"
        with prediction_path.open("w", encoding="utf-8") as handle:
            for record in prediction_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        metrics = {
            "loss": total_loss / max(batch_count, 1),
            "exact_match": exact_match_count / max(len(prediction_records), 1),
            "step_accuracy": {
                str(step_count): step_correct[step_count] / max(step_total[step_count], 1)
                for step_count in sorted(step_total)
            },
        }
        (self.output_dir / f"{split_name}_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return metrics

    def _train_epoch(self, dataset: list) -> float:
        dataloader = self._build_dataloader(dataset, shuffle=True)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        step_count = 0

        for batch_index, batch in enumerate(dataloader, start=1):
            device_batch = self._to_device(batch)
            output = self.model(
                input_ids=device_batch["input_ids"],
                attention_mask=device_batch["attention_mask"],
                labels=device_batch["labels"],
            )
            loss = output.loss / self.gradient_accumulation_steps
            loss.backward()
            total_loss += float(output.loss.item())
            step_count += 1

            if batch_index % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

        if step_count % self.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        return total_loss / max(step_count, 1)

    def _build_prediction_records(
        self,
        batch: dict[str, torch.Tensor | list[str] | list[int]],
        logits: torch.Tensor,
    ) -> list[dict[str, object]]:
        prediction_ids = logits[:, :-1, :].argmax(dim=-1)
        shifted_labels = batch["labels"][:, 1:]
        shifted_answer_mask = batch["answer_mask"][:, 1:].bool()
        records: list[dict[str, object]] = []

        for sample_index, sample_id in enumerate(batch["sample_ids"]):
            relevant_positions = shifted_answer_mask[sample_index] & (
                shifted_labels[sample_index] != -100
            )
            gold_ids = shifted_labels[sample_index][relevant_positions].tolist()
            predicted_ids = prediction_ids[sample_index][relevant_positions].tolist()
            predicted_answer = decode_token_ids(self.tokenizer, predicted_ids)
            answer_correct = predicted_ids == gold_ids and len(gold_ids) > 0
            records.append(
                {
                    "sample_id": sample_id,
                    "step_count": batch["step_counts"][sample_index],
                    "gold_answer": batch["answer_texts"][sample_index],
                    "predicted_answer": predicted_answer,
                    "answer_correct": answer_correct,
                }
            )

        return records

    def _build_dataloader(self, dataset: list, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate_tokenized_examples,
        )

    def _save_checkpoint(
        self,
        output_path: Path,
        history: list[dict[str, float | int]],
        best_epoch: int,
    ) -> None:
        checkpoint = {
            "model_state": self.model.state_dict(),
            "history": history,
            "best_epoch": best_epoch,
        }
        torch.save(checkpoint, output_path)

    def _to_device(
        self,
        batch: dict[str, torch.Tensor | list[str] | list[int]],
    ) -> dict[str, torch.Tensor]:
        return {
            "input_ids": batch["input_ids"].to(self.device),
            "attention_mask": batch["attention_mask"].to(self.device),
            "labels": batch["labels"].to(self.device),
        }
