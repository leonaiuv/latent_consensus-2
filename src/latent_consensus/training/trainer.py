"""最小 trainer 骨架，用于 smoke 验证训练链路。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from latent_consensus.training.metrics import accuracy_from_logits


class Trainer:
    def __init__(
        self,
        model,
        output_dir: Path,
        learning_rate: float,
        max_epochs: int,
    ) -> None:
        self.model = model
        self.output_dir = Path(output_dir)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def evaluate(self, inputs: np.ndarray, targets: np.ndarray) -> dict[str, float]:
        output = self.model.forward(inputs)
        logits = output.logits
        stabilized_logits = logits - np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(stabilized_logits)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        loss = float(
            -np.mean(np.log(probabilities[np.arange(inputs.shape[0]), targets] + 1e-12))
        )
        return {
            "loss": loss,
            "accuracy": accuracy_from_logits(logits, targets),
        }

    def fit(
        self,
        train_inputs: np.ndarray,
        train_targets: np.ndarray,
        val_inputs: np.ndarray,
        val_targets: np.ndarray,
    ) -> dict[str, object]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        history: list[dict[str, float | int]] = []
        best_val_loss = float("inf")
        best_epoch = 0

        for epoch in range(self.max_epochs):
            train_metrics = self.model.train_batch(
                inputs=train_inputs,
                targets=train_targets,
                learning_rate=self.learning_rate,
            )
            val_metrics = self.evaluate(val_inputs, val_targets)
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
            history.append(epoch_record)

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                self._write_checkpoint(
                    self.output_dir / "best_checkpoint.json",
                    history=history,
                    best_epoch=best_epoch,
                )

        self._write_checkpoint(
            self.output_dir / "final_checkpoint.json",
            history=history,
            best_epoch=best_epoch,
        )
        return {"epochs": history, "best_epoch": best_epoch}

    def _write_checkpoint(
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
        output_path.write_text(
            json.dumps(checkpoint, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
