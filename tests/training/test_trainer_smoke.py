import json

import numpy as np

from latent_consensus.models.lc1 import LC1Model
from latent_consensus.training.trainer import Trainer


def test_trainer_runs_single_epoch_and_writes_checkpoints(tmp_path) -> None:
    model = LC1Model(hidden_size=4, num_classes=2, k_steps=2, mutation_scale=0.0, seed=17)
    trainer = Trainer(
        model=model,
        output_dir=tmp_path,
        learning_rate=0.1,
        max_epochs=2,
    )
    train_inputs = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.5, 0.0, 0.0],
            [0.0, 0.5, 1.0, 0.0],
        ]
    )
    train_targets = np.array([0, 1, 0, 1])
    val_inputs = train_inputs.copy()
    val_targets = train_targets.copy()

    history = trainer.fit(
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
    )

    assert len(history["epochs"]) == 2
    assert (tmp_path / "best_checkpoint.json").is_file()
    assert (tmp_path / "final_checkpoint.json").is_file()

    final_checkpoint = json.loads((tmp_path / "final_checkpoint.json").read_text(encoding="utf-8"))
    assert "model_state" in final_checkpoint
    assert "history" in final_checkpoint
