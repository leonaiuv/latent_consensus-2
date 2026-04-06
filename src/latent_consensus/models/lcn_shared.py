"""shared-weight Latent Consensus 最小模型骨架。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ForwardOutput:
    logits: np.ndarray
    processor_states: list[np.ndarray]


class LCNSharedModel:
    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        n_processors: int,
        k_steps: int,
        alpha: float = 0.1,
        observe: bool = True,
        mutation_scale: float = 0.0,
        seed: int = 0,
    ) -> None:
        if n_processors <= 0:
            raise ValueError("n_processors 必须为正整数")
        if k_steps <= 0:
            raise ValueError("k_steps 必须为正整数")

        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.n_processors = n_processors
        self.k_steps = k_steps
        self.alpha = alpha
        self.observe = observe
        self.mutation_scale = mutation_scale
        self.seed = seed

        rng = np.random.default_rng(seed)
        self.readout = rng.normal(0.0, 0.2, size=(hidden_size, num_classes))
        self.processor_offsets = [
            rng.normal(0.0, 0.1, size=(hidden_size,)) for _ in range(n_processors)
        ]
        self.self_scale = 1.1

    def forward(self, inputs: np.ndarray) -> ForwardOutput:
        processor_states = [inputs.copy() for _ in range(self.n_processors)]

        for _ in range(self.k_steps):
            mutated_states: list[np.ndarray] = []
            for processor_index, state in enumerate(processor_states):
                offset = self.processor_offsets[processor_index] * self.mutation_scale
                mutated_states.append(state + offset)

            updated_states: list[np.ndarray] = []
            for processor_index, state in enumerate(mutated_states):
                observe_state = np.zeros_like(state)
                if self.observe and self.n_processors > 1:
                    others = [
                        other_state
                        for other_index, other_state in enumerate(mutated_states)
                        if other_index != processor_index
                    ]
                    observe_state = np.mean(np.stack(others, axis=0), axis=0)

                updated_states.append(
                    np.tanh(state * self.self_scale + self.alpha * observe_state)
                )

            processor_states = updated_states

        consensus_state = np.mean(np.stack(processor_states, axis=0), axis=0)
        logits = consensus_state @ self.readout
        return ForwardOutput(logits=logits, processor_states=processor_states)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(inputs).logits, axis=1)

    def train_batch(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        learning_rate: float,
    ) -> dict[str, float]:
        output = self.forward(inputs)
        logits = output.logits

        stabilized_logits = logits - np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(stabilized_logits)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)

        batch_size = inputs.shape[0]
        target_probabilities = probabilities[np.arange(batch_size), targets]
        loss = float(-np.mean(np.log(target_probabilities + 1e-12)))

        one_hot_targets = np.zeros_like(probabilities)
        one_hot_targets[np.arange(batch_size), targets] = 1.0
        grad_logits = (probabilities - one_hot_targets) / batch_size

        consensus_state = np.mean(np.stack(output.processor_states, axis=0), axis=0)
        grad_readout = consensus_state.T @ grad_logits
        self.readout = self.readout - learning_rate * grad_readout

        accuracy = float(np.mean(np.argmax(logits, axis=1) == targets))
        return {"loss": loss, "accuracy": accuracy}

    def state_dict(self) -> dict[str, object]:
        return {
            "hidden_size": self.hidden_size,
            "num_classes": self.num_classes,
            "n_processors": self.n_processors,
            "k_steps": self.k_steps,
            "alpha": self.alpha,
            "observe": self.observe,
            "mutation_scale": self.mutation_scale,
            "seed": self.seed,
            "readout": self.readout.tolist(),
            "processor_offsets": [offset.tolist() for offset in self.processor_offsets],
        }
