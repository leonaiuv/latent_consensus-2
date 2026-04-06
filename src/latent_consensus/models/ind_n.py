"""Ind-N shared-weight 对照骨架。"""

from __future__ import annotations

from latent_consensus.models.lcn_shared import LCNSharedModel


class IndNSharedModel(LCNSharedModel):
    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        n_processors: int,
        k_steps: int,
        alpha: float = 0.1,
        mutation_scale: float = 0.0,
        seed: int = 0,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            num_classes=num_classes,
            n_processors=n_processors,
            k_steps=k_steps,
            alpha=alpha,
            observe=False,
            mutation_scale=mutation_scale,
            seed=seed,
        )
