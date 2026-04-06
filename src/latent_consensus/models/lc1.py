"""LC-1 最小模型骨架。"""

from __future__ import annotations

from latent_consensus.models.lcn_shared import LCNSharedModel


class LC1Model(LCNSharedModel):
    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        k_steps: int,
        alpha: float = 0.1,
        mutation_scale: float = 0.0,
        seed: int = 0,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            num_classes=num_classes,
            n_processors=1,
            k_steps=k_steps,
            alpha=alpha,
            observe=True,
            mutation_scale=mutation_scale,
            seed=seed,
        )
