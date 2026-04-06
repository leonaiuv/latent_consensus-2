"""基于 GPT-2 的 Latent Consensus Causal LM。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModelForCausalLM


@dataclass
class LatentConsensusCausalLMOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    processor_logits: list[torch.Tensor]
    processor_states: list[torch.Tensor]


class LatentConsensusCausalLM(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        n_processors: int,
        k_steps: int,
        observe: bool,
        alpha_init: float = 0.1,
        dropout: float = 0.1,
        noise_std: float = 0.005,
    ) -> None:
        super().__init__()
        if n_processors <= 0:
            raise ValueError("n_processors 必须为正整数")
        if k_steps <= 0:
            raise ValueError("k_steps 必须为正整数")
        if not hasattr(base_model, "transformer") or not hasattr(base_model, "lm_head"):
            raise TypeError("当前 LatentConsensusCausalLM 仅支持 GPT-2 风格的 Causal LM")

        self.base_model = base_model
        self.n_processors = n_processors
        self.k_steps = k_steps
        self.observe = observe
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        hidden_size = int(base_model.config.n_embd)
        self.recurrence_norm = nn.LayerNorm(hidden_size)
        self.recurrence_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)
        self.noise_std = noise_std

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        n_processors: int,
        k_steps: int,
        observe: bool,
        alpha_init: float = 0.1,
        dropout: float = 0.1,
        noise_std: float = 0.005,
    ) -> "LatentConsensusCausalLM":
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        return cls(
            base_model=base_model,
            n_processors=n_processors,
            k_steps=k_steps,
            observe=observe,
            alpha_init=alpha_init,
            dropout=dropout,
            noise_std=noise_std,
        )

    @classmethod
    def from_config(
        cls,
        config,
        n_processors: int,
        k_steps: int,
        observe: bool,
        alpha_init: float = 0.1,
        dropout: float = 0.1,
        noise_std: float = 0.005,
    ) -> "LatentConsensusCausalLM":
        base_model = AutoModelForCausalLM.from_config(config)
        return cls(
            base_model=base_model,
            n_processors=n_processors,
            k_steps=k_steps,
            observe=observe,
            alpha_init=alpha_init,
            dropout=dropout,
            noise_std=noise_std,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> LatentConsensusCausalLMOutput:
        transformer_outputs = self.base_model.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = transformer_outputs.last_hidden_state
        sequence_mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        processor_states = [hidden_states.clone() for _ in range(self.n_processors)]

        for _ in range(self.k_steps):
            mutated_states = [self._mutate_state(state) * sequence_mask for state in processor_states]
            updated_states: list[torch.Tensor] = []
            for processor_index, state in enumerate(mutated_states):
                observe_state = torch.zeros_like(state)
                if self.observe and self.n_processors > 1:
                    observe_state = self._observe(mutated_states, processor_index)

                merged_state = state + self.alpha * observe_state
                delta = self.recurrence_mlp(self.recurrence_norm(merged_state))
                updated_states.append((state + delta) * sequence_mask)
            processor_states = updated_states

        consensus_state = torch.stack(processor_states, dim=0).mean(dim=0)
        logits = self.base_model.lm_head(consensus_state)
        loss = None if labels is None else self._compute_loss(logits, labels)
        processor_logits = [self.base_model.lm_head(state) for state in processor_states]
        return LatentConsensusCausalLMOutput(
            loss=loss,
            logits=logits,
            processor_logits=processor_logits,
            processor_states=processor_states,
        )

    def _mutate_state(self, state: torch.Tensor) -> torch.Tensor:
        mutated = self.dropout(state)
        if self.training and self.noise_std > 0:
            mutated = mutated + torch.randn_like(mutated) * self.noise_std
        return mutated

    def _observe(
        self,
        processor_states: list[torch.Tensor],
        current_index: int,
    ) -> torch.Tensor:
        others = [
            state
            for index, state in enumerate(processor_states)
            if index != current_index
        ]
        return torch.stack(others, dim=0).mean(dim=0)

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        return loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )
