import torch
from transformers import GPT2Config

from latent_consensus.models.latent_consensus_causal_lm import LatentConsensusCausalLM


def _tiny_config() -> GPT2Config:
    return GPT2Config(
        vocab_size=64,
        n_positions=32,
        n_ctx=32,
        n_embd=32,
        n_layer=2,
        n_head=4,
    )


def test_latent_consensus_causal_lm_produces_logits_and_gradients() -> None:
    torch.manual_seed(7)
    model = LatentConsensusCausalLM.from_config(
        config=_tiny_config(),
        n_processors=3,
        k_steps=2,
        observe=True,
        dropout=0.0,
        noise_std=0.0,
    )
    input_ids = torch.randint(0, 32, (2, 8))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    output.loss.backward()

    assert output.logits.shape == (2, 8, 64)
    assert len(output.processor_logits) == 3
    assert model.alpha.grad is not None


def test_n_equals_one_observe_flag_does_not_change_output() -> None:
    torch.manual_seed(11)
    model_observe_on = LatentConsensusCausalLM.from_config(
        config=_tiny_config(),
        n_processors=1,
        k_steps=2,
        observe=True,
        dropout=0.0,
        noise_std=0.0,
    )
    torch.manual_seed(11)
    model_observe_off = LatentConsensusCausalLM.from_config(
        config=_tiny_config(),
        n_processors=1,
        k_steps=2,
        observe=False,
        dropout=0.0,
        noise_std=0.0,
    )
    model_observe_on.eval()
    model_observe_off.eval()
    input_ids = torch.randint(0, 32, (1, 6))
    attention_mask = torch.ones_like(input_ids)

    output_on = model_observe_on(input_ids=input_ids, attention_mask=attention_mask)
    output_off = model_observe_off(input_ids=input_ids, attention_mask=attention_mask)

    torch.testing.assert_close(output_on.logits, output_off.logits)


def test_k_step_change_is_observable() -> None:
    torch.manual_seed(19)
    model_k1 = LatentConsensusCausalLM.from_config(
        config=_tiny_config(),
        n_processors=2,
        k_steps=1,
        observe=True,
        dropout=0.0,
        noise_std=0.0,
    )
    torch.manual_seed(19)
    model_k5 = LatentConsensusCausalLM.from_config(
        config=_tiny_config(),
        n_processors=2,
        k_steps=5,
        observe=True,
        dropout=0.0,
        noise_std=0.0,
    )
    model_k1.eval()
    model_k5.eval()
    input_ids = torch.randint(0, 32, (1, 6))
    attention_mask = torch.ones_like(input_ids)

    output_k1 = model_k1(input_ids=input_ids, attention_mask=attention_mask)
    output_k5 = model_k5(input_ids=input_ids, attention_mask=attention_mask)

    assert not torch.allclose(output_k1.logits, output_k5.logits)
