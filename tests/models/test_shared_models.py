import numpy as np

from latent_consensus.models.ind_n import IndNSharedModel
from latent_consensus.models.lc1 import LC1Model
from latent_consensus.models.lcn_shared import LCNSharedModel


def test_lc1_forward_returns_expected_shape() -> None:
    model = LC1Model(hidden_size=4, num_classes=3, k_steps=3, mutation_scale=0.0, seed=7)
    inputs = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.4, 0.3, 0.2]])

    output = model.forward(inputs)

    assert output.logits.shape == (2, 3)
    assert len(output.processor_states) == 1


def test_lcn_n1_matches_lc1_when_mutation_is_disabled() -> None:
    lc1 = LC1Model(hidden_size=4, num_classes=3, k_steps=3, mutation_scale=0.0, seed=11)
    lcn = LCNSharedModel(
        hidden_size=4,
        num_classes=3,
        n_processors=1,
        k_steps=3,
        observe=True,
        mutation_scale=0.0,
        seed=11,
    )
    inputs = np.array([[0.2, 0.1, 0.0, 0.3]])

    lc1_output = lc1.forward(inputs)
    lcn_output = lcn.forward(inputs)

    assert np.allclose(lc1_output.logits, lcn_output.logits)


def test_shared_model_changes_when_k_steps_increase() -> None:
    inputs = np.array([[0.3, 0.2, 0.1, 0.0]])
    k1_model = LC1Model(hidden_size=4, num_classes=2, k_steps=1, mutation_scale=0.0, seed=5)
    k5_model = LC1Model(hidden_size=4, num_classes=2, k_steps=5, mutation_scale=0.0, seed=5)

    k1_output = k1_model.forward(inputs)
    k5_output = k5_model.forward(inputs)

    assert not np.allclose(k1_output.logits, k5_output.logits)


def test_ind_n_observe_off_differs_from_lcn_when_processors_diverge() -> None:
    inputs = np.array([[0.4, 0.3, 0.2, 0.1]])
    lcn = LCNSharedModel(
        hidden_size=4,
        num_classes=2,
        n_processors=2,
        k_steps=3,
        observe=True,
        mutation_scale=0.2,
        seed=13,
    )
    ind_n = IndNSharedModel(
        hidden_size=4,
        num_classes=2,
        n_processors=2,
        k_steps=3,
        mutation_scale=0.2,
        seed=13,
    )

    lcn_output = lcn.forward(inputs)
    ind_output = ind_n.forward(inputs)

    assert not np.allclose(lcn_output.logits, ind_output.logits)
