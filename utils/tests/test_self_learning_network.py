import pytest
import torch

from utils.self_learning_network import SelfLearningNet, combine_two

test_configs = [
    ([3, 6, 2], 1, 5),
    ([10, 20, 5], 2, 3),
    ([2, 4, 4, 1], 3, 6),
    ([8, 16, 8, 2], 5, 4),
    ([7, 14, 7, 1], 2, 9),
    ([70, 104, 64, 5, 1, 1, 1, 4, 8, 10, 4, 12], 22, 105),
]


@pytest.mark.parametrize("layers, input_size, output_size", test_configs)
def test_network_consistency(layers, input_size, output_size):
    """Tests that the network produces the same output before and after normalization."""
    n = SelfLearningNet(layers, input_size, output_size)
    i = torch.rand(1, input_size)
    a = n(i)
    n.normalize()
    b = n(i)
    assert torch.all(
        torch.isclose(a, b, atol=1e-6)
    ), "Network output changed after normalization!"


@pytest.mark.parametrize("layers, input_size, output_size", test_configs)
def test_weight_normalization(layers, input_size, output_size):
    """Ensures that all layer weights are properly normalized (L2 norm ≈ 1)."""
    n = SelfLearningNet(layers, input_size, output_size)
    n.normalize()

    for layer in range(n.num_layers):
        weights = n.get_weights(layer)
        norms = torch.norm(weights, p=2, dim=1)
        assert torch.all(
            torch.isclose(norms, torch.ones_like(norms))
        ), f"Layer {layer} weights are not properly normalized!"


def test_combine_keeps_weights():
    """Tests that combining two models preserves weights."""

    n1_loaded = SelfLearningNet([3, 5, 3], 12, 6)
    n2_loaded = SelfLearningNet([3, 5, 3], 12, 6)

    n1_loaded.load_state_dict(torch.load("utils/tests/n1.pt", weights_only=True))
    n2_loaded.load_state_dict(torch.load("utils/tests/n2.pt", weights_only=True))

    n3 = combine_two(
        [n1_loaded, n2_loaded],
        similarity_threshold_in_degree=45,
        new_weight_initialization="zeros",
        seed=42,
    )

    n3_loaded = SelfLearningNet([6, 9, 5, 9], 12, 6)

    n3_loaded.load_state_dict(torch.load("utils/tests/n3.pt", weights_only=True))

    for param1, param2 in zip(n3.parameters(), n3_loaded.parameters()):
        assert torch.allclose(
            param1.data, param2.data, atol=1e-6
        ), f"Mismatch in weights between n3_combined and n3_saved!"

    assert torch.allclose(n3_loaded.output_scaling, n3.output_scaling)
