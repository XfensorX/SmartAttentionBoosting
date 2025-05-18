import pytest

from copy import deepcopy
import torch

from models import SmartAveragingNet

test_configs = [
    ([3, 6, 2], 1, 5, 3, 1, 45),
    ([10, 20, 5], 2, 3, 1, 0, 4),
    ([2, 4, 4, 1], 3, 6, 2, 1, 12),
    ([8, 16, 8, 2], 5, 4, 6, 0, 7),
    ([7, 14, 7, 1], 2, 9, 1, 0, 6),
    ([70, 104, 64, 5, 1, 1, 1, 4, 8, 10, 4, 12], 22, 105, 12, 11, 3),
]


@pytest.mark.parametrize(
    "layers, input_size, output_size, no_of_outputs, trained_output_number, batch_size",
    test_configs,
)
def test_network_consistency(
    layers, input_size, output_size, no_of_outputs, trained_output_number, batch_size
):
    """Tests that the network produces the same output before and after normalization."""
    n = SmartAveragingNet(
        layers, input_size, output_size, no_of_outputs, trained_output_number
    )
    i = torch.rand(batch_size, input_size)
    a = n(i)
    n.normalize()
    b = n(i)
    print(a.shape)
    print(b.shape)
    if no_of_outputs > 1:
        assert torch.all(
            torch.isclose(
                a[:, :, trained_output_number],
                b[:, :, trained_output_number],
                atol=1e-6,
            )
        ), "Network output changed after normalization!"
    else:
        assert torch.all(
            torch.isclose(a, b, atol=1e-6)
        ), "Network output changed after normalization!"


test_configs2 = [
    ([3, 6, 2], 1, 5, 12, 5, 0),
    ([10, 20, 5], 2, 3, 6, 9, 1),
    ([2, 4, 4, 1], 3, 6, 4, 1, 2),
    ([8, 16, 8, 2], 5, 4, 1, 56, 0),
    ([7, 14, 7, 4], 2, 1, 1, 5, 0),
    ([70, 70, 70, 70, 70, 70, 70, 70, 70], 22, 12, 1000, 2048, 7),
]


@pytest.mark.parametrize(
    "layers, input_size, output_size, no_outputs, batch, trained_output_no",
    test_configs2,
)
def test_gradients_of_additional_outputs_do_not_change(
    layers, input_size, output_size, no_outputs, batch, trained_output_no
):

    torch.manual_seed(42)
    n = SmartAveragingNet(
        layers,
        input_size,
        output_size,
        no_outputs,
        trained_output_no,
        activation=torch.nn.Sigmoid(),  ## do not use relu here, as otherwise hidden layer with size 1 could reduce gradients to 0
    )

    trained_last_layer_before = deepcopy(n.output_layers[trained_output_no].weight.data)
    trained_last_layer_bias_before = deepcopy(
        n.output_layers[trained_output_no].bias.data
    )
    if no_outputs > 1:

        other_outputs_last_layers_before = deepcopy(
            torch.stack(
                [
                    n.output_layers[no].weight.data
                    for no in range(no_outputs)
                    if no != trained_output_no
                ]
            )
        )
        other_outputs_last_layers_bias_before = deepcopy(
            torch.stack(
                [
                    n.output_layers[no].bias.data
                    for no in range(no_outputs)
                    if no != trained_output_no
                ]
            )
        )

    input_layer_before = deepcopy(n.get_hidden_params(0)[0])
    input_layer_bias_before = deepcopy(n.get_hidden_params(0)[1])

    sample = torch.randn((batch, input_size))
    optimizer = torch.optim.Adam(n.parameters())
    optimizer.zero_grad()

    y_hat = n(sample)
    assert y_hat.shape == torch.Size((batch, output_size, no_outputs))

    loss = torch.nn.functional.mse_loss(y_hat, torch.rand_like(y_hat))
    loss.backward()
    optimizer.step()

    if no_outputs > 1:
        assert torch.allclose(
            other_outputs_last_layers_before,
            torch.stack(
                [
                    n.output_layers[no].weight.data
                    for no in range(no_outputs)
                    if no != trained_output_no
                ]
            ),
        )
        assert torch.allclose(
            other_outputs_last_layers_bias_before,
            torch.stack(
                [
                    n.output_layers[no].bias.data
                    for no in range(no_outputs)
                    if no != trained_output_no
                ]
            ),
        )

    assert not torch.allclose(
        trained_last_layer_before, n.output_layers[trained_output_no].weight.data
    )

    assert not torch.allclose(
        trained_last_layer_bias_before, n.output_layers[trained_output_no].bias.data
    )

    assert not torch.allclose(input_layer_before, n.get_hidden_params(0)[0])
    assert not torch.allclose(input_layer_bias_before, n.get_hidden_params(0)[1])


@pytest.mark.parametrize(
    "layers, input_size, output_size, no_of_outputs, trained_output_number, batch_size",
    test_configs,
)
def test_combination_leaves_result_and_nets(
    layers,
    input_size,
    output_size,
    no_of_outputs,
    trained_output_number,
    batch_size,
):
    if no_of_outputs == 1:
        return

    torch.manual_seed(42)

    nets = [
        SmartAveragingNet(
            layers, input_size, output_size, no_of_outputs, trained_output_number
        )
        for _ in range(no_of_outputs)
    ]

    sample = torch.randn((batch_size, input_size))
    individual_results = torch.stack([net(sample) for net in nets])  # C x (B x O x C)

    combined_net = SmartAveragingNet.combine(
        nets, similarity_threshold_in_degree=0, seed=42
    )

    combined_results = combined_net(sample)  # (B x O x C)

    torch.manual_seed(42)
    nets_new = [
        SmartAveragingNet(layers, input_size, output_size, no_of_outputs)
        for _ in range(no_of_outputs)
    ]

    individual_results_2 = torch.stack([net(sample) for net in nets_new])

    assert torch.allclose(  # nets are not mutated
        individual_results_2,
        individual_results,
        atol=10e-8,
    )

    assert torch.allclose(  # outputs are not mutated if not combined (threshold 0)
        combined_results,
        individual_results[:, :, :, trained_output_number].permute(1, 2, 0),
        atol=10e-8,
    )


def test_average_of_one_net_is_same():
    torch.manual_seed(42)
    outputs = 23
    inputs = 4
    nets = []
    for i in range(outputs):
        torch.manual_seed(42)
        nets.append(SmartAveragingNet([1234, 1, 54, 345], inputs, 5, outputs, i))

    all_outputs_last_layers_before = deepcopy(
        torch.stack([nets[0].output_layers[no].weight.data for no in range(outputs)])
    )

    all_hidden_layers_before = deepcopy(
        list(nets[0].get_hidden_params(i)[0] for i in range(nets[0].num_hidden_layers))
    )

    all_hidden_layers_bias_before = deepcopy(
        list(nets[0].get_hidden_params(i)[1] for i in range(nets[0].num_hidden_layers))
    )

    n = SmartAveragingNet.average(nets, seed=30)

    assert torch.allclose(
        torch.stack([n.output_layers[no].weight.data for no in range(outputs)]),
        all_outputs_last_layers_before,
    )

    assert n.num_hidden_layers == 4
    for i in range(n.num_hidden_layers):
        assert torch.allclose(n.get_hidden_params(i)[0], all_hidden_layers_before[i])
        assert torch.allclose(
            n.get_hidden_params(i)[1], all_hidden_layers_bias_before[i]
        )


def test_all_layers_are_off():
    n = SmartAveragingNet([1234, 1, 54, 345], 5, 5, 123, 0)
    n.set_training_on_output(None)

    assert all([not param.requires_grad for param in n.parameters()])

    n = SmartAveragingNet([1234, 1, 54, 345], 5, 5, 123, None)
    assert all([not param.requires_grad for param in n.parameters()])
