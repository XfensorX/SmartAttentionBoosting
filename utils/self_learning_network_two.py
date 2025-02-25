import torch
from utils.SelfLearningNet import SelfLearningNet
from utils.self_learning import (
    are_combinable,
    NEW_WEIGHT_OPTIONS,
    get_degree_similarities,
    get_indices_to_match,
    complement_indices,
)


def switch_weight_positions(
    weights: torch.Tensor,
    idx_coming_first: torch.Tensor,
    idx_coming_second: torch.Tensor,
    bias_node_included: bool = True,
):
    return weights[
        :,
        torch.concat(
            [
                torch.tensor([0] if bias_node_included else []),
                idx_coming_first,
                idx_coming_second,
            ],
        ),
    ]


def insert_new_weights(
    weights: torch.Tensor,
    at_position: int,
    no_new_weight_rows: int,
    new_weight_initialization: str,
):
    return torch.concat(
        [
            weights[:, :at_position],
            (
                NEW_WEIGHT_OPTIONS[new_weight_initialization](
                    (weights.shape[0], no_new_weight_rows)
                )
                / weights.shape[0]
            ),
            weights[:, at_position:],
        ],
        dim=1,
    )


def switch_weight_positions_and_insert_net_weights(
    net1: SelfLearningNet,
    net2: SelfLearningNet,
    layer: int,
    new_w1_idx_locations_matched: torch.Tensor,
    new_w1_idx_locations_unmatched: torch.Tensor,
    new_w2_idx_locations_matched: torch.Tensor,
    new_w2_idx_locations_unmatched: torch.Tensor,
    new_weight_initialization: str,
):
    w1 = net1.get_weights(layer)
    w2 = net2.get_weights(layer)

    assert w1.shape[0] == w2.shape[0]

    w1 = switch_weight_positions(
        w1, new_w1_idx_locations_matched, new_w1_idx_locations_unmatched
    )

    w2 = switch_weight_positions(
        w2, new_w2_idx_locations_matched, new_w2_idx_locations_unmatched
    )

    w1 = insert_new_weights(
        w1, w1.shape[1], len(new_w2_idx_locations_unmatched), new_weight_initialization
    )

    w2 = insert_new_weights(
        w2,
        len(new_w2_idx_locations_matched) + 1,
        len(new_w1_idx_locations_unmatched),
        new_weight_initialization,
    )

    net1.set_weights(layer, w1)
    net2.set_weights(layer, w2)


def adjust_for_output_scaling(
    weights: torch.Tensor,
    old_scaling_of_w1: torch.Tensor,
    old_scaling_of_w2: torch.Tensor,
    new_w1_idx_locations_matched: torch.Tensor,
    new_w1_idx_locations_unmatched: torch.Tensor,
    new_w2_idx_locations_matched: torch.Tensor,
    new_w2_idx_locations_unmatched: torch.Tensor,
) -> torch.Tensor:
    ## Adjust the weight positioning based on the previously switched neurons in the last-layer
    adapted_scaling_of_w1 = torch.concat(
        [
            old_scaling_of_w1[
                torch.concat(
                    [new_w1_idx_locations_matched, new_w1_idx_locations_unmatched]
                )
            ],
            torch.full((len(new_w2_idx_locations_unmatched),), 0),
        ]
    )
    adapted_scaling_of_w2 = torch.concat(
        [
            old_scaling_of_w2[new_w2_idx_locations_matched],
            torch.full((len(new_w1_idx_locations_unmatched),), 0),
            old_scaling_of_w2[new_w2_idx_locations_unmatched],
        ]
    )

    # TODO: remove the * weights? maybe use some random noise on top for learning disentanglement?
    return weights * torch.concat(
        [torch.tensor([0]), ((adapted_scaling_of_w1 + adapted_scaling_of_w2) / 2)]
    )


def combine_two(
    nets: list[SelfLearningNet],
    similarity_threshold_in_degree: float = 45,
    new_weight_initialization: str = "zeros",
    seed: int | None = None,
):
    """
    new_weight_initialization: zeros | noise (NEW_WEIGHT_OPTIONS constant)

    Important: the input nets are going to get changed.
    They have to deepcopied before if they get used elsewhere after.

    """
    if seed:
        torch.manual_seed(seed)

    assert are_combinable(nets)

    total_nets = len(nets)

    if total_nets < 2:
        raise ValueError(f"Only provided {total_nets} nets. Cannot combine.")

    if total_nets != 2:
        raise NotImplementedError()

    netC = SelfLearningNet(
        [], nets[0].input_size, nets[0].output_size, 0, nets[0].activation
    )

    net1 = nets[0]
    net2 = nets[1]

    w1_idx_to_match = torch.LongTensor(list(range(1, net1.get_weights(0).shape[1]))) - 1
    w1_idx_not_to_match = torch.LongTensor([])
    w2_idx_to_match = torch.LongTensor(list(range(1, net2.get_weights(0).shape[1]))) - 1
    w2_idx_not_to_match = torch.LongTensor([])

    for layer in range(net1.num_layers):

        # Adjust for added bias weight
        new_w1_idx_locations_matched = w1_idx_to_match + 1
        new_w1_idx_locations_unmatched = w1_idx_not_to_match + 1
        new_w2_idx_locations_matched = w2_idx_to_match + 1
        new_w2_idx_locations_unmatched = w2_idx_not_to_match + 1

        switch_weight_positions_and_insert_net_weights(
            net1,
            net2,
            layer,
            new_w1_idx_locations_matched,
            new_w1_idx_locations_unmatched,
            new_w2_idx_locations_matched,
            new_w2_idx_locations_unmatched,
            new_weight_initialization,
        )

        net1.normalize_layer(layer)
        net2.normalize_layer(layer)

        w1 = net1.get_weights(layer)
        w2 = net2.get_weights(layer)

        degree_similarities = get_degree_similarities(w1, w2)

        w1_idx_to_match, w2_idx_to_match = get_indices_to_match(
            degree_similarities, similarity_threshold_in_degree
        )

        w1_idx_not_to_match = complement_indices(w1, w1_idx_to_match)
        w2_idx_not_to_match = complement_indices(w2, w2_idx_to_match)

        new_weights = torch.cat(
            [
                (w1[w1_idx_to_match] + w2[w2_idx_to_match]) / 2,
                w1[w1_idx_not_to_match],
                w2[w2_idx_not_to_match],
            ]
        )

        netC.append_layer(new_weights.shape[0])
        netC.set_weights(layer, new_weights)

    weights = adjust_for_output_scaling(
        weights=netC.get_weights(layer + 1),
        old_scaling_of_w1=net1.get_output_scaling(),
        old_scaling_of_w2=net2.get_output_scaling(),
        new_w1_idx_locations_matched=w1_idx_to_match,
        new_w1_idx_locations_unmatched=w1_idx_not_to_match,
        new_w2_idx_locations_matched=w2_idx_to_match,
        new_w2_idx_locations_unmatched=w2_idx_not_to_match,
    )

    netC.set_weights(layer + 1, weights)

    return netC
