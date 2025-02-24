from queue import Queue
from random import shuffle
from typing import Callable, Iterable, Sequence
import torch
import scipy


class SelfLearningNet(torch.nn.Module):
    def __init__(
        self,
        hidden_layers: list[int],
        input_size: int = 1,
        output_size: int = 1,
        activation=torch.nn.ReLU(),
    ):
        super().__init__()

        self._hidden_layers = hidden_layers
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        self._initialize_layers()

        self.output_scaling = torch.nn.Parameter(
            torch.ones((output_size,), dtype=torch.float32)
        )

    def _initialize_layers(self):
        incoming_nodes = [self.input_size] + self._hidden_layers
        outgoing_nodes = self._hidden_layers + [self.output_size]

        self.layers = torch.nn.ModuleList(
            [
                # +1 -> adding bias as additional weight
                torch.nn.Linear(incoming + 1, outgoing, bias=False)
                for incoming, outgoing in zip(incoming_nodes, outgoing_nodes)
            ]
        )

    @property
    def num_layers(self):
        return len(self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = add_bias_node(x)
            x = layer(x)
            x = self.activation(x)

        return self.layers[-1](add_bias_node(x)) * self.output_scaling

    def get_output_scaling(self) -> torch.Tensor:
        return self.output_scaling.data

    def set_output_scaling(self, new_scaling: torch.Tensor):
        assert self.output_scaling.shape == new_scaling.shape
        self.output_scaling = torch.nn.Parameter(new_scaling)

    def get_weights(self, layer: int):
        return self.layers[layer].weight.data

    def set_weights(self, layer: int, weights: torch.Tensor):
        self.layers[layer].weight.data = torch.nn.Parameter(weights)

    def full_representation(self):
        result = ["SelfLearningNet Weights:"]
        for i, layer in enumerate(self.layers):
            result.append(f"Layer {i} Weights:")
            result.append(str(layer.weight.detach().cpu().numpy()))
        return "\n".join(result)

    def append_layer(self, nodes: int):
        self._hidden_layers = self._hidden_layers + [nodes]
        old_last_layer = self.layers.pop(-1)
        self.layers.append(
            torch.nn.Linear(old_last_layer.weight.shape[1], nodes, bias=False)
        )
        self.layers.append(
            torch.nn.Linear(
                # add +1 as input for bias weight
                nodes + 1,
                old_last_layer.weight.shape[0],
                bias=False,
            )
        )

    def freeze_layer(self, layer: int):
        self.layers[layer].weight.requires_grad = False

    def unfreeze_layer(self, layer: int):
        self.layers[layer].weight.requires_grad = True

    def freeze_output_scaling(self):
        self.output_scaling.requires_grad = False

    def unfreeze_output_scaling(self):
        self.output_scaling.requires_grad = True

    def freeze_all(self):
        for layer in range(self.num_layers):
            self.freeze_layer(layer)

        self.freeze_output_scaling()

    def unfreeze_all(self):
        for layer in range(self.num_layers):
            self.unfreeze_layer(layer)

        self.unfreeze_output_scaling()

    def normalize_layer(self, layer: int):
        weights = self.get_weights(layer)
        norms = torch.norm(weights, p=2, dim=1)

        self.set_weights(layer, weights / norms.view(-1, 1))

        next_layer = layer + 1

        if next_layer >= self.num_layers:
            self.set_output_scaling(norms * self.get_output_scaling())
        else:
            self.set_weights(
                next_layer,
                self.get_weights(next_layer) * torch.cat((torch.tensor([1]), norms)),
            )

    def normalize(self):
        for layer in range(self.num_layers):
            self.normalize_layer(layer)


def add_bias_node(tensor: torch.Tensor) -> torch.Tensor:
    batch_size = tensor.shape[0]
    bias = torch.ones((batch_size, 1), dtype=tensor.dtype, device=tensor.device)
    return torch.cat((bias, tensor), dim=1)


NEW_WEIGHT_OPTIONS: dict[str, Callable[[Sequence[int]], torch.Tensor]] = {
    "zeros": torch.zeros,
    "noise": torch.randn,
}


def are_combinable(nets: list[SelfLearningNet]):
    if len(nets) < 2:
        return True
    net1 = nets[0]
    return all(
        (
            net1.num_layers == net2.num_layers
            and net1.input_size == net2.input_size
            and net1.output_size == net2.output_size
            and isinstance(net1.activation, type(net2.activation))
        )
        for net2 in nets[1:]
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


def get_degree_similarities(
    weights_a: torch.Tensor,
    weights_b: torch.Tensor,
) -> torch.Tensor:
    """
    Returns degree between each individual weight vector (row-vector) of the two input weight matricies

    If the angle is greated than the similarity threshold, it is set to 181 degrees.
    """
    similarities = weights_a @ weights_b.transpose(0, 1)
    assert not (torch.any(similarities > 1.0001) or torch.any(similarities < -1.0001))

    return similarities.clamp(-1.0, 1.0).arccos().rad2deg()


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


def get_indices_to_match(
    degree_similarities: torch.Tensor, similarity_threshold_in_degree: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a similarity matrix (in degrees), returns the indices of the respective vectors,
    that are the closest to each other and should be matched.

    returns: two tensors with
        - the indices in the 0-dimension
        - the indices of the 1-dimension
    in the order they belong together respectively.
    e.g.:
    (tensor[1,5], tensor [2,4]) -> vectors 1 in 0-dim and 2 in 1-dim should be matched, ...
    """

    OUT_OF_REACH_DEGREE_PENALTY = 99999999999.0

    filtered_degrees = torch.where(
        degree_similarities > similarity_threshold_in_degree,
        torch.full(degree_similarities.shape, OUT_OF_REACH_DEGREE_PENALTY),
        degree_similarities,
    )
    # for better performance, we first permute the matrix to push all non-matching pairs (degree==penalty)
    # to the bottom of the matrix and only calculate based on the rest of the vectors
    # --> 1 if matching, 0 if not matching
    dim0_is_possible_match = (
        (filtered_degrees != OUT_OF_REACH_DEGREE_PENALTY).sum(dim=1).sign()
    )
    dim1_is_possible_match = (
        (filtered_degrees != OUT_OF_REACH_DEGREE_PENALTY).sum(dim=0).sign()
    )

    # Permute possible matched to the upper left corner of the matrix
    dim0_permutation = dim0_is_possible_match.argsort(descending=True)
    dim1_permutation = dim1_is_possible_match.argsort(descending=True)
    permuted_filtered_degrees = filtered_degrees[dim0_permutation, :][
        :, dim1_permutation
    ]
    # all rows and cols without a match should not be included in the assignment search (otherwise they will get matched)
    permuted_idx_dim0, permuted_idx_dim1 = scipy.optimize.linear_sum_assignment(
        permuted_filtered_degrees[
            : sum(dim0_is_possible_match), : sum(dim1_is_possible_match)
        ]
    )
    dim0_idx_to_match = dim0_permutation[permuted_idx_dim0]
    dim1_idx_to_match = dim1_permutation[permuted_idx_dim1]

    # Nevertheless all matches which should have not been merged have to be filtered out
    wrong_match_idx = torch.nonzero(
        filtered_degrees[dim0_idx_to_match, dim1_idx_to_match]
        == OUT_OF_REACH_DEGREE_PENALTY
    ).reshape(-1)

    correct_match_mask = torch.ones(dim0_idx_to_match.shape, dtype=torch.bool)
    correct_match_mask[wrong_match_idx] = False

    return dim0_idx_to_match[correct_match_mask], dim1_idx_to_match[correct_match_mask]


def complement_indices(of: torch.Tensor, indices: torch.Tensor, dim: int = 0):
    all_indices = range(of.shape[dim])
    return torch.LongTensor(list(set(all_indices) - set(indices.tolist())))


def combine(
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
        [], nets[0].input_size, nets[0].output_size, nets[0].activation
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


def get_new_idx_permutation(
    idx_to_match: torch.Tensor, idx_not_to_match: torch.Tensor, not_to_match_offset: int
):
    length_of_matched = len(idx_to_match)
    total_length = len(idx_to_match) + len(idx_not_to_match)

    new_locations = torch.zeros((total_length,), dtype=torch.long)
    new_locations[idx_to_match] = torch.arange(0, length_of_matched, dtype=torch.long)
    new_locations[idx_not_to_match] = not_to_match_offset + torch.arange(
        length_of_matched, total_length, dtype=torch.long
    )

    return new_locations


def switch_weights_like_previous_layer(
    weights: list[torch.Tensor],
    weight_permutations: dict[int, torch.Tensor],
    this_layer_output_size: int,
    last_output_size: int,
) -> torch.Tensor:
    combined_weights = torch.zeros(
        (
            len(weights),
            this_layer_output_size,
            last_output_size + 1,  # +1 for bias node
        )
    )

    for net_no, net_weight in enumerate(weights):
        combined_weights[
            net_no,
            :,
            torch.cat(
                [torch.tensor([0]), weight_permutations[net_no] + 1]
            ),  # shift for bias node
        ] = net_weight

    return combined_weights


def combine_weights(
    w1: torch.Tensor, w2: torch.Tensor, similarity_threshold_in_degree: int
):

    degree_similarities = get_degree_similarities(w1, w2)

    w1_idx_to_match, w2_idx_to_match = get_indices_to_match(
        degree_similarities, similarity_threshold_in_degree
    )

    w1_idx_not_to_match = complement_indices(w1, w1_idx_to_match)
    w2_idx_not_to_match = complement_indices(w2, w2_idx_to_match)

    new_w = torch.cat(
        [
            (w1[w1_idx_to_match] + w2[w2_idx_to_match]) / 2,
            w1[w1_idx_not_to_match],
            w2[w2_idx_not_to_match],
        ]
    )

    new_locations_w1 = get_new_idx_permutation(
        w1_idx_to_match, w1_idx_not_to_match, not_to_match_offset=0
    )

    new_locations_w2 = get_new_idx_permutation(
        w2_idx_to_match,
        w2_idx_not_to_match,
        not_to_match_offset=len(w1_idx_not_to_match),
    )

    return new_w, (new_locations_w1, new_locations_w2)


def combine_several(
    nets: list[SelfLearningNet],
    similarity_threshold_in_degree: float = 45,
    seed: int | None = None,
):
    """
    new_weight_initialization: zeros | noise (NEW_WEIGHT_OPTIONS constant)

    Important: the input nets are going to get changed.
    They have to deepcopied before if they get used elsewhere after.

    """
    if seed:
        torch.manual_seed(seed)

    total_nets = len(nets)

    if total_nets < 2:
        raise ValueError(f"Only provided {total_nets} nets. Cannot combine.")

    assert are_combinable(nets)
    input_size = nets[0].input_size
    output_size = nets[0].output_size
    activation = nets[0].activation
    intermediate_output_sizes = nets[0]._hidden_layers + [output_size]

    netC = SelfLearningNet([], input_size, output_size, activation)

    for net in nets:
        net.normalize()

    last_output_size = input_size
    weight_permutation_of = {
        net_no: torch.arange(0, input_size) for net_no in range(total_nets)
    }

    for layer in range(nets[0].num_layers):

        weights = switch_weights_like_previous_layer(
            [net.get_weights(layer) for net in nets],
            weight_permutation_of,
            this_layer_output_size=intermediate_output_sizes[layer],
            last_output_size=last_output_size,
        )

        combination_queue: Queue[tuple[dict[int, torch.Tensor], torch.Tensor]] = Queue()

        random_net_ordering = torch.randperm(total_nets)

        for index, weight in zip(
            random_net_ordering.tolist(), weights[random_net_ordering]
        ):
            combination_queue.put(
                ({index: torch.arange(0, len(weight), dtype=torch.long)}, weight)
            )

        while combination_queue.qsize() > 1:
            weight_permutations1, w1 = combination_queue.get()
            weight_permutations2, w2 = combination_queue.get()

            new_w, (new_locations_w1, new_locations_w2) = combine_weights(
                w1, w2, similarity_threshold_in_degree
            )

            new_net_permutations = {
                net_no: new_locations_w1[old_idx_perm]
                for net_no, old_idx_perm in weight_permutations1.items()
            } | {
                net_no: new_locations_w2[old_idx_perm]
                for net_no, old_idx_perm in weight_permutations2.items()
            }

            combination_queue.put((new_net_permutations, new_w))

        weight_permutation_of, final_weight = combination_queue.get()
        last_output_size = final_weight.shape[0]

        netC.append_layer(last_output_size)
        netC.set_weights(layer, final_weight)

    return netC
