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

        self.layers = torch.nn.ModuleList([])

        prev_nodes = input_size

        bias_node = 1

        for hidden_size in hidden_layers:
            self.layers.append(
                torch.nn.Linear(prev_nodes + bias_node, hidden_size, bias=False)
            )
            prev_nodes = hidden_size

        self.layers.append(
            torch.nn.Linear(prev_nodes + bias_node, output_size, bias=False)
        )

        self.activation = activation

        self.output_scaling = torch.nn.Parameter(torch.full((output_size,), 1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def add_bias_node(tensor: torch.Tensor) -> torch.Tensor:
            batch_size = x.shape[0]
            bias = torch.ones((batch_size, 1), dtype=x.dtype, device=x.device)
            return torch.cat((bias, x), dim=1)

        for layer in self.layers[:-1]:
            x = add_bias_node(x)

            x = layer(x)
            x = self.activation(x)

        return self.layers[-1](add_bias_node(x)) * self.output_scaling

    def set_output_scaling(self, new_scaling: torch.Tensor):
        assert self.output_scaling.shape == new_scaling.shape
        self.output_scaling = torch.nn.Parameter(new_scaling)

    def get_output_scaling(self) -> torch.Tensor:
        return self.output_scaling.data

    def set_weights(self, layer: int, weights: torch.Tensor):
        self.layers[layer].weight.data = torch.nn.Parameter(weights)

    def get_weights(self, layer: int):
        return self.layers[layer].weight.data

    @property
    def num_layers(self):
        return len(self.layers)

    def summary(self) -> str:
        return self.__repr__()

    def __str__(self):
        result = ["SelfLearningNet Weights:"]
        for i, layer in enumerate(self.layers):
            result.append(f"Layer {i} Weights:")
            result.append(str(layer.weight.detach().cpu().numpy()))
        return "\n".join(result)

    def copy_architecture(self):
        return SelfLearningNet(
            self._hidden_layers, self.input_size, self.output_size, self.activation
        )

    def append_layer(self, nodes: int):
        self._hidden_layers = self._hidden_layers + [nodes]
        bias_node = 1
        old_last_layer = self.layers.pop(-1)
        self.layers.append(
            torch.nn.Linear(old_last_layer.weight.shape[1], nodes, bias=False)
        )
        self.layers.append(
            torch.nn.Linear(
                nodes + bias_node, old_last_layer.weight.shape[0], bias=False
            )
        )

    def freeze_layer(self, layer: int):
        self.layers[layer].requires_grad = False

    def unfreeze_layer(self, layer: int):
        self.layers[layer].requires_grad = True

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
        new_weights = weights / norms.view(-1, 1)

        self.set_weights(layer, new_weights)

        if layer + 1 >= self.num_layers:
            self.set_output_scaling(norms * self.get_output_scaling())
        else:
            next_weights = self.get_weights(layer + 1)

            norms = torch.cat((torch.tensor([1]), norms))

            new_next_weights = next_weights * norms
            self.set_weights(layer + 1, new_next_weights)


def normalize_network(net: SelfLearningNet):
    for layer in range(net.num_layers):
        net.normalize_layer(layer)


def combine(
    net1: SelfLearningNet,
    net2: SelfLearningNet,
    similarity_threshold_in_degree=45,
    new_weight_initialization="zeros",
    seed: int = None,
):
    """
    new_weight_initialization: zeros | noise

    """
    if seed:
        torch.manual_seed(seed)

    theta = similarity_threshold_in_degree
    new_weight_options = {"zeros": torch.zeros, "noise": torch.randn}

    assert net1.num_layers == net2.num_layers
    assert net1.input_size == net2.input_size
    assert net1.output_size == net2.output_size
    assert net1.activation.__class__ == net2.activation.__class__

    netC = SelfLearningNet([], net1.input_size, net1.output_size, net1.activation)

    new_w1_idx_locations = None
    new_w2_idx_locations = None

    for layer in range(net1.num_layers):

        ## Extract the weights for layers to merge
        w1 = net1.get_weights(layer)
        w2 = net2.get_weights(layer)

        ## Adjust the weight positioning based on the previously switched neurons in the pre-layer
        if new_w1_idx_locations:

            w1 = torch.concat(
                [
                    # add [0] for additional bias node
                    w1[:, [0] + new_w1_idx_locations[0] + new_w1_idx_locations[1]],
                    new_weight_options[new_weight_initialization](
                        (w1.shape[0], len(new_w2_idx_locations[1]))
                    )
                    / (w1.shape[0]),
                ],
                dim=1,
            )
        if new_w2_idx_locations:
            w2 = torch.concat(
                [
                    # add [0] for additional bias node
                    w2[:, [0] + new_w2_idx_locations[0]],
                    new_weight_options[new_weight_initialization](
                        (w2.shape[0], len(new_w1_idx_locations[1]))
                    )
                    / w2.shape[0],
                    w2[:, new_w2_idx_locations[1]],
                ],
                dim=1,
            )

        net1.set_weights(layer, w1)
        net1.normalize_layer(layer)
        net2.set_weights(layer, w2)
        net2.normalize_layer(layer)
        w1 = net1.get_weights(layer)
        w2 = net2.get_weights(layer)

        similarities = w1 @ w2.transpose(0, 1)

        assert not (torch.any(similarities > 1.001) or torch.any(similarities < -1.001))

        degree_similarities = torch.rad2deg(
            torch.arccos(torch.clamp(similarities, -1.0, 1.0))
        )

        filtered_degrees = torch.where(
            degree_similarities > theta,
            torch.full(degree_similarities.shape, 181),
            degree_similarities,
        )

        poss_merge_idx_w1 = (filtered_degrees != 181).sum(dim=1).sign()
        poss_merge_idx_w2 = (filtered_degrees != 181).sum(dim=0).sign()

        w1_permutation = poss_merge_idx_w1.argsort(descending=True)
        w2_permutation = poss_merge_idx_w2.argsort(descending=True)
        permuted_filtered_degrees = filtered_degrees[w1_permutation, :][
            :, w2_permutation
        ]

        possible_matches = permuted_filtered_degrees[
            : sum(poss_merge_idx_w1), : sum(poss_merge_idx_w2)
        ]

        permuted_w1_idx, permuted_w2_idx = scipy.optimize.linear_sum_assignment(
            possible_matches
        )
        w1_idx_to_match = w1_permutation[permuted_w1_idx]
        w2_idx_to_match = w2_permutation[permuted_w2_idx]

        w1_idx_not_to_match = list(set(range(len(w1))) - set(w1_idx_to_match.tolist()))
        w2_idx_not_to_match = list(set(range(len(w2))) - set(w2_idx_to_match.tolist()))

        w1_not_to_match = w1[w1_idx_not_to_match]
        w2_not_to_match = w2[w2_idx_not_to_match]

        new_matched_weights = (w1[w1_idx_to_match] + w2[w2_idx_to_match]) / 2

        new_weights = torch.cat([new_matched_weights, w1_not_to_match, w2_not_to_match])

        netC.append_layer(new_weights.shape[0])
        netC.set_weights(layer, new_weights)

        # +1 -> adjust for bias node in next layer
        new_w1_idx_locations = (
            (w1_idx_to_match + 1).tolist(),
            [x + 1 for x in w1_idx_not_to_match],
        )
        new_w2_idx_locations = (
            (w2_idx_to_match + 1).tolist(),
            [x + 1 for x in w2_idx_not_to_match],
        )

        ## Extract the weights for layers to merge

    new_w1_idx_locations = (w1_idx_to_match.tolist(), w1_idx_not_to_match)
    new_w2_idx_locations = (w2_idx_to_match.tolist(), w2_idx_not_to_match)

    s1 = net1.get_output_scaling()
    s2 = net2.get_output_scaling()

    ## Adjust the weight positioning based on the previously switched neurons in the last-layer
    s1 = torch.concat(
        [
            s1[new_w1_idx_locations[0] + new_w1_idx_locations[1]],
            torch.full((len(new_w2_idx_locations[1]),), 1),
        ]
    )
    s2 = torch.concat(
        [
            s2[new_w2_idx_locations[0]],
            torch.full((len(new_w1_idx_locations[1]),), 1),
            s2[new_w2_idx_locations[1]],
        ]
    )
    netC.set_weights(
        layer + 1,
        netC.get_weights(layer + 1)
        * torch.concat([torch.tensor([1]), ((s1 + s2) / 2)]),
    )

    return netC
