import torch

from utils.general import add_bias_node


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
