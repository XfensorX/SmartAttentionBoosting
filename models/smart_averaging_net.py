from dataclasses import dataclass
from queue import Queue
import torch
from utils.self_learning import (
    combine_params,
    switch_params_like_previous_layer,
)
from utils.types import ActivationFunction


@dataclass
class SmartAveragingNetConfig:
    hidden_layer_sizes: list[int]
    input_size: int
    output_size: int
    no_of_outputs: int
    device: torch.device
    trained_output_no: int | None


class SmartAveragingNet(torch.nn.Module):
    def __init__(
        self,
        hidden_layer_sizes: list[int],
        input_size: int,
        output_size: int,
        no_of_outputs: int = 1,
        trained_output_no: int | None = 0,
        activation: ActivationFunction = torch.nn.functional.relu,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        if no_of_outputs < 1:
            raise ValueError("At least one output is needed")

        self.config = SmartAveragingNetConfig(
            hidden_layer_sizes=hidden_layer_sizes,
            input_size=input_size,
            output_size=output_size,
            no_of_outputs=no_of_outputs,
            device=device,
            trained_output_no=-1,  # as otherwise updating will not trigger
        )

        self.activation = activation

        self.hidden_layers = self._get_initialized_hidden_layers()
        self.output_layers = self._get_initialized_output_layers()
        self.set_training_on_output(trained_output_no)

    def _get_initialized_hidden_layers(self):
        if not self.config.hidden_layer_sizes:
            return torch.nn.ModuleList([])

        incoming_nodes = [self.config.input_size] + self.config.hidden_layer_sizes[:-1]
        outgoing_nodes = self.config.hidden_layer_sizes

        return torch.nn.ModuleList(
            [
                torch.nn.Linear(incoming, outgoing, device=self.config.device)
                for incoming, outgoing in zip(incoming_nodes, outgoing_nodes)
            ]
        )

    def _get_initialized_output_layers(self):
        return torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    (
                        self.config.hidden_layer_sizes[-1]
                        if self.config.hidden_layer_sizes
                        else self.config.input_size
                    ),
                    self.config.output_size,
                    device=self.config.device,
                )
                for _ in range(self.config.no_of_outputs)
            ]
        )

    def set_training_on_output(self, output_no: int | None):
        if self.config.trained_output_no == output_no:
            return
        if (output_no is not None) and output_no >= self.config.no_of_outputs:
            raise ValueError(
                f"Can only train outputs with index 0 to {self.config.no_of_outputs - 1}"
            )

        self.config.trained_output_no = output_no
        self._setup_training_only_on_output_no()
        self._setup_faster_output_calculations()

    def _setup_training_only_on_output_no(self):
        for no_of_output_layer, output_layer in enumerate(self.output_layers):
            unfrozen = no_of_output_layer == self.config.trained_output_no
            output_layer.weight.requires_grad = unfrozen
            output_layer.bias.requires_grad = unfrozen

        # TODO is this necessary??
        with torch.no_grad():
            if self.config.trained_output_no is None:
                for parameter in self.parameters():
                    parameter.requires_grad = False

    def _setup_faster_output_calculations(self):

        self._untrained_output_layer: dict[str, torch.nn.Parameter] = {
            "weight": torch.nn.Parameter(
                torch.cat(
                    [
                        (
                            layer.weight.data
                            if output_no != self.config.trained_output_no
                            else torch.zeros_like(layer.weight.data)
                        )
                        for output_no, layer in enumerate(self.output_layers)
                    ]
                ).to(device=self.config.device),
                requires_grad=False,
            ),
            "bias": torch.nn.Parameter(
                torch.cat(
                    [
                        (
                            layer.bias.data
                            if output_no != self.config.trained_output_no
                            else torch.zeros_like(layer.bias.data)
                        )
                        for output_no, layer in enumerate(self.output_layers)
                    ]
                ).to(device=self.config.device),
                requires_grad=False,
            ),
        }

    def set_new_output_params(
        self,
        weights_mapping: dict[int, torch.Tensor],
        bias_mapping: dict[int, torch.Tensor],
    ):
        """
        scaling_mapping: a mapping from the no. of output to the new weights
        """

        for output_no, new_weights in weights_mapping.items():
            assert new_weights.shape == self.output_layers[output_no].weight.shape
            self.output_layers[output_no].weight.data.copy_(new_weights)

        for output_no, new_bias in bias_mapping.items():
            assert new_bias.shape == self.output_layers[output_no].bias.shape
            self.output_layers[output_no].bias.data.copy_(new_bias)

        self._setup_faster_output_calculations()

    @property
    def num_hidden_layers(self):
        return len(self.config.hidden_layer_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)

        #  (([B x H] @  [H x (no_outputs * output_size)]) = [B x (no_outputs * output_size)]
        #  split -> output_size * [B x no_outputs]
        #  stack -> [B x no_outputs x output_size]
        # * [no_outputs x output_size])

        out = torch.stack(
            (
                (x @ self._untrained_output_layer["weight"].T)
                + self._untrained_output_layer["bias"]
            ).split(self.config.output_size, dim=-1),
            dim=-2,
        )
        # -> [B x output_size]
        if self.config.trained_output_no is not None:
            trained_output = self.output_layers[self.config.trained_output_no](x)
            out[:, self.config.trained_output_no, :] = trained_output
        return out.transpose(-1, -2)

    def append_hidden_layer(
        self, nodes: int, postpone_output_layer_update: bool = False
    ):
        last_hidden_layer_output_size_before = (
            self.config.hidden_layer_sizes[-1]
            if self.config.hidden_layer_sizes
            else self.config.input_size
        )
        self.config.hidden_layer_sizes = self.config.hidden_layer_sizes + [nodes]

        self.hidden_layers.append(
            torch.nn.Linear(
                last_hidden_layer_output_size_before, nodes, device=self.config.device
            )
        )

        if not postpone_output_layer_update:
            self.output_layers = self._get_initialized_output_layers()
            self.set_training_on_output(self.config.trained_output_no)

    def get_trained_output_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.trained_output_no is None:
            return (
                self._untrained_output_layer["weights"].data,
                self._untrained_output_layer["bias"].data,
            )

        return (
            self.output_layers[self.config.trained_output_no].weight.data,
            self.output_layers[self.config.trained_output_no].bias.data,
        )

    def get_hidden_params(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.hidden_layers[layer].weight.data,
            self.hidden_layers[layer].bias.data,
        )

    def set_hidden_params(self, layer: int, weights: torch.Tensor, bias: torch.Tensor):
        self.hidden_layers[layer].weight.data.copy_(weights)
        self.hidden_layers[layer].bias.data.copy_(bias)

    def _is_output_layer(self, layer_no: int):
        return layer_no == self.num_hidden_layers

    def normalize_layer(self, layer_no: int):
        """normalizes the given layer_no inside the network.
        - If layer < number of hidden layers, the respective hidden layer gets normalized
        - IF layer ==
        """
        if layer_no > self.num_hidden_layers:
            raise ValueError("Layer too large. Not applicable")

        if self._is_output_layer(layer_no):
            raise ValueError("Last Layer cannot be normalized")

        if self.config.trained_output_no is None:
            raise NotImplementedError("Not important. May not be used.")

        weights, bias = self.get_hidden_params(layer_no)
        norms = torch.sqrt(weights.square().sum(dim=1) + bias.square())
        normed_weights = weights / norms.view(-1, 1)

        self.set_hidden_params(layer_no, normed_weights, bias / norms)

        scaling = norms.to(self.config.device)
        if not self._is_output_layer(layer_no + 1):
            h_weights, h_bias = self.get_hidden_params(layer_no + 1)
            self.set_hidden_params(layer_no + 1, h_weights * scaling, h_bias)
        else:
            weights, bias = self.get_trained_output_params()
            self.set_new_output_params(
                {self.config.trained_output_no: (weights * scaling)},
                {self.config.trained_output_no: bias},
            )

    def normalize(self):
        for layer in range(self.num_hidden_layers):
            self.normalize_layer(layer)

    def add_noise(self, also_on_non_zero_weights: bool = False):
        for layer in range(self.num_hidden_layers):
            weights, bias = self.get_hidden_params(layer)
            w_noise = (
                torch.randn_like(weights) / (weights.shape[0] + weights.shape[1]) * 2
            )
            b_noise = torch.randn_like(bias) / bias.shape[0]
            if also_on_non_zero_weights:
                weights += w_noise
                bias += b_noise
            else:
                weights = torch.where(weights == 0.0, w_noise, weights)
                bias = torch.where(bias == 0.0, b_noise, bias)

            self.set_hidden_params(layer, weights, bias)

    def full_representation(self):
        result = [
            f"SelfLearningNet Weights: (training on output {self.config.trained_output_no})"
        ]
        for i, layer in enumerate(self.hidden_layers):
            result.append(f"Layer {i} Weights:")
            result.append(str(layer.weight.detach().cpu().numpy()))
            result.append(f"Layer {i} Bias:")
            result.append(str(layer.bias.detach().cpu().numpy()))

        for i, layer in enumerate(self.output_layers):
            result.append(f"Output Layer {i} Weights:")
            result.append(str(layer.weight.detach().cpu().numpy()))
            result.append(f"Output Layer {i} Bias:")
            result.append(str(layer.bias.detach().cpu().numpy()))
        return "\n".join(result)

    @staticmethod
    def are_combinable(nets: list["SmartAveragingNet"]):
        if len(nets) < 2:
            return True
        net1 = nets[0]
        return all(
            (
                net1.num_hidden_layers == net2.num_hidden_layers
                and net1.config.input_size == net2.config.input_size
                and net1.config.output_size == net2.config.output_size
                and isinstance(net1.activation, type(net2.activation))
            )
            for net2 in nets[1:]
        )

    @staticmethod
    def combine(
        nets: list["SmartAveragingNet"],
        similarity_threshold_in_degree: float = 45,
        seed: int | None = None,
    ) -> "SmartAveragingNet":
        if seed:
            torch.manual_seed(seed)

        total_nets = len(nets)

        if total_nets < 2:
            raise ValueError(f"Only provided {total_nets} nets. Cannot combine.")

        assert SmartAveragingNet.are_combinable(nets)
        input_size = nets[0].config.input_size
        output_size = nets[0].config.output_size
        activation = nets[0].activation
        intermediate_output_sizes = nets[0].config.hidden_layer_sizes

        netC = SmartAveragingNet(
            [],
            input_size,
            output_size,
            no_of_outputs=len(nets),
            trained_output_no=None,
            activation=activation,
            device=nets[0].config.device,
        )

        for net in nets:
            net.normalize()

        last_output_size = input_size
        weight_permutation_of = {
            net_no: torch.arange(0, input_size) for net_no in range(total_nets)
        }

        for layer in range(nets[0].num_hidden_layers):

            weights, bias = switch_params_like_previous_layer(
                [net.get_hidden_params(layer) for net in nets],
                weight_permutation_of,
                this_layer_output_size=intermediate_output_sizes[layer],
                last_output_size=last_output_size,
            )

            combination_queue: Queue[
                tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor]
            ] = Queue()

            random_net_ordering = torch.randperm(total_nets)
            for index, weight, bias in zip(
                random_net_ordering.tolist(),
                weights[random_net_ordering],
                bias[random_net_ordering],
            ):
                combination_queue.put(
                    (
                        {index: torch.arange(0, len(weight), dtype=torch.long)},
                        weight,
                        bias,
                    )
                )

            while combination_queue.qsize() > 1:
                weight_permutations1, w1, b1 = combination_queue.get()
                weight_permutations2, w2, b2 = combination_queue.get()

                (new_w, new_b), (new_locations_w1, new_locations_w2) = combine_params(
                    w1,
                    w2,
                    b1,
                    b2,
                    similarity_threshold_in_degree,
                    added_zeros_per_row=last_output_size
                    - ([input_size] + intermediate_output_sizes)[layer],
                )

                new_net_permutations = {
                    net_no: new_locations_w1[old_idx_perm]
                    for net_no, old_idx_perm in weight_permutations1.items()
                } | {
                    net_no: new_locations_w2[old_idx_perm]
                    for net_no, old_idx_perm in weight_permutations2.items()
                }

                combination_queue.put((new_net_permutations, new_w, new_b))

            weight_permutation_of, final_weight, final_bias = combination_queue.get()

            last_output_size = final_weight.shape[0]

            netC.append_hidden_layer(
                last_output_size,
                postpone_output_layer_update=(layer != (nets[0].num_hidden_layers - 1)),
            )
            netC.set_hidden_params(layer, final_weight, final_bias)

        all_output_weights, all_output_bias = switch_params_like_previous_layer(
            [net.get_trained_output_params() for net in nets],
            weight_permutation_of,
            this_layer_output_size=output_size,
            last_output_size=last_output_size,
        )

        new_output_weights = {
            net_no: all_output_weights[net_no, :, :] for net_no in range(len(nets))
        }
        new_output_bias = {
            net_no: all_output_bias[net_no, :] for net_no in range(len(nets))
        }

        netC.set_new_output_params(new_output_weights, new_output_bias)
        return netC

    @staticmethod
    def average(
        nets: list["SmartAveragingNet"],
        seed: int | None = None,
    ) -> "SmartAveragingNet":
        if seed:
            torch.manual_seed(seed)

        assert SmartAveragingNet.are_combinable(nets)

        netC = SmartAveragingNet(
            nets[0].config.hidden_layer_sizes,
            nets[0].config.input_size,
            nets[0].config.output_size,
            no_of_outputs=len(nets),
            trained_output_no=None,
            activation=nets[0].activation,
            device=nets[0].config.device,
        )

        for layer in range(nets[0].num_hidden_layers):
            netC.set_hidden_params(
                layer,
                torch.mean(
                    torch.stack([net.get_hidden_params(layer)[0] for net in nets]),
                    dim=0,
                ),
                torch.mean(
                    torch.stack([net.get_hidden_params(layer)[1] for net in nets]),
                    dim=0,
                ),
            )

        new_output_weights = {
            net_no: net.get_trained_output_params()[0]
            for net_no, net in enumerate(nets)
        }
        new_output_bias = {
            net_no: net.get_trained_output_params()[1]
            for net_no, net in enumerate(nets)
        }
        netC.set_new_output_params(new_output_weights, new_output_bias)
        return netC
