from dataclasses import dataclass
from queue import Queue
import torch
from utils.self_learning import (
    combine_weights,
    switch_weights_like_previous_layer,
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

        self._cached_bias = {}

    def _get_initialized_hidden_layers(self):
        if not self.config.hidden_layer_sizes:
            return torch.nn.ModuleList([])

        bias_node = 1  # for overview and understanding
        incoming_nodes = [self.config.input_size] + self.config.hidden_layer_sizes[:-1]
        outgoing_nodes = self.config.hidden_layer_sizes

        return torch.nn.ModuleList(
            [
                # +1 -> adding bias as additional weight
                torch.nn.Linear(
                    incoming + bias_node,
                    outgoing,
                    bias=False,
                    device=self.config.device,
                )
                for incoming, outgoing in zip(incoming_nodes, outgoing_nodes)
            ]
        )

    def _get_initialized_output_layers(self):
        bias_node = 1  # for overview and understanding
        return torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    (
                        self.config.hidden_layer_sizes[-1]
                        if self.config.hidden_layer_sizes
                        else self.config.input_size
                    )
                    + bias_node,
                    self.config.output_size,
                    device=self.config.device,
                    bias=False,
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
            output_layer.weight.requires_grad = (
                no_of_output_layer == self.config.trained_output_no
            )

        with torch.no_grad():
            if self.config.trained_output_no is None:
                for parameter in self.parameters():
                    parameter.requires_grad = False

    def _setup_faster_output_calculations(self):

        self._untrained_output_layer = torch.nn.Parameter(
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
        )

    def set_new_output_weights(self, weights_mapping: dict[int, torch.Tensor]):
        """
        scaling_mapping: a mapping from the no. of output to the new weights
        """

        for output_no, new_weights in weights_mapping.items():
            assert new_weights.shape == self.output_layers[output_no].weight.shape
            self.output_layers[output_no].weight.data = torch.nn.Parameter(new_weights)

        self._setup_faster_output_calculations()

    @property
    def num_hidden_layers(self):
        return len(self.config.hidden_layer_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]  # batch_size
        if b not in self._cached_bias:
            self._cached_bias[b] = torch.ones(
                (b, 1),
                dtype=torch.float32,
                device=self.config.device,
            )

        for layer in self.hidden_layers:
            x = torch.hstack((self._cached_bias[b], x))

            x = layer(x)
            x = self.activation(x)

        x = torch.hstack((self._cached_bias[b], x))

        #  (([B x H] @  [H x (no_outputs * output_size)]) = [B x (no_outputs * output_size)]
        #  split -> output_size * [B x no_outputs]
        #  stack -> [B x no_outputs x output_size]
        # * [no_outputs x output_size])

        out = torch.stack(
            (x @ self._untrained_output_layer.T).split(self.config.output_size, dim=-1),
            dim=-2,
        )
        # -> [B x output_size]
        if self.config.trained_output_no is not None:
            trained_output = self.output_layers[self.config.trained_output_no](x)
            out[:, self.config.trained_output_no, :] = trained_output
        out = out.transpose(-1, -2)
        return out

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
                # add +1 as input for bias weight
                last_hidden_layer_output_size_before + 1,
                nodes,
                bias=False,
                device=self.config.device,
            )
        )

        if not postpone_output_layer_update:
            self.output_layers = self._get_initialized_output_layers()
            self.set_training_on_output(self.config.trained_output_no)

    def get_trained_output_weights(self):
        if self.config.trained_output_no is None:
            return self._untrained_output_layer

        return self.output_layers[self.config.trained_output_no].weight.data

    def get_hidden_weights(self, layer: int):
        return self.hidden_layers[layer].weight.data

    def set_hidden_weights(self, layer: int, weights: torch.Tensor):
        self.hidden_layers[layer].weight.data = torch.nn.Parameter(weights)

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

        weights = self.get_hidden_weights(layer_no)
        norms = weights.norm(p=2, dim=1)
        normed_weights = weights / norms.view(-1, 1)

        self.set_hidden_weights(layer_no, normed_weights)

        scaling = torch.cat(
            (torch.tensor([1], device=self.config.device), norms.to(self.config.device))
        )
        if not self._is_output_layer(layer_no + 1):
            self.set_hidden_weights(
                layer_no + 1,
                self.get_hidden_weights(layer_no + 1) * scaling,
            )

        else:
            self.set_new_output_weights(
                {
                    self.config.trained_output_no: (
                        self.get_trained_output_weights() * scaling
                    )
                }
            )

    def normalize(self):
        for layer in range(self.num_hidden_layers):
            self.normalize_layer(layer)

    def add_noise(self, also_on_non_zero_weights: bool = False):
        for layer in range(self.num_hidden_layers):
            weights = self.get_hidden_weights(layer)
            noise = (
                torch.randn_like(weights) / (weights.shape[0] + weights.shape[1]) * 2
            )
            if also_on_non_zero_weights:
                weights += noise
            else:
                weights = torch.where(weights == 0.0, noise, weights)

            self.set_hidden_weights(layer, weights)

    def full_representation(self):
        result = [
            f"SelfLearningNet Weights: (training on output {self.config.trained_output_no})"
        ]
        for i, layer in enumerate(self.hidden_layers):
            result.append(f"Layer {i} Weights:")
            result.append(str(layer.weight.detach().cpu().numpy()))

        for i, layer in enumerate(self.output_layers):
            result.append(f"Output Layer {i} Weights:")
            result.append(str(layer.weight.detach().cpu().numpy()))
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

            weights = switch_weights_like_previous_layer(
                [net.get_hidden_weights(layer) for net in nets],
                weight_permutation_of,
                this_layer_output_size=intermediate_output_sizes[layer],
                last_output_size=last_output_size,
            )

            combination_queue: Queue[tuple[dict[int, torch.Tensor], torch.Tensor]] = (
                Queue()
            )

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
                    w1,
                    w2,
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

                combination_queue.put((new_net_permutations, new_w))

            weight_permutation_of, final_weight = combination_queue.get()

            last_output_size = final_weight.shape[0]

            netC.append_hidden_layer(
                last_output_size,
                postpone_output_layer_update=(layer != (nets[0].num_hidden_layers - 1)),
            )
            netC.set_hidden_weights(layer, final_weight)

        all_output_weights = switch_weights_like_previous_layer(
            [net.get_trained_output_weights() for net in nets],
            weight_permutation_of,
            this_layer_output_size=output_size,
            last_output_size=last_output_size,
        )

        new_output_weights = {
            net_no: all_output_weights[net_no, :, :] for net_no in range(len(nets))
        }

        netC.set_new_output_weights(new_output_weights)
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
            netC.set_hidden_weights(
                layer,
                torch.mean(
                    torch.stack([net.get_hidden_weights(layer) for net in nets]), dim=0
                ),
            )

        new_output_weights = {
            net_no: net.get_trained_output_weights() for net_no, net in enumerate(nets)
        }

        netC.set_new_output_weights(new_output_weights)
        return netC
