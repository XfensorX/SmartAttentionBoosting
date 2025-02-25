from queue import Queue
from typing import Literal
import torch

from utils.general import add_bias_node
from utils.self_learning import (
    combine_weights,
    switch_weights_like_previous_layer,
)


class MultiOutputNet(torch.nn.Module):
    def __init__(
        self,
        hidden_layer_sizes: list[int],
        input_size: int,
        output_size: int,
        no_of_outputs: int = 1,
        trained_output_no: int | None = 0,
        output_accumulator: Literal["average", "attention"] = "average",
        activation=torch.nn.ReLU(),
    ):
        super().__init__()

        if no_of_outputs < 1:
            raise ValueError("At least one output is needed")

        self._hidden_layer_sizes = hidden_layer_sizes
        self.input_size = input_size
        self.output_size = output_size
        self.no_of_outputs = no_of_outputs
        self.trained_output_no = -1  # as otherwise updating will not trigger
        self.activation = activation

        self.hidden_layers = self._get_initialized_hidden_layers()
        self.output_layers = self._get_initialized_output_layers()
        self.output_scalings = self._get_initialized_output_scalings()
        self.set_training_on_output(trained_output_no)

        self.accumulator_type = output_accumulator
        self._initialize_accumulator()

    def _initialize_accumulator(self):
        if self.accumulator_type == "average":
            pass
        elif self.accumulator_type == "attention":
            raise NotImplementedError("Attention not Implemented Yet")
        else:
            raise ValueError(
                f"Output Accumulator {self.accumulator_type} is not available"
            )

    def _get_initialized_hidden_layers(self):
        if not self._hidden_layer_sizes:
            return torch.nn.ModuleList([])

        bias_node = 1  # for overview and understanding
        incoming_nodes = [self.input_size] + self._hidden_layer_sizes[:-1]
        outgoing_nodes = self._hidden_layer_sizes

        return torch.nn.ModuleList(
            [
                # +1 -> adding bias as additional weight
                torch.nn.Linear(incoming + bias_node, outgoing, bias=False)
                for incoming, outgoing in zip(incoming_nodes, outgoing_nodes)
            ]
        )

    def _get_initialized_output_layers(self):
        bias_node = 1  # for overview and understanding
        return torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    (
                        self._hidden_layer_sizes[-1]
                        if self._hidden_layer_sizes
                        else self.input_size
                    )
                    + bias_node,
                    self.output_size,
                    bias=False,
                )
                for _ in range(self.no_of_outputs)
            ]
        )

    def _get_initialized_output_scalings(self):
        return torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.ones((self.output_size,), dtype=torch.float32))
                for _ in range(self.no_of_outputs)
            ]
        )

    def set_training_on_output(self, output_no: int | None):
        if self.trained_output_no == output_no:
            return
        if (output_no is not None) and output_no >= self.no_of_outputs:
            raise ValueError(
                f"Can only train outputs with index 0 to {self.no_of_outputs - 1}"
            )

        self.trained_output_no = output_no
        self._setup_training_only_on_output_no()
        self._setup_faster_output_calculations()

    def _setup_training_only_on_output_no(self):
        for no_of_output_layer, (output_layer, output_scaling) in enumerate(
            zip(self.output_layers, self.output_scalings)
        ):
            need_gradients = (no_of_output_layer == self.trained_output_no) or (
                self.trained_output_no is None
            )

            output_layer.weight.requires_grad = need_gradients
            output_scaling.requires_grad = need_gradients

    def _setup_faster_output_calculations(self):

        self._untrained_output_layer = torch.cat(
            [
                (
                    layer.weight.data
                    if output_no != self.trained_output_no
                    else torch.zeros_like(layer.weight.data)
                )
                for output_no, layer in enumerate(self.output_layers)
            ]
        )
        self._untrained_output_scaling = torch.stack(
            [
                (
                    scaling.data
                    if output_no != self.trained_output_no
                    else torch.zeros_like(scaling.data)
                )
                for output_no, scaling in enumerate(self.output_scalings)
            ],
        )

    def set_new_output_scalings(self, scaling_mapping: dict[int, torch.Tensor]):
        """
        scaling_mapping: a mapping from the no. of output to the new scaling vector
        """

        for output_no, new_scaling in scaling_mapping.items():
            assert new_scaling.shape == self.output_scalings[output_no].shape

            self.output_scalings[output_no].data = torch.nn.Parameter(
                new_scaling.float()
            )

        self._setup_faster_output_calculations()

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
        return len(self._hidden_layer_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = add_bias_node(x)

            x = layer(x)
            x = self.activation(x)

        x = add_bias_node(x)

        #  (([B x H] @  [H x (no_outputs * output_size)]) = [B x (no_outputs * output_size)]
        #  split -> output_size * [B x no_outputs]
        #  stack -> [B x no_outputs x output_size]
        # * [no_outputs x output_size])

        out = (
            torch.stack(
                (x @ self._untrained_output_layer.T).split(self.output_size, dim=-1),
                dim=-2,
            )
            * self._untrained_output_scaling
        )
        # -> [B x output_size]
        if self.trained_output_no is not None:
            trained_output = (
                self.output_layers[self.trained_output_no](x)
                * self.output_scalings[self.trained_output_no]
            )
            out[:, self.trained_output_no, :] = trained_output

        out = out.transpose(-1, -2)
        if self.no_of_outputs == 1:
            out = out.squeeze(-1)

        return out

    def append_hidden_layer(
        self, nodes: int, postpone_output_layer_update: bool = False
    ):
        last_hidden_layer_output_size_before = (
            self._hidden_layer_sizes[-1]
            if self._hidden_layer_sizes
            else self.input_size
        )
        self._hidden_layer_sizes = self._hidden_layer_sizes + [nodes]

        self.hidden_layers.append(
            torch.nn.Linear(
                # add +1 as input for bias weight
                last_hidden_layer_output_size_before + 1,
                nodes,
                bias=False,
            )
        )

        if not postpone_output_layer_update:
            self.output_layers = self._get_initialized_output_layers()
            self.output_scalings = self._get_initialized_output_scalings()
            self.set_training_on_output(self.trained_output_no)

    def get_trained_output_weights(self):
        if self.trained_output_no is None:
            return self._untrained_output_layer

        return self.output_layers[self.trained_output_no].weight.data

    def get_trained_output_scalings(self):
        if self.trained_output_no is None:
            return self._untrained_output_scaling

        return self.output_scalings[self.trained_output_no].data

    def get_hidden_weights(self, layer: int):
        return self.hidden_layers[layer].weight.data

    def set_hidden_weights(self, layer: int, weights: torch.Tensor):
        self.hidden_layers[layer].weight.data = torch.nn.Parameter(weights)

    # def freeze_layer(self, layer: int):
    #     self.hidden_layers[layer].weight.requires_grad = False

    # def unfreeze_layer(self, layer: int):
    #     self.hidden_layers[layer].weight.requires_grad = True

    # def freeze_output_scaling(self):
    #     self.output_scaling.requires_grad = False

    # def unfreeze_output_scaling(self):
    #     self.output_scaling.requires_grad = True

    # def freeze_all(self):
    #     for layer in range(self.num_hidden_layers):
    #         self.freeze_layer(layer)

    #     self.freeze_output_scaling()

    # def unfreeze_all(self):
    #     for layer in range(self.num_hidden_layers):
    #         self.unfreeze_layer(layer)

    #     self.unfreeze_output_scaling()

    def _is_output_layer(self, layer_no: int):
        return layer_no == self.num_hidden_layers

    def normalize_layer(self, layer_no: int):
        """normalizes the given layer_no inside the network.
        - If layer < number of hidden layers, the respective hidden layer gets normalized
        - IF layer ==
        """
        if layer_no > self.num_hidden_layers:
            raise ValueError("Layer too large. Not applicable")

        if self.trained_output_no is None:
            raise NotImplementedError("Not important. May not be used.")

        weights = (
            self.get_trained_output_weights()
            if self._is_output_layer(layer_no)
            else self.get_hidden_weights(layer_no)
        )

        norms = torch.norm(weights, p=2, dim=1)

        normed_weights = weights / norms.view(-1, 1)

        if self._is_output_layer(layer_no):
            self.set_new_output_weights({self.trained_output_no: normed_weights})
        else:
            self.set_hidden_weights(layer_no, normed_weights)

        if self._is_output_layer(layer_no):
            self.set_new_output_scalings(
                {self.trained_output_no: (norms * self.get_trained_output_scalings())}
            )
        elif self._is_output_layer(layer_no + 1):
            self.set_new_output_weights(
                {
                    self.trained_output_no: (
                        self.get_trained_output_weights()
                        * torch.cat((torch.tensor([1]), norms))
                    )
                }
            )
        else:
            self.set_hidden_weights(
                layer_no + 1,
                self.get_hidden_weights(layer_no + 1)
                * torch.cat((torch.tensor([1]), norms)),
            )

    def normalize(self):
        for layer in range(self.num_hidden_layers + 1):
            self.normalize_layer(layer)

    def full_representation(self):
        result = [
            f"SelfLearningNet Weights: (training on output {self.trained_output_no})"
        ]
        for i, layer in enumerate(self.hidden_layers):
            result.append(f"Layer {i} Weights:")
            result.append(str(layer.weight.detach().cpu().numpy()))

        for i, (layer, scaling) in enumerate(
            zip(self.output_layers, self.output_scalings)
        ):
            result.append(f"Output Layer {i} Weights:")
            result.append(str(layer.weight.detach().cpu().numpy()))
            result.append("With scaling: ")
            result.append(str(scaling.detach().cpu().numpy()))
        return "\n".join(result)

    @staticmethod
    def are_combinable(nets: list["MultiOutputNet"]):
        if len(nets) < 2:
            return True
        net1 = nets[0]
        return all(
            (
                net1.num_hidden_layers == net2.num_hidden_layers
                and net1.input_size == net2.input_size
                and net1.output_size == net2.output_size
                and isinstance(net1.activation, type(net2.activation))
            )
            for net2 in nets[1:]
        )

    @staticmethod
    def combine(
        nets: list["MultiOutputNet"],
        similarity_threshold_in_degree: float = 45,
        add_noise: bool = False,
        seed: int | None = None,
    ) -> "MultiOutputNet":
        if seed:
            torch.manual_seed(seed)

        total_nets = len(nets)

        if total_nets < 2:
            raise ValueError(f"Only provided {total_nets} nets. Cannot combine.")

        assert MultiOutputNet.are_combinable(nets)
        input_size = nets[0].input_size
        output_size = nets[0].output_size
        activation = nets[0].activation
        intermediate_output_sizes = nets[0]._hidden_layer_sizes

        netC = MultiOutputNet(
            [],
            input_size,
            output_size,
            no_of_outputs=len(nets),
            trained_output_no=None,
            activation=activation,
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

            if add_noise:
                final_weight += (
                    torch.randn(final_weight.shape)
                    / (final_weight.shape[0] + final_weight.shape[1])
                    * 2
                )

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

        new_output_scalings = {
            net_no: net.get_trained_output_scalings() for net_no, net in enumerate(nets)
        }

        netC.set_new_output_scalings(new_output_scalings)
        netC.set_new_output_weights(new_output_weights)
        return netC
