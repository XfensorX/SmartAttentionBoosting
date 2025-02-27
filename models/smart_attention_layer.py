from copy import deepcopy
import math
from typing import Literal

import torch

from models import MultiOutputNet
from utils.types import ActivationFunction


class SmartAttentionLayer(torch.nn.Module):
    def __init__(
        self,
        value_network: MultiOutputNet,
        query_network: MultiOutputNet,
        key_network: MultiOutputNet,
        device: torch.device = torch.device("cpu"),
    ):

        self.device = device

        super().__init__()
        self.value_network = value_network
        self.query_network = query_network
        self.key_network = key_network

        self.value_network.to(self.device)
        self.query_network.to(self.device)
        self.key_network.to(self.device)

        self.to(self.device)

    @classmethod
    def initialize_from_scratch(
        cls,
        input_size: int,
        output_size: int,
        num_clients: int,
        this_client_id: int | None,
        prediction_network_architecture: list[int],
        input_importance_network_architecture: list[int],
        client_importance_network_architecture: list[int],
        original_input_size: int | None = None,
        activation: ActivationFunction = torch.nn.functional.relu,
        device: torch.device = torch.device("cpu"),
    ):
        if original_input_size is None:
            original_input_size = input_size

        value_network = MultiOutputNet(
            hidden_layer_sizes=prediction_network_architecture,
            input_size=input_size,
            output_size=output_size,
            no_of_outputs=num_clients,
            trained_output_no=this_client_id,
            activation=activation,
            device=device,
        )

        query_network = MultiOutputNet(
            hidden_layer_sizes=input_importance_network_architecture,
            input_size=original_input_size,
            output_size=output_size,
            no_of_outputs=num_clients,
            trained_output_no=this_client_id,
            activation=activation,
            device=device,
        )

        key_network = MultiOutputNet(
            hidden_layer_sizes=client_importance_network_architecture,
            input_size=input_size,
            output_size=num_clients,
            no_of_outputs=num_clients,
            trained_output_no=this_client_id,
            activation=activation,
            device=device,
        )

        new_model = cls(value_network, query_network, key_network, device=device)

        return new_model

    def get_client_model(
        self, client_no: int, add_noise: bool
    ) -> "SmartAttentionLayer":
        global_value_network = deepcopy(self.value_network)
        global_query_network = deepcopy(self.query_network)
        global_key_network = deepcopy(self.key_network)

        for network in [global_value_network, global_query_network, global_key_network]:
            network.set_training_on_output(client_no)
            if add_noise:
                network.add_noise()

        return SmartAttentionLayer(
            global_value_network, global_query_network, global_key_network
        )

    @classmethod
    def get_global_model(
        cls,
        client_models: list["SmartAttentionLayer"],
        similarity_threshold_in_degree: float,
        method: Literal["combine", "average"],
    ):
        if method == "combine":
            new_query_network = MultiOutputNet.combine(
                [model.query_network for model in client_models],
                similarity_threshold_in_degree=similarity_threshold_in_degree,
            )
            new_value_network = MultiOutputNet.combine(
                [model.value_network for model in client_models],
                similarity_threshold_in_degree=similarity_threshold_in_degree,
            )
            new_key_network = MultiOutputNet.combine(
                [model.key_network for model in client_models],
                similarity_threshold_in_degree=similarity_threshold_in_degree,
            )
        elif method == "average":
            new_query_network = MultiOutputNet.average(
                [model.query_network for model in client_models]
            )
            new_value_network = MultiOutputNet.average(
                [model.value_network for model in client_models]
            )
            new_key_network = MultiOutputNet.average(
                [model.key_network for model in client_models]
            )
        else:
            raise ValueError(f"Method {method} not implemented")

        return cls(new_value_network, new_query_network, new_key_network)

    def forward(
        self,
        layer_input: torch.Tensor,
        original_input: torch.Tensor | None = None,
    ):
        if original_input is None:
            original_input = layer_input

        value = self.value_network(layer_input)  # outputs (B x O x C)
        query = self.query_network(original_input)  # outputs (B x O x C)
        key = self.key_network(layer_input)  # outputs (B x C x C)

        scale_factor = 1 / math.sqrt(self.query_network.output_size)

        importances = ((query @ key.transpose(-1, -2)) * scale_factor).softmax(dim=-1)

        # using custom calculations here, as this better fits the purpose
        predictions = (importances * value).sum(dim=-1)

        return predictions
