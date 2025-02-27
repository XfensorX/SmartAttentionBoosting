import copy
from typing import Literal

import torch

from models import MultiOutputNet, SmartAttentionLayer
from utils.types import ActivationFunction


class SmartAttentionBoosting(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_clients: int,
        prediction_network_architecture: list[int],
        input_importance_network_architecture: list[int],
        client_importance_network_architecture: list[int],
        activation: ActivationFunction = torch.nn.functional.relu,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_clients = num_clients
        self.prediction_network_architecture = prediction_network_architecture
        self.input_importance_network_architecture = (
            input_importance_network_architecture
        )
        self.client_importance_network_architecture = (
            client_importance_network_architecture
        )
        self.activation = activation
        self.device = device

        self.boosting_layers: torch.nn.ModuleList = torch.nn.ModuleList([])

    def add_new_boosting_layer(self):

        self.boosting_layers.append(
            SmartAttentionLayer.initialize_from_scratch(
                self.output_size if self.boosting_layers else self.input_size,
                self.output_size,
                self.num_clients,
                None,
                self.prediction_network_architecture,
                self.input_importance_network_architecture,
                self.client_importance_network_architecture,
                self.input_size,
                self.activation,
                self.device,
            )
        )

    def forward(self, x: torch.Tensor):
        if not self.boosting_layers:
            raise ValueError("No layer created yet.")

        original_input = x

        prediction = self.boosting_layers[0](original_input)

        for layer in self.boosting_layers[1:]:
            prediction = prediction + layer(prediction, original_input)

        return prediction

    def __copy__(self):
        """
        Copies an empty Model of this type with the same parameters.
        """

        return SmartAttentionBoosting(
            self.input_size,
            self.output_size,
            self.num_clients,
            self.prediction_network_architecture,
            self.input_importance_network_architecture,
            self.client_importance_network_architecture,
            self.activation,
            self.device,
        )

    def get_client_model(self, client_no: int, add_noise: bool):
        if not self.boosting_layers:
            raise ValueError("No layers yet")

        model = copy.copy(self)  # shallow copy
        model.boosting_layers = torch.nn.ModuleList([])

        for module in self.boosting_layers[:-1]:
            model.boosting_layers.append(module)

        model.boosting_layers.append(
            self.boosting_layers[-1].get_client_model(client_no, add_noise)
        )

        return model

    def register_new_client_models(
        self,
        client_models: list["SmartAttentionBoosting"],
        similarity_threshold_in_degree: float,
        method: Literal["combine", "average"],
    ):
        self.boosting_layers.pop(-1)
        self.boosting_layers.append(
            SmartAttentionLayer.get_global_model(
                [client_model.boosting_layers[-1] for client_model in client_models],
                similarity_threshold_in_degree,
                method=method,
            )
        )
