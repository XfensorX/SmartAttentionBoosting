from copy import deepcopy
from models import MultiOutputNet

from utils.types import ActivationFunction


import torch


class SmartAttentionLayer(torch.nn.Module):
    def __init__(
        self,
        value_network: MultiOutputNet,
        query_network: MultiOutputNet,
        key_network: MultiOutputNet,
    ):
        super().__init__()
        self.value_network = value_network
        self.query_network = query_network
        self.key_network = key_network

    @classmethod
    def initialize_from_scratch(
        cls,
        input_size: int,
        output_size: int,
        num_clients: int,
        this_client_id: int,
        prediction_network_architecture: list[int],
        input_importance_network_architecture: list[int],
        client_importance_network_architecture: list[int],
        activation: ActivationFunction = torch.nn.ReLU(),
    ):
        value_network = MultiOutputNet(
            hidden_layer_sizes=prediction_network_architecture,
            input_size=input_size,
            output_size=output_size,
            no_of_outputs=num_clients,
            trained_output_no=this_client_id,
            activation=activation,
        )

        query_network = MultiOutputNet(
            hidden_layer_sizes=input_importance_network_architecture,
            input_size=input_size,
            output_size=output_size,
            no_of_outputs=num_clients,
            trained_output_no=this_client_id,
            activation=activation,
        )

        key_network = MultiOutputNet(
            hidden_layer_sizes=client_importance_network_architecture,
            input_size=output_size,  #! different
            output_size=num_clients,  #! different
            no_of_outputs=num_clients,
            trained_output_no=this_client_id,
            activation=activation,
        )

        return cls(value_network, query_network, key_network)

    def get_client_model(self, client_no: int, add_noise: bool):
        global_value_network = deepcopy(self.value_network)
        global_query_network = deepcopy(self.query_network)
        global_key_network = deepcopy(self.key_network)

        if add_noise:
            for network in [
                global_value_network,
                global_query_network,
                global_key_network,
            ]:
                network.set_training_on_output(client_no)
                network.add_noise()

        return SmartAttentionLayer(
            global_value_network, global_query_network, global_key_network
        )

    @classmethod
    def get_global_model(
        cls,
        client_models: list["SmartAttentionLayer"],
        similarity_threshold_in_degree: float,
    ):
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

        return cls(new_value_network, new_query_network, new_key_network)

    def forward(self, x: torch.Tensor):
        value = self.value_network(x)  # outputs (B x O x C)
        query = self.query_network(x)  # outputs (B x O x C)
        key = self.key_network(x)  # outputs (B x C x C)

        scale_factor = 1 / torch.sqrt(query.size(-1))

        importances = ((query @ key.transpose(-1, -2)) * scale_factor).softmax(dim=-1)

        # using custom calculations here, as this better fits the purpose
        predictions = (importances * value).sum(dim=-1)

        return predictions
