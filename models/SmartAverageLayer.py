from copy import deepcopy
from models.MultiOutputNet import MultiOutputNet


import torch


class SmartAverageLayer(torch.nn.Module):
    def __init__(
        self,
        prediction_network: MultiOutputNet,
    ):
        super().__init__()
        self.prediction_network = prediction_network
        self.prediction_mask = torch.full(
            (prediction_network.output_size, prediction_network.no_of_outputs),
            1 / prediction_network.no_of_outputs,
        )

    # FIXME: rename...
    @classmethod
    def initialize_from_scratch(
        cls,
        input_size: int,
        output_size: int,
        num_clients: int,
        this_client_id: int,
        prediction_network_architecture: list[int],
        activation=torch.nn.ReLU(),
    ):
        prediction_network = MultiOutputNet(
            hidden_layer_sizes=prediction_network_architecture,
            input_size=input_size,
            output_size=output_size,
            no_of_outputs=num_clients,
            trained_output_no=this_client_id,
            activation=activation,
        )

        smart_average_model = cls(prediction_network)

        a = 0.9  # this shifts the weight of the current network in the first setup to around 90%
        smart_average_model.prediction_mask[:, this_client_id] += torch.log2(
            torch.tensor([a * (num_clients - 1) / (1 - a)])
        )

        smart_average_model.prediction_mask = torch.softmax(
            smart_average_model.prediction_mask, dim=1
        )
        return smart_average_model

    def get_client_model(self, client_no: int, add_noise: bool):
        copied_global = deepcopy(self.prediction_network)
        copied_global.set_training_on_output(client_no)
        if add_noise:
            copied_global.add_noise()

        return SmartAverageLayer(copied_global)

    @classmethod
    def get_global_model(
        cls,
        client_models: list["SmartAverageLayer"],
        similarity_threshold_in_degree: float,
        add_noise: bool = True,
    ):
        new_prediction_network = MultiOutputNet.combine(
            [model.prediction_network for model in client_models],
            similarity_threshold_in_degree=similarity_threshold_in_degree,
        )
        return cls(new_prediction_network)

    def forward(self, x: torch.Tensor):
        prediction = self.prediction_network(x)  # outputs (B x O x C)
        return (prediction * self.prediction_mask).sum(dim=-1)
