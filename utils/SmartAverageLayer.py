from utils.MultiOutputNet import MultiOutputNet


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
    def from_scratch(
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
            output_accumulator="average",
            activation=activation,
        )
        return cls(prediction_network)

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
            add_noise=add_noise,
        )
        return cls(new_prediction_network)

    def forward(self, x: torch.Tensor):
        prediction = self.prediction_network(x)  # outputs (B x O x C)
        return (prediction * self.prediction_mask).sum(dim=-2)
