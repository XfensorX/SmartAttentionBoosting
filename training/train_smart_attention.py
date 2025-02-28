from typing import Tuple

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
from models import SmartAttentionLayer
from training.general import (
    final_evaluation,
    get_feature_sizes,
    get_loss_fn,
    get_optimizer,
    register_test_loss,
    register_train_loss,
    train_one_epoch,
)
from utils.general import get_logging_dir

MAX_THETA = 181


def train_smart_attention(
    config: utils.Config,
    test_dataloader: torch.utils.data.DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    client_dataloaders: dict[
        int, torch.utils.data.DataLoader[Tuple[torch.Tensor, torch.Tensor]]
    ],
    device: torch.device,
    data_name: str,
    model_name: str,
):
    assert config.num_clients is not None
    assert config.add_noise_in_training is not None
    assert config.input_importance_network_architecture is not None
    assert config.client_importance_network_architecture is not None
    assert config.communication_rounds is not None
    assert config.communication_rounds_training is not None
    assert config.similarity_threshold_in_degree is not None
    assert config.aligning_method is not None

    input_features, output_features = get_feature_sizes(test_dataloader)

    global_model = SmartAttentionLayer.initialize_from_scratch(
        input_size=input_features,
        output_size=output_features,
        num_clients=config.num_clients,
        this_client_id=None,
        prediction_network_architecture=config.architecture,
        input_importance_network_architecture=config.input_importance_network_architecture,
        client_importance_network_architecture=config.client_importance_network_architecture,
        device=device,
    )
    loss_fn = get_loss_fn(config.loss_fn)
    writer = SummaryWriter(get_logging_dir(model_name, data_name))

    for communication_round in tqdm(
        range(config.communication_rounds), desc="Communication Rounds"
    ):
        is_aligning_round = communication_round >= config.communication_rounds_training

        client_models = [
            global_model.get_client_model(
                client_id, config.add_noise_in_training and not is_aligning_round
            )
            for client_id in range(config.num_clients)
        ]
        optimizers = {
            client_id: get_optimizer(
                config.optimizer,
                config.learning_rate,
                client_models[client_id].parameters(),
            )
            for client_id in range(config.num_clients)
        }

        for model in client_models:
            model.to(device)

        # Train clients
        for client_id in range(config.num_clients):
            for epoch in tqdm(
                range(config.client_epochs),
                leave=False,
                desc=f"Client {client_id} Training",
            ):
                epoch_loss = train_one_epoch(
                    client_models[client_id],
                    client_dataloaders[client_id],
                    optimizers[client_id],
                    loss_fn,
                    device,
                )

                register_train_loss(
                    writer,
                    epoch_loss,
                    communication_round * config.client_epochs + epoch,
                    client_id,
                )

        # Aggregate global model
        global_model = SmartAttentionLayer.get_global_model(
            client_models,
            (
                config.similarity_threshold_in_degree
                if not is_aligning_round
                else MAX_THETA
            ),
            method=config.aligning_method if is_aligning_round else "combine",
        )

        register_test_loss(
            writer,
            global_model,
            loss_fn,
            communication_round * config.client_epochs,
            test_dataloader,
        )

    final_evaluation(writer, global_model, test_dataloader, config)
