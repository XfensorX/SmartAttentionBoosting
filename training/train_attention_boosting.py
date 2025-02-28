from typing import Tuple

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
from data import artificial_1D_linear as data
from experiments.artificial_1D_linear.documentation import (
    evaluate,
    plot_data_split,
    plot_predictions,
)
from experiments.artificial_1D_linear.smart_fed_avg_util import (
    register_client_test_losses,
    train_client,
)
from models import SmartAttentionBoosting, SmartAttentionLayer
from training.general import (
    final_evaluation,
    get_feature_sizes,
    get_loss_fn,
    get_optimizer,
    register_test_loss,
    register_train_loss,
    train_one_epoch,
)

MAX_THETA = 181


def train_attention_boosting(
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
    assert config.boosting_rounds is not None

    input_features, output_features = get_feature_sizes(test_dataloader)

    writer = SummaryWriter(utils.general.get_logging_dir(model_name, data_name))
    loss_fn = get_loss_fn(config.loss_fn)

    global_model = SmartAttentionBoosting(
        input_features,
        output_features,
        config.num_clients,
        config.architecture,
        config.input_importance_network_architecture,
        config.client_importance_network_architecture,
        device=device,
    )

    for boosting_round in tqdm(range(config.boosting_rounds), desc="Boosting Round"):

        global_model.add_new_boosting_layer()

        for communication_round in tqdm(
            range(config.communication_rounds), desc="Communication Round", leave=False
        ):
            is_aligning_round = (
                communication_round >= config.communication_rounds_training
            )

            client_models = [
                global_model.get_client_model(
                    client_id, config.add_noise_in_training and not is_aligning_round
                )
                for client_id in range(config.num_clients)
            ]

            for model in client_models:
                model.to(device)
            # train each client individually

            for client_no in range(config.num_clients):
                optimizer = get_optimizer(
                    config.optimizer,
                    config.learning_rate,
                    client_models[client_no].parameters(),
                )

                for epoch in tqdm(
                    range(config.client_epochs),
                    desc=f"Training client {client_no}",
                    leave=False,
                ):
                    epoch_loss = train_one_epoch(
                        client_models[client_no],
                        client_dataloaders[client_no],
                        optimizer,
                        loss_fn,
                        device,
                    )
                    register_train_loss(
                        writer,
                        epoch_loss,
                        (
                            boosting_round * config.communication_rounds
                            + communication_round
                        )
                        * config.client_epochs
                        + epoch,
                        client_no,
                    )

            global_model.register_new_client_models(
                client_models,
                (
                    config.similarity_threshold_in_degree
                    if not is_aligning_round
                    else MAX_THETA
                ),
                method=config.aligning_method if is_aligning_round else "combine",
            )
            global_model.to(device)

            register_test_loss(
                writer,
                global_model,
                loss_fn,
                boosting_round * config.communication_rounds * config.client_epochs
                + communication_round * config.client_epochs,
                test_dataloader,
            )

    final_evaluation(writer, global_model, test_dataloader, config)
