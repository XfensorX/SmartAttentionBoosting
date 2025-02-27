from copy import deepcopy

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
from models import DenseNetwork
from training.general import (
    final_evaluation,
    get_feature_sizes,
    get_loss_fn,
    get_optimizer,
    register_test_loss,
    register_train_loss,
    train_one_epoch,
)


def train_fed_avg(
    config: utils.Config,
    test_dataloader,
    client_dataloaders: dict[int, torch.utils.data.DataLoader],
    device,
    data_name: str,
    model_name: str,
):
    input_features, output_features = get_feature_sizes(test_dataloader)

    global_model = DenseNetwork(
        DenseNetwork.Config(
            input_features,
            config.architecture,
            output_features,
            torch.nn.ReLU,
            use_batch_norm=config.batch_norm,
            use_layer_norm=config.layer_norm,
            dropout_rate=config.dropout_rate,
        )
    )

    writer = SummaryWriter(utils.general.get_logging_dir(model_name, data_name))
    loss_fn = get_loss_fn(config.loss_fn)

    for communication_round in tqdm(
        range(config.communication_rounds), desc="Communication Round"
    ):
        client_models = {
            client_id: deepcopy(global_model) for client_id in range(config.num_clients)
        }
        optimizers = {
            client_id: get_optimizer(
                config.optimizer,
                config.learning_rate,
                client_models[client_id].parameters(),
            )
            for client_id in range(config.num_clients)
        }

        for m in client_models.values():
            m.to(device)

        for client_id in range(config.num_clients):
            for epoch in tqdm(
                range(config.client_epochs),
                leave=False,
                desc=f"Client Epochs Client {client_id}/{config.num_clients}",
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

        global_model = utils.federated_learning.average_models(
            list(client_models.values())
        )

        register_test_loss(
            writer,
            global_model,
            loss_fn,
            communication_round * config.client_epochs,
            test_dataloader,
        )

    final_evaluation(writer, global_model, test_dataloader, config)
