import json
from copy import deepcopy
from typing import Callable, Literal, Tuple

import torch
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from tqdm import tqdm

import utils
from models import DenseNetwork, SmartAttentionLayer
from utils.Config import Config
from utils.general import get_logging_dir


def get_optimizer(
    optimizer_class: Literal["adam"], lr: float, parameters
) -> torch.optim.Optimizer:
    if optimizer_class == "adam":
        return torch.optim.Adam(parameters, lr=lr)
    else:
        raise ValueError("Optimizer not implemented")


def get_loss_fn(
    loss_fn_type: str,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if loss_fn_type == "bce_with_logits":
        return torch.nn.BCEWithLogitsLoss()
    elif loss_fn_type == "mse":
        return torch.nn.MSELoss()
    else:
        raise ValueError("Loss function not implemented")


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader[Tuple[torch.Tensor, ...]],
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
):
    model.train()
    epoch_losses: list[float] = []
    for x, y in dataloader:
        optimizer.zero_grad()
        x, y = x.to(device), y.to(torch.float32).to(device)
        y_hat = model(x)

        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    return sum(epoch_losses) / len(epoch_losses)


def train_fed_avg(
    config: Config,
    test_dataloader,
    client_dataloaders: dict[int, torch.utils.data.DataLoader],
    device,
    data_name: str,
    model_name: str,
):
    x_sample, y_sample = next(((x, y) for (x, y) in test_dataloader))
    input_features = x_sample.shape[1]
    output_features = y_sample.shape[1]

    model_config = DenseNetwork.Config(
        input_features,
        config.architecture,
        output_features,
        torch.nn.ReLU,
        use_batch_norm=config.batch_norm,
        use_layer_norm=config.layer_norm,
        dropout_rate=config.dropout_rate,
    )
    global_model = DenseNetwork(model_config)

    writer = SummaryWriter(get_logging_dir(model_name, data_name))
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
                writer.add_scalar(
                    f"Loss/train/client{client_id}",
                    epoch_loss,
                    communication_round * config.client_epochs + epoch,
                )

        global_model = utils.federated_learning.average_models(
            list(client_models.values())
        )
        y_hats, ys = utils.evaluation.evaluate(
            global_model, test_dataloader, from_logits=True, return_outputs_only=True
        )
        writer.add_scalar(
            "Loss/test",
            loss_fn(y_hats.to(torch.float), ys.to(torch.float)).item(),
            communication_round * config.client_epochs,
        )

    metrics = utils.evaluation.evaluate(global_model, test_dataloader, from_logits=True)
    writer.add_hparams(dict(metrics), {}, run_name=".")
    writer.add_text("hparams", json.dumps(config.model_dump(), indent=4))
    writer.add_text(
        "Model Summary",
        str(summary(global_model, input_size=(1, input_features), verbose=0)),
    )
    writer.add_graph(global_model, torch.randn(1, input_features))
    writer.flush()
    writer.close()


def train_basic_nn(
    config: Config,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    data_name: str,
    model_name: str,
):
    x_sample, y_sample = next(((x, y) for (x, y) in test_dataloader))
    input_features = x_sample.shape[1]
    output_features = y_sample.shape[1]

    model_config = DenseNetwork.Config(
        input_features,
        config.architecture,
        output_features,
        torch.nn.ReLU,
        use_batch_norm=config.batch_norm,
        use_layer_norm=config.layer_norm,
        dropout_rate=config.dropout_rate,
    )
    model = DenseNetwork(model_config).to(device)

    # Initialize logging
    writer = SummaryWriter(get_logging_dir(model_name, data_name))

    dummy_input = torch.randn(1, input_features)  # Example input for visualization
    writer.add_graph(model, dummy_input)

    # Initialize optimizer & loss function
    optimizer = get_optimizer(
        config.optimizer, config.learning_rate, model.parameters()
    )
    loss_fn = get_loss_fn(config.loss_fn)

    # Training loop
    for epoch in tqdm(range(config.client_epochs), desc="Training Basic NN"):
        epoch_loss = train_one_epoch(
            model, train_dataloader, optimizer, loss_fn, device
        )
        writer.add_scalar("Loss/train", epoch_loss, epoch)

        y_hats, ys = utils.evaluation.evaluate(
            model, test_dataloader, from_logits=True, return_outputs_only=True
        )
        writer.add_scalar(
            "Loss/test",
            loss_fn(y_hats.to(torch.float), ys.to(torch.float)).item(),
            epoch,
        )

    # Final evaluation
    metrics = utils.evaluation.evaluate(model, test_dataloader, from_logits=True)
    writer.add_hparams(dict(metrics), {}, run_name=".")
    writer.add_text("hparams", json.dumps(config.model_dump(), indent=4))
    writer.add_text(
        "Model Summary", str(summary(model, input_size=(1, input_features), verbose=0))
    )
    writer.flush()
    writer.close()


def train_smart_attention(
    config,
    test_dataloader: torch.utils.data.DataLoader,
    client_dataloaders: dict[int, torch.utils.data.DataLoader],
    device,
    data_name: str,
    model_name: str,
):

    x_sample, y_sample = next(((x, y) for (x, y) in test_dataloader))
    input_features = x_sample.shape[1]
    output_features = y_sample.shape[1]

    MAX_THETA = 181  # Degree threshold for alignment

    # Initialize model
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

    # Initialize logging
    writer = SummaryWriter(get_logging_dir(model_name, data_name))

    # Initialize loss function
    loss_fn = get_loss_fn(config.loss_fn)

    for communication_round in tqdm(
        range(config.communication_rounds), desc="Communication Rounds"
    ):
        is_aligning_round = communication_round >= config.communication_rounds_training

        # Initialize client models
        client_models = {
            client_id: global_model.get_client_model(
                client_id, config.add_noise_in_training and not is_aligning_round
            )
            for client_id in range(config.num_clients)
        }
        optimizers = {
            client_id: get_optimizer(
                config.optimizer,
                config.learning_rate,
                client_models[client_id].parameters(),
            )
            for client_id in range(config.num_clients)
        }

        for model in client_models.values():
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
                writer.add_scalar(
                    f"Loss/train/client{client_id}",
                    epoch_loss,
                    communication_round * config.client_epochs + epoch,
                )

        # Aggregate global model
        global_model = SmartAttentionLayer.get_global_model(
            list(client_models.values()),
            config.similarity_threshold_in_degree if is_aligning_round else MAX_THETA,
            method=config.aligning_method if is_aligning_round else "combine",
        )

        # Evaluate model
        y_hats, ys = utils.evaluation.evaluate(
            global_model, test_dataloader, from_logits=True, return_outputs_only=True
        )
        writer.add_scalar(
            "Loss/test",
            loss_fn(y_hats.to(torch.float), ys.to(torch.float)).item(),
            communication_round * config.client_epochs,
        )
        writer.add_scalar(
            "total_params",
            summary(global_model, verbose=0).total_params,
            communication_round * config.client_epochs,
        )

    # Final evaluation
    metrics = utils.evaluation.evaluate(global_model, test_dataloader, from_logits=True)
    writer.add_hparams(dict(metrics), {}, run_name=".")
    writer.add_text("hparams", json.dumps(config.model_dump(), indent=4))
    writer.add_text(
        "Model Summary",
        str(summary(global_model, input_size=(1, input_features), verbose=0)),
    )
    dummy_input = torch.randn(1, input_features)  # Example input
    writer.add_graph(global_model, dummy_input)
    writer.flush()
    writer.close()
