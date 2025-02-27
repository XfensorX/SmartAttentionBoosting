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


def train_basic_nn(
    config: utils.Config,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    data_name: str,
    model_name: str,
):
    input_features, output_features = get_feature_sizes(test_dataloader)
    model = DenseNetwork(
        DenseNetwork.Config(
            input_features,
            config.architecture,
            output_features,
            torch.nn.ReLU,
            use_batch_norm=config.batch_norm,
            use_layer_norm=config.layer_norm,
            dropout_rate=config.dropout_rate,
        )
    ).to(device)

    writer = SummaryWriter(utils.general.get_logging_dir(model_name, data_name))
    loss_fn = get_loss_fn(config.loss_fn)

    optimizer = get_optimizer(
        config.optimizer, config.learning_rate, model.parameters()
    )

    for epoch in tqdm(range(config.client_epochs), desc="Training Basic NN"):
        epoch_loss = train_one_epoch(
            model, train_dataloader, optimizer, loss_fn, device
        )

        register_train_loss(writer, epoch_loss, epoch)
        register_test_loss(
            writer,
            model,
            loss_fn,
            epoch,
            test_dataloader,
        )

    final_evaluation(writer, model, test_dataloader, config)
