import json
from typing import Callable, Literal, Tuple

import torch
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

import utils


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader[Tuple[torch.Tensor, ...]],
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
) -> float:
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


def get_feature_sizes(dataloader):
    x_sample, y_sample = next(((x, y) for (x, y) in dataloader))
    input_features = x_sample.shape[1]
    output_features = y_sample.shape[1]
    return input_features, output_features


def final_evaluation(
    writer: SummaryWriter,
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    config: utils.Config,
):
    input_features, _ = get_feature_sizes(test_dataloader)
    metrics = utils.evaluation.evaluate(model, test_dataloader, from_logits=True)
    writer.add_hparams(dict(metrics), {}, run_name=".")
    writer.add_text(
        "hparams", json.dumps(config.model_dump(exclude_none=True), indent=4)
    )
    writer.add_text(
        "Model Summary", str(summary(model, input_size=(1, input_features), verbose=0))
    )
    writer.add_graph(model, torch.randn(1, input_features))
    writer.flush()
    writer.close()


def register_test_loss(
    writer: SummaryWriter,
    model: torch.nn.Module,
    loss_fn,
    epoch: int,
    test_dataloader: torch.utils.data.DataLoader,
):

    y_hats, ys = utils.evaluation.evaluate(
        model, test_dataloader, from_logits=True, return_outputs_only=True
    )
    writer.add_scalar(
        "Loss/test",
        loss_fn(y_hats.to(torch.float), ys.to(torch.float)).item(),
        epoch,
    )
    writer.add_scalar(
        "total_params",
        summary(model, verbose=0).total_params,
        epoch,
    )


def register_train_loss(
    writer: SummaryWriter, loss: float, epoch: int, client_no: int | None = None
):
    if client_no is None:
        writer.add_scalar("Loss/train", loss, epoch)
    else:
        writer.add_scalar(f"Loss/train/client{client_no}", loss, epoch)
