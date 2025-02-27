from typing import Callable, Literal, Tuple
import torch


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
