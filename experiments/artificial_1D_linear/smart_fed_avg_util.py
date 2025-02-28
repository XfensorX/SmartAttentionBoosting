from typing import Callable, Iterable, Tuple
import torch
import torch.utils.tensorboard
from tqdm import tqdm
from experiments.artificial_1D_linear.documentation import (
    evaluate,
    plot_predictions,
)


def train_client(
    client_no: int,
    client_model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader[Tuple[torch.Tensor, ...]],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    no_epochs: int,
    communication_round: int,
    writer: torch.utils.tensorboard.writer.SummaryWriter,
    learning_rate: float = 0.001,
    device: torch.device = torch.device("cpu"),
):

    client_model.to(device)

    client_model.train()
    optimizer = torch.optim.Adam(client_model.parameters(), lr=learning_rate)

    for epoch in tqdm(
        range(no_epochs), desc=f"Training client {client_no}", leave=False
    ):
        losses: list[float] = []
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = client_model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        writer.add_scalar(
            f"loss/client{client_no}",
            sum(losses) / len(losses),
            communication_round * no_epochs + epoch,
        )


def register_client_test_losses(
    clients: list[torch.nn.Module],
    client_ids: Iterable[int],
    writer: torch.utils.tensorboard.writer.SummaryWriter,
    communication_round: int,
    plot_client_predictions: bool = False,
    device: torch.device = torch.device("cpu"),
):
    for client, client_no in zip(clients, client_ids):
        writer.add_scalar(
            f"test_loss/client{client_no}",
            evaluate(client, device=device),
            communication_round,
        )

        if plot_client_predictions:
            plot_predictions(
                client,
                f"Client {client_no}",
                writer,
                epoch=communication_round,
                name_add_on=f" Client {client_no}",
                device=device,
            )
