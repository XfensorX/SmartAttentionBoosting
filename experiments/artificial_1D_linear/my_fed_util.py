from typing import Callable, Iterable, Tuple
import torch
import torch.utils.tensorboard
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
):

    client_model.train()
    optimizer = torch.optim.Adam(client_model.parameters(), lr=learning_rate)

    for epoch in range(no_epochs):
        losses: list[float] = []
        for x, y in data_loader:
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
):
    for client, client_no in zip(clients, client_ids):
        writer.add_scalar(
            f"test_loss/client{client_no}", evaluate(client), communication_round
        )

        ###!!!!!!!!!!!!! NOTE: remove
        plot_predictions(
            client,
            f"Client {client_no}",
            writer,
            epoch=communication_round,
            name_add_on=f" Client {client_no}",
        )
        #############
