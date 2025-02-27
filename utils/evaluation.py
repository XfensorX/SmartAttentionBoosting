from functools import cached_property
from typing import Tuple

import torch
from torchinfo import summary
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)

from .general import DataclassWithCachedProperties


@DataclassWithCachedProperties(
    "Metrics", not_shown=["inputs", "targets", "model_summary"]
)
class Metrics:
    inputs: torch.Tensor
    targets: torch.Tensor
    model_summary: str
    model_parameter_count: int

    @cached_property
    def bin_accuracy(self) -> float:
        return BinaryAccuracy()(self.inputs, self.targets).item()

    @cached_property
    def bin_precision(self) -> float:
        return BinaryPrecision()(self.inputs, self.targets).item()

    @cached_property
    def bin_recall(self) -> float:
        return BinaryRecall()(self.inputs, self.targets).item()

    @cached_property
    def bin_f1score(self) -> float:
        return BinaryF1Score()(self.inputs, self.targets).item()

    @cached_property
    def mse(self) -> float:
        return torch.nn.functional.mse_loss(
            self.inputs.float(), self.targets.float()
        ).item()


def evaluate(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader[Tuple[torch.Tensor, ...]],
    from_logits: bool = False,
    return_outputs_only: bool = False,
) -> Tuple[torch.Tensor] | Metrics:
    running_ys: list[torch.Tensor] = []
    running_yhats: list[torch.Tensor] = []

    model.eval()

    with torch.no_grad():
        for x, y in test_dataloader:
            y_hat = model(x)
            if from_logits and not return_outputs_only:
                y_hat = y_hat.sigmoid().round()
            running_ys.append(y)
            running_yhats.append(y_hat)

    ys = torch.vstack(running_ys)
    y_hats = torch.vstack(running_yhats)

    if return_outputs_only:
        return y_hats, ys

    return Metrics(
        inputs=y_hats,
        targets=ys,
        model_summary=get_model_summary(model),
        model_parameter_count=summary(model, verbose=0).total_params,
    )


def get_model_summary(model: torch.nn.Module) -> str:
    return str(model)
