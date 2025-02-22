from dataclasses import asdict, dataclass
from typing import Callable

import torch
import torch.nn as nn


class DenseNetwork(torch.nn.Module):
    @dataclass
    class Config:
        input_nodes: int
        hidden_layers: list[int]
        out_nodes: int
        get_activation: Callable[[], nn.Module] = torch.nn.ReLU
        use_batch_norm: bool = False
        use_layer_norm: bool = False
        dropout_rate: float = 0.0

        def __iter__(self):
            yield from asdict(self).items()

    def __init__(self, config: "DenseNetwork.Config"):
        super().__init__()

        layers: list[nn.Module] = []
        prev_nodes = config.input_nodes
        is_input_layer: bool = True

        for hidden_nodes in config.hidden_layers:
            layers.append(
                dense_inception_module(
                    prev_nodes,
                    hidden_nodes,
                    config.use_batch_norm and not is_input_layer,
                    config.use_layer_norm and not is_input_layer,
                    config.dropout_rate,
                    config.get_activation(),
                )
            )
            prev_nodes = hidden_nodes
            is_input_layer = False

        layers.append(nn.Linear(prev_nodes, config.out_nodes))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def dense_inception_module(
    in_nodes: int,
    out_nodes: int,
    use_batch_norm: bool,
    use_layer_norm: bool,
    dropout_rate: float,
    activation: nn.Module,
):
    inception_layers: list[nn.Module] = []
    if use_batch_norm:
        inception_layers.append(nn.BatchNorm1d(in_nodes))
    if use_layer_norm:
        inception_layers.append(nn.LayerNorm(in_nodes))

    inception_layers.append(nn.Linear(in_nodes, out_nodes))
    inception_layers.append(activation)

    if dropout_rate > 0.0:
        inception_layers.append(nn.Dropout(dropout_rate))

    return nn.Sequential(*inception_layers)
