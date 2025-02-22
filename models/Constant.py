from typing import Any

import torch


class Constant(torch.nn.Module):
    def __init__(self, value: Any, out_dim: tuple[int]):
        super().__init__()
        self.value = value
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return torch.full((batch_size,) + self.out_dim, self.value)
