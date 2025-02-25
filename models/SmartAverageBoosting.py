import torch

from models.SmartAverageLayer import SmartAverageLayer


class SmartAverageBoosting(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.boosting_layers: list[SmartAverageLayer] = []
        self.add_new_boosting_layer()

    def add_new_boosting_layer(self):
        self.back_boosting_layers.append(SmartAverageLayer(None))

    def forward(self, x: torch.Tensor):
        if self.back_boosting_layers:
            x = self.back_boosting_layers[0](x)

        for layer in self.back_boosting_layers[1:]:
            x = x + layer(x)

        return x
