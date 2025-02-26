import torch


from typing import Callable


ActivationFunction = Callable[[torch.Tensor], torch.Tensor]
