import io
from dataclasses import asdict, dataclass
from functools import cached_property

import torch
import torchvision
from PIL import Image


def add_bias_node(tensor: torch.Tensor) -> torch.Tensor:
    batch_size = tensor.shape[0]
    bias = torch.ones((batch_size, 1), dtype=tensor.dtype, device=tensor.device)
    return torch.cat((bias, tensor), dim=1)


def DataclassWithCachedProperties(name: str = "", not_shown: list[str] | None = None):
    def wrapper(cls):

        cls = dataclass()(cls)

        def __repr__(self):
            return name + str({k: f"{v:.3f}" for k, v in dict(self).items()})

        def __iter__(self):
            yield from self.__iter_all_initialized_properties()
            yield from self.__iter_all_cached_properties()

        def __iter_all_cached_properties(self):
            for prop in dir(self):
                if isinstance(getattr(type(self), prop, None), cached_property):
                    yield prop, getattr(self, prop)

        def __iter_all_initialized_properties(self):
            for k, v in asdict(self).items():
                if not_shown is None or k not in not_shown:
                    yield k, v

        # Attach methods to the class
        cls.__repr__ = __repr__
        cls.__iter_all_cached_properties = __iter_all_cached_properties
        cls.__iter_all_initialized_properties = __iter_all_initialized_properties
        cls.__iter__ = __iter__

        return cls

    return wrapper


def figure_to_tensor_image(matplotlib_figure) -> torch.Tensor:
    buf = io.BytesIO()
    matplotlib_figure.savefig(buf, format="png")
    buf.seek(0)

    image = Image.open(buf)
    return torchvision.transforms.ToTensor()(image)
