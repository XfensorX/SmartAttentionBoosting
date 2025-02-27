from typing import Optional, Tuple

import torch


class StandardScaler:
    def __init__(self) -> None:
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

    def fit(self, vector: torch.Tensor) -> None:
        self.mean = vector.mean(0, keepdim=True)
        self.std = vector.std(0, unbiased=False, keepdim=True)

    def transform(self, vector: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("fit() must be called before transform()")
        return (vector - self.mean) / self.std


def split_dataset_into_subsets(
    dataset: torch.utils.data.TensorDataset,
    no_clients: int,
    client_distribution: str,
    seed: int = 42,
):
    """client_distribution:
    - 'random': randomly distributed onto clients
    - 'interval': input feature space is split into equally sized intervals, based on dataset_order
    """
    subsets: list[torch.utils.data.Subset[Tuple[torch.Tensor, ...]]] = []

    if client_distribution == "random":
        subsets = torch.utils.data.random_split(
            dataset,
            [1 / no_clients] * no_clients,
            generator=torch.Generator().manual_seed(seed),
        )

    elif client_distribution == "interval":
        indices = list(range(len(dataset)))
        remainder = len(dataset) % no_clients
        interval_length = len(dataset) // no_clients

        start_idx = 0
        for i in range(no_clients):
            end = start_idx + interval_length + (1 if i < remainder else 0)
            subsets.append(torch.utils.data.Subset(dataset, indices[start_idx:end]))
            start_idx = end
    else:
        raise ValueError(
            f"Wrong option for client_distribution. {client_distribution} not implemented"
        )

    return subsets
