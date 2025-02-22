from typing import Tuple
import torch


def _get_x_y(seed: int = 42):
    torch.manual_seed(seed)
    x = torch.linspace(-1, 1, 1000)
    y = torch.where(x > 0, -(((2 * x) ** 2)), 2 * x) + torch.randn((1000,)) * 0.1
    return x, y


def get_data(
    split: str, train_ratio: float = 0.8, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = _get_x_y()

    train_size = int(train_ratio * len(x))

    torch.manual_seed(seed)
    perm = torch.randperm(x.size(0))
    (train_idx, _), (test_idx, _) = perm[:train_size].sort(), perm[train_size:].sort()
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if split == "train":
        return x_train, y_train
    elif split == "valid":
        raise ValueError("Only train and test split available for this dataset")
    elif split == "test":
        return x_test, y_test
    else:
        raise ValueError(f"Split {split} not implemented.")


def get_dataset(
    split: str, train_ratio: float = 0.8, seed: int = 42
) -> torch.utils.data.TensorDataset:
    x, y = get_data(split, train_ratio, seed)
    return torch.utils.data.TensorDataset(x.reshape(-1, 1), y.reshape(-1, 1))


def get_dataloader(
    split: str,
    batch_size: int = -1,
    shuffle: bool = False,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> torch.utils.data.DataLoader[Tuple[torch.Tensor, ...]]:
    dataset = get_dataset(split, train_ratio, seed)
    if batch_size <= 0:
        batch_size = len(dataset)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _split_dataset_into_subsets(
    dataset: torch.utils.data.TensorDataset,
    no_clients: int,
    client_distribution: str,
    seed: int = 42,
):
    """client_distribution:
    - 'random': randomly distributed onto clients
    - 'interval': input feature space is split into equally sized intervals
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


def get_client_test_dataloaders(
    no_clients: int,
    client_distribution: str,
    batch_size: int = -1,
    shuffle: bool = False,
    train_ratio: float = 0.8,
    seed: int = 42,
):
    """client_distribution:
    - 'random': randomly distributed onto clients
    - 'interval': input feature space is split into equally sized intervals
    """
    dataset = get_dataset("train", train_ratio, seed)

    subsets = _split_dataset_into_subsets(
        dataset, no_clients, client_distribution, seed
    )

    return [
        torch.utils.data.DataLoader(
            subset,
            batch_size=len(subset) if batch_size <= 0 else batch_size,
            shuffle=shuffle,
        )
        for subset in subsets
    ]
