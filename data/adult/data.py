import pandas as pd
from typing import Tuple
import torch

from utils.data import StandardScaler, split_dataset_into_subsets


def get_data(split: str):
    # TODO: Refactor
    adult_header = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "earning",
    ]
    data = pd.read_csv(
        "../../data/adult/adult.data", names=adult_header, index_col=False
    )
    test_data = pd.read_csv(
        "../../data/adult/adult.test",
        names=adult_header,
        index_col=False,
        dtype=data.dtypes.to_dict(),
        header=0,
    )
    train_data_len = len(data)
    assert all(test_data.dtypes == data.dtypes)

    data = pd.concat([data, test_data])

    assert all(data == data.dropna())

    for header in adult_header:
        if data[header].dtype == "object":
            data[header] = data[header].str.strip()

    data["earning"] = data["earning"].str.replace("<=50K.", "<=50K")
    data["earning"] = data["earning"].str.replace(">50K.", ">50K")

    dummies = pd.get_dummies(data, prefix_sep="_")
    assert set(adult_header) == set(
        [words[0] for words in dummies.columns.str.split("_")]
    )
    dummies["sex"] = dummies["sex_Male"].astype(int)
    dummies.drop(["sex_Male", "sex_Female"], inplace=True, axis=1)

    Y = dummies["earning_<=50K"].astype(int)
    dummies.drop(["earning_<=50K", "earning_>50K"], inplace=True, axis=1)
    X = dummies.astype(float)

    Y = torch.tensor(Y.values).unsqueeze(1)
    X = torch.tensor(X.values, dtype=torch.float32)

    X, X_test = X[0:train_data_len], X[train_data_len:]
    Y, Y_test = Y[0:train_data_len], Y[train_data_len:]

    if split == "train":
        return X, Y
    elif split == "validation":
        raise ValueError("Was not explicitly provided.")
    elif split == "test":
        return X_test, Y_test
    else:
        raise ValueError("Split not found")


def get_dataset(split: str, seed: int = 42) -> torch.utils.data.TensorDataset:
    torch.manual_seed(seed)
    X_raw, Y = get_data("train")
    X_test, Y_test = get_data("test")

    scaler = StandardScaler()
    scaler.fit(X_raw)

    if split == "train":
        return torch.utils.data.TensorDataset(scaler.transform(X_raw), Y)

    elif split == "test":

        return torch.utils.data.TensorDataset(scaler.transform(X_test), Y_test)

    else:
        raise ValueError("Not Implemented this split")


def get_client_train_dataloaders(
    no_clients: int,
    client_distribution: str,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 42,
) -> dict[int, torch.utils.data.DataLoader[Tuple[torch.Tensor, ...]]]:
    """client_distribution:
    - 'random': randomly distributed onto clients
    - 'interval': input feature space is split into equally sized intervals
    """
    dataset = get_dataset("train")

    subsets = split_dataset_into_subsets(dataset, no_clients, client_distribution, seed)

    return {
        i: torch.utils.data.DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        for i, subset in enumerate(subsets)
    }
