from typing import Optional

import pandas as pd
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


def adult(split: str):
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

    data = pd.read_csv("data/adult/adult.data", names=adult_header, index_col=False)
    test_data = pd.read_csv(
        "data/adult/adult.test",
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
