import time
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torch.utils.tensorboard

from data import artificial_1D_linear as data
from utils.general import figure_to_tensor_image

COLORS = [
    (0 / 255, 51 / 255, 102 / 255),  # Navy Blue
    (0 / 255, 128 / 255, 128 / 255),  # Teal
    (255 / 255, 128 / 255, 0 / 255),  # Orange
    (220 / 255, 20 / 255, 60 / 255),  # Crimson
    (34 / 255, 139 / 255, 34 / 255),  # Forest Green
    (128 / 255, 0 / 255, 128 / 255),  # Purple
    (255 / 255, 215 / 255, 0 / 255),  # Gold
    (70 / 255, 130 / 255, 180 / 255),  # Steel Blue
    (199 / 255, 21 / 255, 133 / 255),  # Medium Violet Red
    (255 / 255, 69 / 255, 0 / 255),  # Red-Orange
    (47 / 255, 79 / 255, 79 / 255),  # Dark Slate Gray
    (0 / 255, 191 / 255, 255 / 255),  # Deep Sky Blue
    (218 / 255, 112 / 255, 214 / 255),  # Orchid
    (154 / 255, 205 / 255, 50 / 255),  # Yellow-Green
    (160 / 255, 82 / 255, 45 / 255),  # Sienna
    (255 / 255, 99 / 255, 71 / 255),  # Tomato
    (210 / 255, 105 / 255, 30 / 255),  # Chocolate
    (46 / 255, 139 / 255, 87 / 255),  # Sea Green
    (138 / 255, 43 / 255, 226 / 255),  # Blue Violet
    (255 / 255, 182 / 255, 193 / 255),  # Light Pink
    (176 / 255, 196 / 255, 222 / 255),  # Light Steel Blue
    (152 / 255, 251 / 255, 152 / 255),  # Pale Green
    (255 / 255, 160 / 255, 122 / 255),  # Light Salmon
    (72 / 255, 61 / 255, 139 / 255),  # Dark Slate Blue
    (245 / 255, 222 / 255, 179 / 255),  # Wheat
    (250 / 255, 128 / 255, 114 / 255),  # Salmon
    (127 / 255, 255 / 255, 212 / 255),  # Aquamarine
    (255 / 255, 228 / 255, 196 / 255),  # Bisque
    (64 / 255, 224 / 255, 208 / 255),  # Turquoise
    (176 / 255, 224 / 255, 230 / 255),  # Powder Blue
]


def plot_predictions(
    model, name: str, summary_writer: torch.utils.tensorboard.writer.SummaryWriter
):
    model.eval()

    x_test, y_test = data.get_data("test")
    x_train, y_train = data.get_data("train")

    fig, ax = plt.subplots(figsize=(12, 5))

    y_hat_test = model(x_test.reshape(-1, 1)).detach()
    y_hat_train = model(x_train.reshape(-1, 1)).detach()

    ax.plot(
        x_test,
        y_test,
        color=COLORS[0],
        label="Original Data (test)",
    )
    ax.plot(
        x_train,
        y_train,
        color=COLORS[1],
        label="Original Data (train)",
    )
    ax.plot(
        x_test,
        y_hat_test,
        color=COLORS[2],
        label=f"{name} prediction (test)",
    )

    ax.plot(
        x_train,
        y_hat_train,
        color=COLORS[3],
        label=f"{name} prediction (train)",
    )
    ax.set_title("Prediction Plot")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()

    image_tensor = figure_to_tensor_image(fig)
    summary_writer.add_image("Prediction Plot", image_tensor, 0)


def plot_data_split(
    data_loaders: list[torch.utils.data.DataLoader[Tuple[torch.Tensor, ...]]],
    summary_writer: torch.utils.tensorboard.writer.SummaryWriter,
):

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, subset in enumerate(data_loaders):
        x_vals, y_vals = zip(
            *sorted(
                [subset.dataset.dataset[idx] for idx in subset.dataset.indices],
                key=lambda x: x[0],
            )
        )
        plt.plot(x_vals, y_vals, color=COLORS[i], label=f"Client {i+1} Data")

    ax.set_xlabel("X values")
    ax.set_ylabel("Y values")
    ax.set_title("Splitted Dataset Visualization")
    ax.legend()
    ax.grid()

    fig.tight_layout()

    image_tensor = figure_to_tensor_image(fig)
    summary_writer.add_image("Data Split", image_tensor, 0)


def evaluate(model):
    model.eval()
    test_dataloader = data.get_dataloader("test")
    assert 1 == len(test_dataloader)

    for x, y in test_dataloader:
        return torch.nn.MSELoss()(model(x), y).item()


def get_logging_dir(name: str):
    return f"../../logs/artificial_1D_linear/{name}/{time.strftime('%m-%d-%H-%M-%S', time.localtime())}"


def plot_dataset():
    plt.plot(*data.get_data("train"), label="Training Data")
    plt.plot(*data.get_data("test"), label="Test Data")
    plt.title("1D Artificial Dataset")
    plt.xlabel("Input Value")
    plt.ylabel("Output Value")
    plt.legend()
    plt.grid()
    plt.show()
