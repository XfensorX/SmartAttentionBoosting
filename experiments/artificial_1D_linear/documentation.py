import io

import matplotlib.pyplot as plt
import torch
import torch.utils.tensorboard
import torchvision
from PIL import Image

from data import artificial_1D_linear as data


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
        color=(0 / 255, 51 / 255, 102 / 255),
        label="Original Data (test)",
    )
    ax.plot(
        x_train,
        y_train,
        color=(0 / 255, 128 / 255, 128 / 255),
        label="Original Data (train)",
    )
    ax.plot(
        x_test,
        y_hat_test,
        color=(255 / 255, 128 / 255, 0 / 255),
        label=f"{name} prediction (test)",
    )

    ax.plot(
        x_train,
        y_hat_train,
        color=(220 / 255, 20 / 255, 60 / 255),
        label=f"{name} prediction (train)",
    )
    ax.set_title("Prediction Plot")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    image = Image.open(buf)

    summary_writer.add_image(
        "Prediction Plot", torchvision.transforms.ToTensor()(image), 0
    )
