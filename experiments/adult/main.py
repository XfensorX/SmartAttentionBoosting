import argparse

import torch

import data.adult
from training.train_basic_nn import train_basic_nn
from training.train_fed_avg import train_fed_avg
from training.train_smart_attention import train_smart_attention
from utils.Config import Config

BATCH_SIZE = 256
DEVICE = torch.device("cpu")


def run_basic_nn_experiment(config: Config, model_name: str):
    test_dataloader = torch.utils.data.DataLoader(
        data.adult.get_dataset("test"), batch_size=BATCH_SIZE, shuffle=False
    )
    train_dataloader = torch.utils.data.DataLoader(
        data.adult.get_dataset("train"), batch_size=BATCH_SIZE, shuffle=True
    )
    train_basic_nn(
        config, train_dataloader, test_dataloader, DEVICE, "adult", model_name
    )


def run_fed_avg_experiment(config: Config, model_name: str):
    test_dataloader = torch.utils.data.DataLoader(
        data.adult.get_dataset("test"), batch_size=BATCH_SIZE, shuffle=False
    )

    client_dataloaders = data.adult.get_client_train_dataloaders(
        config.num_clients,
        config.client_data_distribution,
        BATCH_SIZE,
        True,
        sort_by="age",
    )

    train_fed_avg(
        config, test_dataloader, client_dataloaders, DEVICE, "adult", model_name
    )


def run_smart_attention_experiment(config: Config, model_name: str):

    test_dataloader = torch.utils.data.DataLoader(
        data.adult.get_dataset("test"), batch_size=BATCH_SIZE, shuffle=False
    )
    client_dataloaders = data.adult.get_client_train_dataloaders(
        config.num_clients,
        config.client_data_distribution,
        BATCH_SIZE,
        True,
        sort_by="age",
    )

    train_smart_attention(
        config, test_dataloader, client_dataloaders, DEVICE, "adult", model_name
    )


def run_attention_boosting_experiment(config: Config, model_name: str):
    raise NotImplementedError()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Training Script")
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        required=True,
        help="Paths to one or more YAML configuration files",
    )
    args = parser.parse_args()

    for config_path in args.config:
        print(f"\n🔹 Running experiment with {config_path}...\n")

        config = Config.load(config_path)

        model_name = config_path.replace(".yaml", "")

        torch.random.manual_seed(42)
        if config.network_type == "basic_nn":
            run_basic_nn_experiment(config, model_name)
        elif config.network_type == "fed_avg":
            run_fed_avg_experiment(config, model_name)
        elif config.network_type == "smart_attention":
            run_smart_attention_experiment(config, model_name)
        elif config.network_type == "attention_boosting":
            run_attention_boosting_experiment(config, model_name)
        else:
            raise ValueError(
                "Network type, " + config.network_type + " , not implemented."
            )
