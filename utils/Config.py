from typing import Literal
import yaml
from pydantic import BaseModel, Field


class Config(BaseModel):
    network_type: str = Field(...)
    optimizer: Literal["adam"] = Field(...)
    learning_rate: float = Field(..., gt=0)
    loss_fn: Literal["mse", "bce_with_logits"] = Field(...)
    batch_norm: bool = Field(...)
    layer_norm: bool = Field(...)
    dropout_rate: float = Field(..., ge=0, le=1)
    client_epochs: int = Field(..., gt=0)
    num_clients: int | None = Field(..., gt=0)
    communication_rounds: int | None = Field(..., gt=0)
    client_data_distribution: str | None = Field(...)
    architecture: list[int] = Field(...)
    similarity_threshold_in_degree: int | None = Field(..., ge=0)
    aligning_method: Literal["combine", "average"] | None = Field(...)
    input_importance_network_architecture: list[int] | None = Field(...)
    client_importance_network_architecture: list[int] | None = Field(...)
    communication_rounds_training: int | None = Field(...)
    add_noise_in_training: bool | None = Field(...)
    @staticmethod
    def load(config_path: str):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return Config(**config)
