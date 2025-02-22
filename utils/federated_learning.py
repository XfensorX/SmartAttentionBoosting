import torch
import copy


def average_models(
    clients: list[torch.nn.Module], weights: list[float] | torch.Tensor | None = None
):
    """
    Averages the given models, optionally weighted by the given weights.
    The weights are relative, they do not have to be normalized.

    All models should have the same parameter names and shapes.
    """
    if not clients:
        raise ValueError("clients list cannot be empty")

    if weights is None:
        weights = torch.ones((len(clients),))
    assert len(weights) == len(clients)
    weights = torch.as_tensor(weights, dtype=torch.float32) / sum(weights)

    global_model = copy.deepcopy(clients[0])

    all_state_dicts = [client.state_dict() for client in clients]
    assert all(
        set(s.keys()) == set(all_state_dicts[0].keys()) for s in all_state_dicts
    ), "Not all models have the same parameters"

    global_model.load_state_dict(
        {
            param_name: torch.sum(
                torch.stack(
                    [
                        state[param_name] * weight
                        for state, weight in zip(all_state_dicts, weights)
                    ]
                ),
                dim=0,
            )
            for param_name in all_state_dicts[0]
        }
    )

    return global_model
