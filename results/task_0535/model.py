import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_values, _ = torch.cummax(x, dim=0)
        return max_values


input_dim = 20000  # Increased from 512 to increase computation


def get_inputs():
    x = torch.randn(input_dim, 50)  # Increased second dimension from 3 to 50
    return [x]


def get_init_inputs():
    return []