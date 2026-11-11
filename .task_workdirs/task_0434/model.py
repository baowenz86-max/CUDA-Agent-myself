import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.abs(x)
        x = torch.cumsum(x, dim=0)
        return x


def get_inputs():
    return [torch.randn(1000, 3, 512, 256)]


def get_init_inputs():
    return []