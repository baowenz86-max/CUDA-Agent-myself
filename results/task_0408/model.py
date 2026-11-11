import torch


class Model(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_max, _ = torch.max(x, dim=self.dim, keepdim=True)
        x_prod = torch.prod(x, dim=self.dim, keepdim=True)
        return x_max, x_prod


batch_size = 512
channels = 3
height = 2048  # Increased from 64 to increase computation on dim=2
width = 128


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [2]  # Keep dimension as 2 (height dimension)