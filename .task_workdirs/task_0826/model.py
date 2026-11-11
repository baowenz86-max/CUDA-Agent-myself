import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_clone = torch.clone(x)
        x_rotated = torch.rot90(x_clone, k=1, dims=(1, 2))
        return x_rotated


batch_size = 512
channels = 3
height = 320
width = 320


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []