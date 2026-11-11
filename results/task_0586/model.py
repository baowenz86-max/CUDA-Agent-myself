import torch


class Model(torch.nn.Module):
    def __init__(self, power_factor: int):
        super().__init__()
        self.power_factor = power_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.abs(x)
        x = torch.pow(x, self.power_factor)
        return x


batch_size = 2048  # Increased to 2048 to increase computation
width = 512
height = 256
power_factor = 3


def get_inputs():
    x = torch.randn(batch_size, height, width)
    return [x]


def get_init_inputs():
    return [power_factor]