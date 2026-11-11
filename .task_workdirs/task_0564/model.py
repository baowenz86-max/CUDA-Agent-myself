import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.max(x, dim=1)[0]  # Apply torch.max operation
        x = torch.log1p(x)  # Apply torch.log1p operation
        return x


batch_size = 512
channel_size = 10
height = 256
width = 256


def get_inputs():
    # Assuming the input tensor has the shape [batch_size, channel_size, height, width]
    x = torch.randn(batch_size, channel_size, height, width)
    return [x]


def get_init_inputs():
    # Since the model doesn't require any parameters, return an empty list
    return []