import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.expm1(x)
        return x


batch_size = 512
channels = 3
height = 320
width = 320


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []