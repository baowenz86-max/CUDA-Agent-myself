import torch


class Model(torch.nn.Module):
    def __init__(self, tensor_shape, mean, std):
        super().__init__()
        self.tensor = torch.nn.Parameter(torch.normal(mean=mean, std=std, size=tensor_shape))

    def forward(self, x):
        x = torch.max(x, self.tensor)
        return x


tensor_shape = (100, 100)
mean = 0.0
std = 1.0
batch_size = 512
channels = 42
height = 100
width = 100


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [tensor_shape, mean, std]