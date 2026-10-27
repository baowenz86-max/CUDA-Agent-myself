import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, tensor, mean):
        super().__init__()
        self.tensor = torch.tensor(tensor)
        self.mean = mean

    def forward(self, x):
        return torch.tril(x, diagonal=self.tensor) + self.mean * torch.mean(x, dim=(2, 3), keepdim=True)


tensor = -2
mean = 0.5
batch_size = 512  # Increased batch size to 512 to ensure at least one dimension >=512


def get_inputs():
    return [torch.randn(batch_size, 3, 256, 256)]  # Increased spatial dimensions to 256x256


def get_init_inputs():
    return [tensor, mean]