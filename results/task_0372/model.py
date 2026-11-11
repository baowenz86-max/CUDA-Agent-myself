import torch
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = torch.abs(x)
        x = F.hardsigmoid(x)
        return x


# Hyperparameters determining input shape
batch_size = 512  # At least one dimension >=512


def get_inputs():
    return [torch.randn(batch_size, 32, 128, 128)]  # Increased spatial dimensions to 128x128


def get_init_inputs():
    return []