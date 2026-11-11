import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, has_nan):
        super().__init__()
        self.has_nan = has_nan
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        if self.has_nan:
            x = torch.isnan(x)
        else:
            x = self.relu6(x)
        return x


# Hyperparameters
batch_size = 512


def get_inputs():
    x = torch.randn(batch_size, 32, 256, 256)  # Increased spatial dimensions to 256x256
    return [x]


def get_init_inputs():
    return [torch.tensor(0.0, dtype=torch.float32)]