import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, padding, value):
        super().__init__()
        self.pad = nn.ConstantPad1d(padding, value)

    def forward(self, x):
        x = F.softmax(x, dim=1)
        x = self.pad(x)
        return x


batch_size = 64
num_features = 128
length = 8192  # Increased from 512 to 8192 to increase runtime
padding = (10, 10)
value = 0.0


def get_inputs():
    return [torch.randn(batch_size, num_features, length)]


def get_init_inputs():
    return [padding, value]