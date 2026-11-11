import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=True)  # dim=1 because the input is [batch_size, in_features, height, width]
        x = self.relu(x)
        return x


in_features = 512  # Increased from 128 to increase computation per mean operation
batch_size = 512


def get_inputs():
    return [torch.randn(batch_size, in_features, 64, 64)]


def get_init_inputs():
    return [in_features]