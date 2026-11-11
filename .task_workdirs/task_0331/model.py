import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_features, bias_shape):
        super().__init__()
        self.fc = nn.Linear(in_features, in_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.fc(x)
        x = F.gelu(x)
        threshold = torch.tensor(0.5)
        x = torch.gt(x, threshold)
        x = x + self.bias
        return x


batch_size = 128
in_features = 1536  # Increased from 512 to increase computation
bias_shape = (in_features, 1, 1)


def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, bias_shape]