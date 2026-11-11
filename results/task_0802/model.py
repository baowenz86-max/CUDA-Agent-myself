import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_classes, num_prelu_features, prelu_alpha_value):
        super().__init__()
        self.softmax = F.softmax
        self.prelu = nn.PReLU(num_parameters=num_prelu_features, init=prelu_alpha_value)

    def forward(self, x):
        x = self.softmax(x, dim=1)
        x = self.prelu(x)
        return x


num_classes = 10
num_prelu_features = 20
prelu_alpha_value = 0.25
batch_size = 512  # Increased batch size to ensure at least one dimension >=512


def get_inputs():
    return [
        torch.randn(batch_size, num_prelu_features, 128, 128)
    ]  # Input shape now (512, 20, 128, 128) to increase computation


def get_init_inputs():
    return [num_classes, num_prelu_features, prelu_alpha_value]