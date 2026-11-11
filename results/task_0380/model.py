import torch
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self, in_features):
        super(Model, self).__init__()
        self.in_features = in_features

    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=True)
        x = F.leaky_relu(x)
        return x


batch_size = 128  # Increased from 16 to 128 to increase runtime
in_features = 64
width = 512
height = 256


def get_inputs():
    x = torch.randn(batch_size, in_features, height, width)
    return [x]


def get_init_inputs():
    return [in_features]