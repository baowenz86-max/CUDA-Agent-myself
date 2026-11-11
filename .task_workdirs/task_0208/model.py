import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(in_features)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.log_softmax(x)
        return x


batch_size = 512
in_features = 16
num_classes = 10
height, width = 64, 64  # Increased from 32x32 to 64x64


def get_inputs():
    return [torch.randn(batch_size, in_features, height, width)]


def get_init_inputs():
    return [in_features, num_classes]