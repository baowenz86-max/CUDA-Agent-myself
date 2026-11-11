import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.sync_batchnorm = nn.SyncBatchNorm(num_features)
        self.num_classes = num_classes

    def forward(self, x, elements):
        x = self.sync_batchnorm(x)
        mask = torch.isin(x, elements)
        return x, mask


batch_size = 512
num_features = 32
height, width = 36, 36
num_classes = 5


def get_inputs():
    x = torch.randn(batch_size, num_features, height, width)
    elements = torch.tensor([0.0, 1.0, -1.0])
    return [x, elements]


def get_init_inputs():
    return [num_features, num_classes]