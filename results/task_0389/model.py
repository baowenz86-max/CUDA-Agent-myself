import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        # Assuming `num_classes` is the number of channels in the input feature tensor
        self.batchnorm = nn.BatchNorm2d(num_features)
        self.softmax = nn.Softmax(dim=1)  # Softmax along the channel dimension

    def forward(self, x):
        x = self.batchnorm(x)
        x = self.softmax(x)
        return x


def get_inputs():
    # Increased spatial dimensions to 100x100 to increase computation time
    x = torch.randn(16, 512, 100, 100)  # Batch size 16, 512 channels, 100x100 spatial dimensions
    return [x]


def get_init_inputs():
    # Assuming the number of channels in the input feature tensor is 512
    return [512, 512]  # 512 channels, 512 as it's also the number of output features for the Softmax