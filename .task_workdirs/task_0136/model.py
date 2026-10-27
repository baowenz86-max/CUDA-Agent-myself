import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(input_features, 10)

    def forward(self, x):
        x = self.sigmoid(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def get_inputs():
    batch_size, channels, height, width = 512, 3, 256, 256  # Increased height/width to 256
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    # Input features after pooling: 3 * (256/2) * (256/2) = 3*128*128
    input_features = 3 * 128 * 128
    return [input_features]