import torch


class Model(torch.nn.Module):
    def __init__(self, in_channels, bias_shape):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.bias = torch.nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.sigmoid(x)
        x = self.relu(x)
        x = x + self.bias
        return x


batch_size = 16
in_channels = 3
bias_shape = (in_channels, 1, 1)
height, width = 2048, 2048  # Increased spatial dimensions to increase computation


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, bias_shape]