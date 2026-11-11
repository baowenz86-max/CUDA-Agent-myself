import torch


class Model(torch.nn.Module):
    def __init__(self, in_features, out_features, bias):
        super(Model, self).__init__()
        self.batch_norm = torch.nn.BatchNorm2d(in_features)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.tanh(x)
        return x


batch_size = 32
in_channels = 3
height = 512
width = 256


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [3, 16, True]