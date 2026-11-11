import torch


class Model(torch.nn.Module):
    def __init__(self, fmod_divisor):
        super(Model, self).__init__()
        self.fmod_divisor = fmod_divisor

    def forward(self, x):
        x = torch.sum(x, dim=1)
        x = torch.fmod(x, self.fmod_divisor)
        return x


batch_size = 512
feature_dim = 2048  # Increased from 128 to increase computation
height, width = 32, 32


def get_inputs():
    return [torch.randn(batch_size, feature_dim, height, width)]


def get_init_inputs():
    return [10]