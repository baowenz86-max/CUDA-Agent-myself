import torch


class Model(torch.nn.Module):
    def __init__(self, mean_dim):
        super().__init__()
        self.mean_dim = mean_dim

    def forward(self, x):
        x = torch.gt(x, torch.mean(x, dim=self.mean_dim, keepdim=True))
        return x


batch_size = 16
channels = 3
height = 2048
width = 2048


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [2]