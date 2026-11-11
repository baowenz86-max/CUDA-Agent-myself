import torch


class Model(torch.nn.Module):
    def __init__(self, cat_dim):
        super().__init__()
        self.cat_dim = cat_dim

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=self.cat_dim)
        x = torch.ceil(x)
        return x


batch_size = 512
channels = 3
height = 256
width = 256


def get_inputs():
    x1 = torch.randn(batch_size, channels, height, width)
    x2 = torch.randn(batch_size, channels, height, width)
    return [x1, x2]


def get_init_inputs():
    return [0]  # The dimension along which to concatenate