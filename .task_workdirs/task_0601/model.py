import torch


class Model(torch.nn.Module):
    def __init__(self, cat_dim):
        super(Model, self).__init__()
        self.cat_dim = cat_dim

    def forward(self, x1, x2):
        x1 = torch.cat((x1, x2), dim=self.cat_dim)
        x1 = torch.rot90(x1, k=1, dims=(2, 3))
        return x1


batch_size = 512
channels = 3
height = 256
width = 256


def get_inputs():
    x1 = torch.randn(batch_size, channels, height, width)
    x2 = torch.randn(batch_size, channels, height, width)
    return [x1, x2]


def get_init_inputs():
    return [1]  # The dimension to concatenate along