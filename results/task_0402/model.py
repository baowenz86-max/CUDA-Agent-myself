import torch


class Model(torch.nn.Module):
    def __init__(self, dim):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.mean(x, dim=self.dim)


batch_size = 512
channels = 3
height = 1024
width = 1024


def get_inputs():
    # Create a random tensor of size [batch_size, channels, height, width]
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    # The dim argument for torch.mean
    return [0]