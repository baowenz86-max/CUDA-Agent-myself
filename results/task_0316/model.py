import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = torch.sigmoid(x)
        x = torch.multiply(x, 2)  # replace 2 with a tensor or a variable
        return x


batch_size = 512
channels = 3
height, width = 512, 512  # Increased spatial dimensions to increase computation


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)  # adjust size according to your needs
    return [x]


def get_init_inputs():
    return []