import torch


class Model(torch.nn.Module):
    def __init__(self, input_size, num_channels):
        super(Model, self).__init__()
        self.input_size = input_size
        self.num_channels = num_channels

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        x = torch.multiply(x, self.input_size)
        return x


batch_size = 16
x_channels = 8
y_channels = 4
height, width = 1024, 1024  # Increased from 512 to 1024 to increase computation


def get_inputs():
    x = torch.randn(batch_size, x_channels, height, width)
    y = torch.randn(batch_size, y_channels, height, width)
    return [x, y]


def get_init_inputs():
    return [16, 12]