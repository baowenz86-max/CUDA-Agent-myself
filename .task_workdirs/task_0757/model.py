import torch


class Model(torch.nn.Module):
    def __init__(self, num_inputs, num_classes):
        super(Model, self).__init__()

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = torch.ne(x, 0)
        return x


batch_size = 512
channels = 32
height = 128
width = 128


def get_inputs():
    x1 = torch.randn(batch_size, channels, height, width)
    x2 = torch.randn(batch_size, channels, height, width)
    return [x1, x2]


def get_init_inputs():
    num_inputs = 2
    num_classes = 1
    return [num_inputs, num_classes]