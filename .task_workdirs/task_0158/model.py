import torch


class Model(torch.nn.Module):
    def __init__(self, input_size, mean_value):
        super().__init__()
        self.mean = mean_value

    def forward(self, x):
        x = torch.mean(x, dim=self.mean)
        x = x + self.mean
        return x


batch_size = 512
height, width = 512, 512  # Increased spatial dimensions to increase computation
channels = 3
input_size = (batch_size, channels, height, width)
mean_value = 1


def get_inputs():
    return [torch.randn(input_size)]


def get_init_inputs():
    return [input_size[1], mean_value]