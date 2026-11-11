import torch


class Model(torch.nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()
        self.input_size = input_size

    def forward(self, x):
        mean = torch.mean(x, dim=self.input_size)
        equal = torch.equal(x, mean.expand_as(x))
        return mean, equal


# Hyperparameters determining input shape
batch_size = 16
channels = 3
height = 2048  # Increased to increase computation
width = 2048   # Increased to increase computation


def get_inputs():
    # Create random input tensor with shape (batch_size, channels, height, width)
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    # The input size can be 0, 1, 2, or 3, which corresponds to
    # the entire tensor, the batch dimension, the channels dimension,
    # or either the height or width dimension
    return [0]