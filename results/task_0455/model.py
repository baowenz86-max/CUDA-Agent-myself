import torch


class Model(torch.nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.input_dim = input_dim

    def forward(self, x):
        max_val, _ = torch.max(x, dim=self.input_dim)
        argmin_val = torch.argmin(x, dim=self.input_dim)
        return max_val, argmin_val


# Define hyperparameters for input shape
batch_size = 256  # Increased from 16 to increase computation
channels = 3
height = 512
width = 256


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    # Keep reducing over channels dimension (dimension 1)
    return [1]