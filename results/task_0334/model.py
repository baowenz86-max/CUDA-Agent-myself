import torch


class Model(torch.nn.Module):
    def __init__(self, tensor_shape, dim):
        super().__init__()
        self.tensor = torch.nn.Parameter(torch.randn(*tensor_shape))
        self.dim = dim

    def forward(self, x):
        x = torch.cumprod(x, self.dim)
        return x


# Hyperparameters to control input shape
input_dim1 = 8192  # Increased from 512 to increase computation
input_dim2 = 3
input_dim3 = 4


def get_inputs():
    return [torch.randn(input_dim1, input_dim2, input_dim3)]


def get_init_inputs():
    return [get_inputs()[0].shape, 0]