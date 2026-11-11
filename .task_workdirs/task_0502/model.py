import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = torch.matmul(x, y)
        z = torch.softmax(z, dim=0)
        return z


# Hyperparameters to control input dimensions
x_dim1, x_dim2 = 2048, 1024  # Increased matrix dimensions for more computation
y_dim2 = 2048  # y_dim1 must match x_dim2 for matrix multiplication


def get_inputs():
    x = torch.randn(x_dim1, x_dim2)
    y = torch.randn(x_dim2, y_dim2)
    return [x, y]


def get_init_inputs():
    return []