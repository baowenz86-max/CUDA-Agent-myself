import torch


class Model(torch.nn.Module):
    def __init__(self, device):
        super(Model, self).__init__()
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        x = torch.abs(x)
        return x


# Hyperparameters to control input shape
dim1 = 10000
dim2 = 200


def get_inputs():
    x = torch.randn(dim1, dim2)  # Shape (10000, 200)
    return [x]


def get_init_inputs():
    return ["cpu"]