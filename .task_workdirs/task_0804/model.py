import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x1, x2):
        return torch.stack([x1, x2])


batch_size = 512


def get_inputs():
    x1 = torch.randn(batch_size, 3, 512, 512)
    x2 = torch.randn(batch_size, 3, 512, 512)
    return [x1, x2]


def get_init_inputs():
    return []  # No parameters needed for this forward function