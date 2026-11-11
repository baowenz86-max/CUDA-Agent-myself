import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=True)
        x = torch.mean(x, dim=2, keepdim=True)
        return x


def get_inputs():
    return [torch.randn(32, 64, 256, 2048)]


def get_init_inputs():
    return []