import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.log_softmax(x, dim=1)
        x = torch.triu(x)
        return x


batch_size = 512


def get_inputs():
    return [torch.randn(batch_size, 64, 100, 100)]


def get_init_inputs():
    return []