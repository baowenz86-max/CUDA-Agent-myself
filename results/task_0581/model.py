import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        max_val, _ = torch.max(x, dim=1, keepdim=True)
        cloned = torch.clone(x)
        return max_val, cloned


def get_inputs():
    return [torch.randn(4096, 128, 512)]


def get_init_inputs():
    return []