import torch


class Model(torch.nn.Module):
    def __init__(self, scalar):
        super(Model, self).__init__()
        self.scalar = torch.nn.Parameter(torch.tensor(scalar))

    def forward(self, x):
        max_val, _ = torch.max(x, dim=1, keepdim=True)
        scalar_tensor = torch.scalar_tensor(self.scalar.item())
        return max_val * scalar_tensor


def get_inputs():
    batch_size, channels, height, width = 512, 64, 128, 128
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [0.5]