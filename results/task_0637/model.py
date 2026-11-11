import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        max_val, _ = torch.max(x, dim=1, keepdim=True)
        max_val, _ = torch.max(max_val, dim=2, keepdim=False)
        return max_val


def get_inputs():
    return [torch.randn(512, 32, 256, 256)]  # Increased spatial dimensions to 256x256


def get_init_inputs():
    return []