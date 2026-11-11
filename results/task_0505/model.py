import torch


class Model(torch.nn.Module):
    def __init__(self, feature_size, num_groups, eps=1e-5, affine=True):
        super().__init__()
        self.group_norm = torch.nn.GroupNorm(num_groups=num_groups, num_channels=feature_size, eps=eps, affine=affine)

    def forward(self, x):
        x = self.group_norm(x)
        return x


feature_size = 512
num_groups = 32


def get_inputs():
    return [torch.randn(16, feature_size, 256, 256)]  # Increased spatial dimensions to 256x256


def get_init_inputs():
    return [feature_size, num_groups]