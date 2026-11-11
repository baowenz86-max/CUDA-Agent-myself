import torch


class Model(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return torch.cat((x1, x2), dim=self.dim)


def get_inputs():
    x1 = torch.randn(512, 64, 128, 128)  # Increased spatial dimensions to 128x128
    x2 = torch.randn(512, 64, 128, 128)
    return [x1, x2]


def get_init_inputs():
    return [1]  # Dim along which the tensors will be concatenated.