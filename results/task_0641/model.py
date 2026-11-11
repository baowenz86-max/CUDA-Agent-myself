import torch


class Model(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_sum = torch.sum(x, dim=self.dim)
        x_nonzero_count = torch.count_nonzero(x, dim=self.dim)
        return x_sum, x_nonzero_count


batch_size = 512
height, width = 512, 512  # Increased spatial dimensions to 512x512
channels = 3
x = torch.randn(batch_size, channels, height, width)


def get_inputs():
    return [x]


def get_init_inputs():
    return [2]  # Changed to sum over height dimension (now size 512)