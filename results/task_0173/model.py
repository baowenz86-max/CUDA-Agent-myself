import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.einsum_input = "bhwc->bwh"  # einsum equation for reshaping input from [batch, height, width, channels] to [batch, width, height]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.einsum(self.einsum_input, x)  # apply einsum operation
        x = torch.abs(x)  # apply absolute function
        return x


def get_inputs():
    batch_size, height, width, channels = 256, 128, 1024, 3  # Increased batch_size from 64 to 256
    x = torch.randn(batch_size, height, width, channels)
    return [x]


def get_init_inputs():
    return []  # No additional initial arguments needed