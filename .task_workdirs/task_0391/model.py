import torch


class Model(torch.nn.Module):
    def __init__(self, mask_shape, scatter_dim):
        super().__init__()
        self.mask_shape = mask_shape
        self.scatter_dim = scatter_dim

    def forward(self, mask, input_tensor, scatter_values):
        mask = mask.expand(*self.mask_shape)
        return torch.masked_scatter(input_tensor, mask, scatter_values)


batch_size = 16
channels = 3
height = 1024  # Increased height to increase computation
width = 2048   # Increased width to increase computation


def get_inputs():
    input_tensor = torch.randn(batch_size, channels, height, width)
    scatter_values = torch.randn(batch_size, channels, height, width // 2)

    # Efficient mask generation using tensor operations for large dimensions
    mask = torch.arange(width) % 2 == 0  # Shape: (width,)
    mask = mask.unsqueeze(0).repeat(height, 1)  # Shape: (height, width)
    mask = mask.unsqueeze(0).unsqueeze(0).repeat(batch_size, channels, 1, 1)  # Shape: (batch_size, channels, height, width)

    return [mask, input_tensor, scatter_values]


def get_init_inputs():
    mask_shape = (batch_size, channels, height, width)
    scatter_dim = 3

    return [mask_shape, scatter_dim]