import torch
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self, input_size, pad_width, pad_value):
        super(Model, self).__init__()
        self.input_size = input_size
        self.pad_width = pad_width
        self.pad_value = pad_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.view_as_real(x)  # Converts complex number to real number representation
        x = F.pad(x, self.pad_width, value=self.pad_value)  # Pads the tensor
        return x


input_size = 1024  # Hyperparameter controlling input dimension


def get_inputs():
    # Generate random tensor with complex numbers, adjusted to input_size
    batch_size = 64
    x = torch.randn(batch_size, input_size, input_size) + 1j * torch.randn(batch_size, input_size, input_size)
    return [x]


def get_init_inputs():
    # Update input_size to match the actual input dimension
    pad_width = (1, 1, 1, 1)
    pad_value = 0
    return [input_size, pad_width, pad_value]