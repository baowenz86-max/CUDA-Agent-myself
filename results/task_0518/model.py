import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.softmin = nn.Softmin(dim=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmin(x)


# Define hyperparameters for input shape
batch_size = 64
sequence_length = 70000  # Increased to increase computation time
features = 10


# Define a function to get inputs for the forward method
def get_inputs():
    # Input tensor shape: [batch_size, sequence_length, features] with increased sequence_length
    return [torch.randn(batch_size, sequence_length, features)]


# Define a function to get the initialization parameters
def get_init_inputs():
    return [1]