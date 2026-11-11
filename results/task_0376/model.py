import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        x = torch.selu(x)  # Apply SELU activation
        x = torch.sigmoid(x)  # Apply Sigmoid activation
        return x


# Define the function to generate input data
def get_inputs():
    # Generate random input data with increased batch size and feature dimension
    x = torch.randn(1024, 4096)  # Adjusted input shape: [1024, 4096]
    return [x]


# Define the function to get the initialization arguments
def get_init_inputs():
    in_features = 4096  # Increased input feature size
    out_features = 4096  # Increased output feature size
    return [in_features, out_features]