import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        # Check for NaNs in the input tensor and return the indices
        nan_indices = torch.isnan(x)
        # Apply batch normalization
        x = self.batch_norm(x)
        return x, nan_indices


num_features = 64
batch_size = 128
sequence_length = 15360  # Increased from 1024 to increase computation


def get_inputs():
    return [torch.randn(batch_size, num_features, sequence_length)]


def get_init_inputs():
    return [num_features]